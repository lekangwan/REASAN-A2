"""对照原 MoE 源码检查导出动作、历史复位、关节映射与显式 PD。"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a2_adapter.low_level import A2LowLevel
from a2_adapter.robot import robot_contract


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = json.loads((ROOT/"assets/a2/SOURCE.json").read_text())
    for relative, entry in manifest["files"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == entry["sha256"], relative

    # 独立 CPU 进程中按文件加载 A2 网络，不安装或覆盖 REASAN 的同名包。
    for name in ("rsl_rl", "rsl_rl.modules"):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    source = ROOT/"upstream/a2_reference/rsl_rl/rsl_rl/modules"
    load_module("rsl_rl.modules.utils", source/"utils.py")
    original = load_module("rsl_rl.modules.actor_critic_moe_cts", source/"actor_critic_moe_cts.py")
    with contextlib.redirect_stdout(io.StringIO()):
        model = original.ActorCriticMoECTS(45, 263, 12, 2, 5, expert_num=8).eval()
    checkpoint = torch.load(ROOT/"assets/a2/model_103500.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    exported = torch.jit.load(str(ROOT/"assets/a2/student_model_103500.pt"), map_location="cpu").eval()
    torch.manual_seed(20261117)
    history = torch.zeros(2, 5, 45)
    maximum_error = 0.0
    with torch.inference_mode():
        for step in range(32):
            if step in (7, 19):
                untouched = history[1].clone()
                mask = torch.tensor([True, False])
                model.reset(mask)
                history[mask] = 0
                assert torch.equal(history[1], untouched)
            observation = torch.randn(2, 45)
            expected = model.act_inference(observation)
            actual, history = exported(observation, history)
            maximum_error = max(maximum_error, float((actual-expected).abs().max()))
            assert torch.isfinite(actual).all() and torch.equal(history, model.history)
    assert maximum_error <= 1e-6

    # 用已有原生仿真观测构造乱序关节，防止关节顺序碰巧相同而漏检。
    contract = robot_contract()
    with np.load(ROOT/"assets/a2/native_reference/trace.npz") as data:
        observation = torch.tensor(data["observation"][0, :2])
    names = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")
             for leg in ("FL", "FR", "RL", "RR")]
    mapping = [names.index(name) for name in contract["joint_names"]]
    default = torch.tensor([[contract["default_joint_angles"][name] for name in names]]).repeat(2, 1)
    position, velocity = default.clone(), torch.zeros_like(default)
    position[:, mapping] += observation[:, 9:21]
    velocity[:, mapping] = observation[:, 21:33]/.05
    robot = SimpleNamespace(device="cpu", joint_names=names, data=SimpleNamespace(
        default_joint_pos=default, joint_pos=position, joint_vel=velocity,
        root_ang_vel_b=observation[:, :3]/.25, projected_gravity_b=observation[:, 3:6]))
    robot.set_joint_effort_target = lambda effort: setattr(robot, "effort", effort.clone())
    low = A2LowLevel(robot)
    low.commands[:] = observation[:, 6:9]/torch.tensor([2., 2., .25])
    low.previous_action[:] = observation[:, 33:45]
    observation_error = float((low.observation()-observation).abs().max())
    assert observation_error < 1e-6
    history = torch.zeros_like(low.history)
    action_error = 0.0
    with torch.inference_mode():
        for tick in range(400):
            # REASAN 和 A2 都是每四个物理步更新一次策略。
            if tick % 4 == 0:
                expected, history = exported(low.observation(), history)
                expected = expected.clamp(-100, 100)
            effort = low.apply()
            action_error = max(action_error, float((low.previous_action-expected).abs().max()))
            assert torch.equal(history, low.history)
            desired = default[:, mapping]+.25*expected
            pd = (low.kp*(desired-position[:, mapping])-low.kd*velocity[:, mapping]).clamp(-low.limits, low.limits)
            assert torch.equal(effort[:, mapping], pd)
        assert low.inference_calls.tolist() == [100, 100] and low.phase.tolist() == [0, 0]
        assert action_error == 0
        fields = ("history", "previous_action", "target", "commands", "phase", "inference_calls", "physics_steps")
        untouched = {name: getattr(low, name)[1].clone() for name in fields}
        low.reset(torch.tensor([0]))
        assert all(torch.equal(getattr(low, name)[1], value) for name, value in untouched.items())
        assert not low.history[0].any() and not low.previous_action[0].any()

    result = {"passed": True, "scope": "a2_moe_policy_and_pd_interface", "source_files_verified": len(manifest["files"]),
              "checkpoint_iter": checkpoint.get("iter"), "checkpoint_infos": checkpoint.get("infos"),
              "strict_state_dict_keys": len(checkpoint["model_state_dict"]),
              "original_to_exported_max_action_error": maximum_error, "compared_batches": 32,
              "native_observation_max_error": observation_error, "adapter_action_max_error": action_error,
              "policy_to_lab_joint_indices": mapping, "physics_steps": 400, "inference_calls_per_env": [100, 100],
              "selective_reset_preserves_other_environment": True,
              "policy_sha256": hashlib.sha256((ROOT/"assets/a2/student_model_103500.pt").read_bytes()).hexdigest()}
    (args.out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
