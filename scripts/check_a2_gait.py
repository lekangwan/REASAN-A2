"""原 A2 资产和冻结学生策略的平地检查，不接入导航与安全层。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.)
    args = parser.parse_args()
    if args.seconds <= 2 or abs(args.seconds/.02-round(args.seconds/.02)) > 1e-6:
        parser.error("seconds 必须大于两秒且为 0.02 秒的整数倍")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"passed": False, "completed": False, "scope": "a2_flat_ground_only",
              "navigation_connected": False, "ppo_updates": 0, "seconds_per_command": args.seconds}
    app = sim = None

    def save():
        result["wall_seconds"] = time.monotonic()-started
        (args.out/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    try:
        save()
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device="cuda:0", kit_args=(
            "--/plugins/carb.tasking.plugin/threadCount=4 "
            "--/persistent/isaac/asset_root/cloud=/asset-cache")).app
        import numpy as np
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils import configclass
        from a2_adapter.robot import asset_path, robot_cfg, robot_contract
        from a2_adapter.low_level import A2LowLevel

        torch.set_num_threads(4)
        torch.manual_seed(0)
        contract = robot_contract()
        reference = ROOT/"assets/a2/native_reference"
        native = json.loads((reference/"result.json").read_text())
        assert native["passed"]
        cases = native["cases"]

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.)))
            a2 = robot_cfg(asset_path())
            contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/A2/.*", history_length=1)

        result["phase"] = "scene_create"
        save()
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device="cuda:0",
            physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=2**18)))
        scene = InteractiveScene(SceneCfg(num_envs=len(cases), env_spacing=20.))
        sim.reset()
        robot = scene["a2"]
        mapping = [robot.joint_names.index(name) for name in contract["joint_names"]]
        assert len(robot.joint_names) == 12 and len(robot.body_names) == 17
        assert set(robot.joint_names) == set(contract["joint_names"])
        mass = robot.data.default_mass.sum(-1)
        assert float((mass-contract["total_urdf_mass_kg"]).abs().max()) < .01
        (args.out/"asset_review.json").write_text(json.dumps({"joint_names": robot.joint_names,
            "body_names": robot.body_names, "policy_to_lab_joint_indices": mapping,
            "mass_kg": mass.cpu().tolist(), "usd": str(asset_path())}, indent=2)+"\n")

        initial = json.loads((reference/"initial_state.json").read_text())
        with np.load(reference/"trace.npz") as data:
            initial_observation = torch.tensor(data["observation"][0], device=robot.device)
        state = torch.tensor(initial["root_state"], device=robot.device)
        state[:, 3:7] = state[:, [6, 3, 4, 5]].clone()  # 原生 xyzw 转 Lab wxyz。
        state[:, :3] += scene.env_origins
        q, dq = robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.joint_vel)
        q[:, mapping] = initial_observation[:, 9:21]+robot.data.default_joint_pos[:, mapping]
        dq[:, mapping] = initial_observation[:, 21:33]/.05
        robot.write_root_state_to_sim(state)
        robot.write_joint_state_to_sim(q, dq)
        scene.reset()
        sim.forward()
        scene.update(0.)

        low = A2LowLevel(robot)
        low.commands[:] = torch.tensor([case["command"] for case in cases], device=robot.device)
        low.previous_action[:] = initial_observation[:, 33:45]
        error = float((low.observation()-initial_observation).abs().max())
        assert error < 1e-4, "当前初始观测与原生参考不一致"
        base = scene["contacts"].body_names.index("base_link")
        failed = torch.zeros(len(cases), dtype=torch.bool, device=robot.device)
        contact_max = torch.zeros(len(cases), device=robot.device)
        min_height = torch.full((len(cases),), float("inf"), device=robot.device)
        min_upright = torch.ones(len(cases), device=robot.device)
        trace = []
        steps = round(args.seconds/.02)
        result.update(phase="policy_run", initial_observation_max_error=error)
        save()
        rollout_started = time.monotonic()
        with torch.inference_mode():
            for step in range(steps):
                for _ in range(4):
                    effort = low.apply()
                    assert torch.isfinite(effort).all() and torch.isfinite(low.previous_action).all()
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(.005)
                    root = robot.data.root_state_w
                    upright = -robot.data.projected_gravity_b[:, 2]
                    contact = scene["contacts"].data.net_forces_w[:, base].norm(dim=-1)
                    assert torch.isfinite(root).all()
                    failed |= (root[:, 2] < .2) | (upright < .5) | (contact > 1.)
                    contact_max = torch.maximum(contact_max, contact)
                    min_height = torch.minimum(min_height, root[:, 2])
                    min_upright = torch.minimum(min_upright, upright)
                trace.append({"root_state": root.cpu().numpy().copy(),
                    "body_velocity": torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), -1).cpu().numpy().copy(),
                    "action": low.previous_action.cpu().numpy().copy()})
        arrays = {key: np.stack([row[key] for row in trace]) for key in trace[0]}
        np.savez_compressed(args.out/"trace.npz", **arrays)
        warmup = round(2/.02)
        result.update(completed=True, passed=not bool(failed.any()), phase="finished",
            physics_dt=.005, inference_dt=.02, physics_steps=steps*4,
            inference_calls_per_env=low.inference_calls.cpu().tolist(),
            rollout_wall_seconds=time.monotonic()-rollout_started,
            flat_check_criteria={"minimum_height_m": .2, "minimum_upright": .5, "maximum_base_contact_n": 1.},
            cases=[{"name": case["name"], "command": case["command"], "stable": not bool(failed[i]),
                    "steady_body_velocity_mean": arrays["body_velocity"][warmup:, i].mean(0).tolist(),
                    "base_contact_max_n": float(contact_max[i]), "minimum_height_m": float(min_height[i]),
                    "minimum_upright": float(min_upright[i])} for i, case in enumerate(cases)],
            policy_sha256=hashlib.sha256((ROOT/"assets/a2/student_model_103500.pt").read_bytes()).hexdigest())
        assert bool((low.inference_calls == steps).all())
        assert bool((low.physics_steps == steps*4).all())
        save()
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
        save()
    finally:
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        if app is not None:
            app.close(wait_for_replicator=False)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
