"""导出原A2学生网络；历史显式输入输出，支持运行器按环境独立复位。"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"upstream/a2_rl_gym"


class StudentStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.student_moe_encoder = copy.deepcopy(model.student_moe_encoder).cpu().eval()
        self.actor = copy.deepcopy(model.actor).cpu().eval()

    def forward(self, observation: torch.Tensor, history: torch.Tensor):
        history = torch.cat((history[:, 1:], observation.unsqueeze(1)), dim=1)
        latent, _ = self.student_moe_encoder(history.flatten(1))
        return self.actor(torch.cat((latent, observation), dim=1)), history


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE/relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    torch.set_num_threads(4)
    for name in ("rsl_rl", "rsl_rl.modules"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    load("rsl_rl.modules.utils", "rsl_rl/rsl_rl/modules/utils.py")
    module = load("rsl_rl.modules.actor_critic_moe_cts", "rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py")
    model = module.ActorCriticMoECTS(45, 263, 12, 8, 5, expert_num=8).cpu().eval()
    checkpoint = ROOT/"model_103500.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
    out = ROOT/"assets/a2"
    out.mkdir(parents=True, exist_ok=True)
    path = out/"student_model_103500.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.jit.script(StudentStep(model)).save(str(path))
    policy = torch.jit.load(str(path)).eval()
    torch.manual_seed(20261116)
    history, max_error = torch.zeros(8, 5, 45), 0.
    with torch.inference_mode():
        for i in range(32):
            if i in (5, 16, 24):
                mask = torch.tensor([(k+i) % 3 == 0 for k in range(8)])
                model.reset(mask)
                history[mask] = 0
            obs = torch.randn(8, 45)
            expected = model.act_inference(obs)
            actual, history = policy(obs, history)
            max_error = max(max_error, float((expected-actual).abs().max()))
            assert torch.equal(history, model.history)
    assert max_error <= 1e-6
    manifest = {"source_commit": "d3d8929ae615504eb9edf454714505c3c01dbcba",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "student_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "observation_dim": 45, "history_shape_per_env": [5, 45], "action_dim": 12,
        "inference_period_seconds": .02, "original_inference_max_absolute_error": max_error,
        "compared_batches": 32, "batch_size": 8, "selective_reset_batches": [5, 16, 24],
        "learned_modules": "original StudentMoEEncoder and actor unchanged",
        "wrapper_difference": "explicit history input/output instead of internal buffer"}
    (out/"STUDENT_POLICY.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
