"""短时间验证官方导出策略及原始环境，不执行训练或正式成功率评估。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


class HeadlessViewport:
    """原步态演示逐步更新镜头；无窗口检查仅忽略这项显示操作。"""

    def update_view_location(self, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["loco", "filter", "nav"], default="loco")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.num_envs < 1:
        parser.error("seconds 和 num-envs 必须为正数")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        parser.error("请为每次运行使用新的输出目录")

    started = time.monotonic()
    result = {"task": args.task, "scope": "pretrained_interface_and_motion_check",
              "input_source": "proprioception" if args.task == "loco" else "simulated_rays",
              "num_envs": args.num_envs, "dynamic_obstacles": args.dynamic,
              "completed_steps": 0, "finite": True, "terminated": 0, "truncated": 0,
              "completed": False}
    app = env = None

    def save():
        result["wall_seconds"] = time.monotonic() - started
        (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def phase(name):
        result["phase"] = name
        save()
        print(json.dumps({"phase": name, "steps": result["completed_steps"]}), flush=True)

    try:
        phase("application_start")
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device="cuda:0", kit_args=(
            "--/plugins/carb.tasking.plugin/threadCount=4 "
            "--/persistent/isaac/asset_root/cloud=/asset-cache")).app
        phase("application_ready")

        import gymnasium as gym
        import torch
        import go2_lidar.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        torch.set_num_threads(4)
        task_name = {"loco": "Unitree-Go2-Locomotion", "filter": "Unitree-Go2-Filter",
                     "nav": "Unitree-Go2-Navigation"}[args.task]
        cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=args.num_envs)
        cfg.seed = 0
        cfg.is_play_env = True
        # 单环境不需要原 4096 环境规模的接触缓冲区；不改变动力学或控制参数。
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
        if args.task != "loco":
            cfg.events.push_robot = None  # 与官方 play_filter / play_nav 一致。
            cfg.wait_for_key = False
            cfg.use_dynamic_obstacle = args.dynamic
            cfg.use_predicted_rays = False
            cfg.set_raycaster_measure_pattern("11x")

        result["physics_dt"] = cfg.sim.dt
        result["control_dt"] = cfg.sim.dt * cfg.decimation
        result["expected_steps"] = math.ceil(args.seconds / result["control_dt"])
        result["runtime_overrides"] = {"seed": cfg.seed, "headless": True,
                                       "gpu_max_rigid_patch_count": cfg.sim.physx.gpu_max_rigid_patch_count}
        phase("environment_create")
        env = gym.make(task_name, cfg=cfg)
        raw = env.unwrapped
        if args.task == "loco" and raw.viewport_camera_controller is None:
            raw.viewport_camera_controller = HeadlessViewport()
            result["runtime_overrides"]["viewport"] = "ignore_camera_updates_without_window"

        policy_path = Path("logs/rsl_rl/go2_lidar") / f"{args.task}_1/exported/policy.pt"
        result["policy_path"] = str(policy_path)
        result["policy_sha256"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        policy = torch.jit.load(str(policy_path), map_location="cuda:0").eval()
        # 导出模型默认只分配一份 LSTM 状态，为实际并行数量重新分配。
        state_names = ("hidden_state", "cell_state") if args.task == "loco" else ("h_state", "c_state")
        for name in state_names:
            state = getattr(policy, name)
            setattr(policy, name, torch.zeros(state.shape[0], args.num_envs, state.shape[2], device="cuda:0"))

        observations, _ = env.reset()
        result["observation_shape"] = list(observations["policy"].shape)
        if args.task != "loco":
            result["initial_active_dynamic_obstacles"] = raw._num_active_obstacles.cpu().tolist()
            result["maximum_active_dynamic_obstacles"] = raw._num_active_obstacles.cpu().tolist()
        if args.task == "nav":
            result["initial_goal_distance"] = (raw._cmd_goal_w[:, :2] - raw._robot.data.root_pos_w[:, :2]).norm(dim=-1).cpu().tolist()
        result["episode_boundaries"] = []
        phase("policy_ready")
        records = []
        minimum_height = float("inf")
        minimum_upright = float("inf")
        travelled = torch.zeros(args.num_envs, device="cuda:0")
        previous_position = raw._robot.data.root_pos_w[:, :2].clone()
        rollout_started = time.monotonic()
        with torch.inference_mode():
            for step in range(result["expected_steps"]):
                obs = observations["policy"]
                if args.task == "loco":
                    actions = policy(obs)
                else:
                    p0, p1 = cfg.obs_ranges["proprio"]
                    r0, r1 = cfg.obs_ranges["actor_ray"]
                    actions = policy(obs[:, p0:p1], obs[:, r0:r1])
                if not bool(torch.isfinite(actions).all()) or not bool(torch.isfinite(obs).all()):
                    result["finite"] = False
                    raise RuntimeError("策略观测或动作出现非有限数值")
                observations, reward, terminated, truncated, _ = env.step(actions)
                result["completed_steps"] = step + 1
                result["terminated"] += int(terminated.sum())
                result["truncated"] += int(truncated.sum())
                done = terminated | truncated
                for name in state_names:
                    getattr(policy, name)[:, done, :] = 0
                if bool(done.any()):
                    result["episode_boundaries"].append({"step": step + 1,
                        "terminated": terminated.cpu().tolist(), "truncated": truncated.cpu().tolist()})
                position = raw._robot.data.root_pos_w
                upright = -raw._robot.data.projected_gravity_b[:, 2]
                minimum_height = min(minimum_height, float(position[:, 2].min()))
                minimum_upright = min(minimum_upright, float(upright.min()))
                delta = (position[:, :2] - previous_position).norm(dim=-1)
                travelled += delta * ~done  # 自动复位的瞬移不计入运动距离。
                previous_position = position[:, :2].clone()
                if not bool(torch.isfinite(position).all()) or not bool(torch.isfinite(reward).all()):
                    result["finite"] = False
                    raise RuntimeError("物理状态或奖励出现非有限数值")
                if step % 5 == 0 or bool(done.any()):
                    record = {"step": step + 1, "time_s": (step + 1) * result["control_dt"],
                              "position": position.cpu().tolist(),
                              "velocity_body": raw._robot.data.root_com_lin_vel_b.cpu().tolist(),
                              "upright": upright.cpu().tolist(), "action": actions.cpu().tolist(),
                              "terminated": terminated.cpu().tolist(), "truncated": truncated.cpu().tolist()}
                    if args.task != "loco":
                        active = raw._num_active_obstacles.cpu().tolist()
                        record["active_dynamic_obstacles"] = active
                        result["maximum_active_dynamic_obstacles"] = [max(a, b) for a, b in zip(
                            result["maximum_active_dynamic_obstacles"], active)]
                        record["above_ground_dynamic_obstacles"] = [sum(
                            float(obj.data.root_pos_w[index, 2]) > -10.0 for obj in raw._obstacles)
                            for index in range(args.num_envs)]
                    if args.task == "nav":
                        record["goal_distance"] = (raw._cmd_goal_w[:, :2] - position[:, :2]).norm(dim=-1).cpu().tolist()
                    records.append(record)

        result.update({"completed": True, "phase": "finished", "simulation_seconds": raw.step_dt * result["completed_steps"],
                       "rollout_wall_seconds": time.monotonic() - rollout_started,
                       "minimum_world_height": minimum_height, "minimum_upright": minimum_upright,
                       "travelled_metres": travelled.cpu().tolist(),
                       "gpu_peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
        (args.out / "trajectory.json").write_text(json.dumps(records, allow_nan=False) + "\n")
        phase("finished")
    except Exception:
        result["error"] = traceback.format_exc()
        phase("failed")
        traceback.print_exc()
    finally:
        save()
        if env is not None:
            env.close()
        if app is not None:
            app.close(wait_for_replicator=False)
    if not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
