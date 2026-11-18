"""固定 REASAN 导航/安全模型，以冻结 A2 MoE 步态运行相同静态路线。"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def wilson(successes, count):
    if count == 0:
        return None
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0, center-half), min(1, center+half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("/workspace/assets/evaluation/public_static_v1"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--num-envs", type=int, choices=[1, 2], default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats 必须大于零")
    if args.out.exists() and any(path.name != "console.log" for path in args.out.iterdir()):
        parser.error("请使用新的输出目录")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.assets/"manifest.json").read_text())
    actual_hash = hashlib.sha256((args.assets/"terrain.npz").read_bytes()).hexdigest()
    if actual_hash != manifest["terrain_sha256"]:
        raise RuntimeError("冻结场景文件与清单不符")
    a2_source = json.loads(Path("/workspace/assets/a2/SOURCE.json").read_text())
    for relative, entry in a2_source["files"].items():
        if hashlib.sha256((Path("/workspace")/relative).read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("A2 资产或策略与来源清单不符: "+relative)
    started = time.monotonic()
    app = env = None
    report = {"complete": False, "split": args.split, "num_envs": args.num_envs,
              "repeats": args.repeats, "terrain_sha256": actual_hash,
              "source_commit": manifest["source_commit"], "input_source": "simulated_rays",
              "success_distance_m": 0.5, "failure_priority": "collision_or_tilt_before_success_before_timeout",
              "criterion_source": "locally_defined_not_claimed_as_paper_protocol",
              "outer_policy_reset_checks": 0, "completed_episodes": 0}

    def save():
        report["wall_seconds"] = time.monotonic()-started
        (args.out/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        save()
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device="cuda:0", kit_args=(
            "--/plugins/carb.tasking.plugin/threadCount=4 "
            "--/persistent/isaac/asset_root/cloud=/asset-cache")).app
        import numpy as np
        import torch
        import go2_lidar.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from go2_lidar.terrain.train_terrain_cfg import GO2_TEST_TERRAIN_CFG
        from scripts.static_eval_env import FrozenTestTerrain
        from a2_adapter.nav_env import A2StaticEvaluationEnv, configure_a2

        torch.set_num_threads(4)
        trials = []
        batches = []
        for repeat in range(args.repeats):
            for family_index, family in enumerate(("sparse", "dense", "maze")):
                group = []
                for case in manifest["cases"]:
                    if case["split"] != args.split or case["family"] != family:
                        continue
                    rng = np.random.default_rng(np.random.SeedSequence([20261117, family_index, case["variant"], repeat]))
                    trial = {**case, "repeat": repeat, "trial_id": f"{case['case_id']}_r{repeat:02d}",
                             "initial_yaw": float(rng.uniform(-np.pi, np.pi)),
                             "joint_scale": rng.uniform(0.6, 1.4, 12).tolist()}
                    group.append(trial)
                    trials.append(trial)
                batches.extend(group[i:i+args.num_envs] for i in range(0, len(group), args.num_envs))
        report["expected_episodes"] = len(trials)
        (args.out/"trials.json").write_text(json.dumps(trials, indent=2)+"\n")
        cfg = parse_env_cfg("Unitree-Go2-Navigation", device="cuda:0", num_envs=args.num_envs)
        cfg.seed = 0
        cfg.is_play_env = True
        cfg.wait_for_key = False
        cfg.use_dynamic_obstacle = False
        cfg.use_predicted_rays = False
        cfg.events.push_robot = None
        cfg.set_raycaster_measure_pattern("11x")
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
        cfg.terrain.terrain_generator = GO2_TEST_TERRAIN_CFG.copy()
        cfg.terrain.terrain_generator.class_type = FrozenTestTerrain
        cfg.terrain.terrain_generator.frozen_asset = str(args.assets/"terrain.npz")
        report["phase"] = "environment_create"
        save()
        configure_a2(cfg)
        env = A2StaticEvaluationEnv(cfg=cfg, out=args.out)
        from go2_lidar.tasks.go2_nav_env import Go2NavEnv
        for name in ("_get_observations", "_augment_ray_obs", "_get_rewards"):
            if getattr(type(env), name) is not getattr(Go2NavEnv, name):
                raise RuntimeError("原导航方法未按预期直接继承: "+name)
        report["inherited_official_methods_checked"] = ["_get_observations", "_augment_ray_obs", "_get_rewards"]
        if len(env._obstacles) != 0:
            raise RuntimeError("静态评估出现动态物体")

        policy_path = Path("logs/rsl_rl/go2_lidar/nav_1/exported/policy.pt")
        policy = torch.jit.load(str(policy_path), map_location="cuda:0").eval()
        for name in ("h_state", "c_state"):
            state = getattr(policy, name)
            setattr(policy, name, torch.zeros(state.shape[0], args.num_envs, state.shape[2], device="cuda:0"))
        report["models"] = {name: hashlib.sha256((Path("logs/rsl_rl/go2_lidar")/f"{name}_1/exported/policy.pt").read_bytes()).hexdigest()
                            for name in ("filter", "nav")}
        report["models"]["a2_loco"] = hashlib.sha256(Path("/workspace/assets/a2/student_model_103500.pt").read_bytes()).hexdigest()
        report["robot"] = "A2"
        report["migration_scope"] = "frozen_REASAN_nav_filter_with_existing_A2_MoE_student"
        report["physics"] = {"dt": cfg.sim.dt, "control_dt": env.step_dt, "contact_threshold_n": 1.0,
                             "upright_threshold": 0.75, "gpu_max_rigid_patch_count": 2**18}
        report["runtime"] = {"torch": torch.__version__, "device": torch.cuda.get_device_name(),
                             "masses": env._robot.root_physx_view.get_masses().cpu().tolist(),
                             "joint_names": env._robot.joint_names, "body_names": env._robot.body_names,
                             "policy_joint_indices": env._a2_loco.joints,
                             "effective_kp": env._a2_loco.kp.cpu().tolist(), "effective_kd": env._a2_loco.kd.cpu().tolist(),
                             "motor_strengths": env._a2_motor_strength.cpu().tolist(),
                             "physics_parameters": env._physics_parameters.cpu().tolist()}
        report["script_sha256"] = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                   for name in ("evaluate_a2_static.py", "static_eval_env.py")}
        for relative in ("a2_adapter/nav_env.py", "a2_adapter/low_level.py", "a2_adapter/robot.py"):
            report["script_sha256"][relative] = hashlib.sha256((Path("/workspace")/relative).read_bytes()).hexdigest()
        report["startup_wall_seconds"] = time.monotonic()-started
        rollout_started = time.monotonic()
        report["phase"] = "evaluation"
        total_steps = 0
        with torch.inference_mode():
            for batch_number, batch in enumerate(batches):
                torch.manual_seed(10000+batch_number)
                observations, _ = env.set_batch(batch)
                policy.h_state.zero_()
                policy.c_state.zero_()
                budget = math.ceil(max(case["deadline_s"] for case in batch)/env.step_dt)+1
                for _ in range(budget):
                    obs = observations["policy"]
                    if not torch.isfinite(obs).all():
                        raise RuntimeError("评估观测出现非有限数值")
                    p0, p1 = cfg.obs_ranges["proprio"]
                    r0, r1 = cfg.obs_ranges["actor_ray"]
                    env._a2_navigation_input = obs[:, p0:p1].clone()
                    env._a2_navigation_rays = obs[:, r0:r1].clone()
                    if not torch.equal(env._a2_navigation_rays, env._obs_actor_rays):
                        raise RuntimeError("导航与安全模型收到的射线不一致")
                    action = policy(obs[:, p0:p1], obs[:, r0:r1])
                    if not torch.isfinite(action).all():
                        raise RuntimeError("策略输出出现非有限数值")
                    action[~env.active_trials] = 0
                    observations, _, terminated, truncated, _ = env.step(action)
                    done = terminated | truncated
                    if done.any():
                        for name in ("h_state", "c_state"):
                            getattr(policy, name)[:, done] = 0
                        for model, names in ((policy, ("h_state", "c_state")),
                                             (env._filter_policy, ("h_state", "c_state")),
                                             ):
                            for name in names:
                                if torch.count_nonzero(getattr(model, name)[:, done]):
                                    raise RuntimeError("回合结束后循环状态未清零")
                        for name in ("history", "previous_action", "commands", "phase", "inference_calls", "physics_steps"):
                            if torch.count_nonzero(getattr(env._a2_loco, name)[done]):
                                raise RuntimeError("A2 回合结束后状态未清零: "+name)
                        if not torch.equal(env._a2_loco.target[done], env._a2_loco.default[done]):
                            raise RuntimeError("A2 复位后关节目标不是默认姿态")
                        report["outer_policy_reset_checks"] += int(done.sum())
                    total_steps += 1
                    if not env.active_trials.any():
                        break
                if env.active_trials.any():
                    raise RuntimeError("路线超过预算后未产生终止记录")
                report["completed_episodes"] = len(env.episodes)
                report["completed_batches"] = batch_number+1
                save()

        report["rollout_wall_seconds"] = time.monotonic()-rollout_started
        report["control_steps"] = total_steps
        report["a2_control_period_checks"] = env.control_period_checks
        report["selective_reset_checks"] = env.selective_reset_checks
        report["commands_outside_candidate_source_range"] = {"records": env.command_records,
            "outside": env.command_outside_candidate, "candidate_limits": [2., 1., 2.],
            "additional_command_clipping": False, "actual_checkpoint_training_ranges_unknown": True}
        report["per_family"] = {}
        for family in ("sparse", "dense", "maze"):
            episodes = [record for record in env.episodes if record["family"] == family]
            counts = Counter(record["outcome"] for record in episodes)
            report["per_family"][family] = {"episodes": len(episodes), "outcomes": dict(counts),
                "success_rate": counts["success"]/len(episodes),
                "wilson_95_interval": wilson(counts["success"], len(episodes))}
        (args.out/"initial_states.json").write_text(json.dumps(env.recorded_initial_states, indent=2)+"\n")
        report["complete"] = len(env.episodes) == len(trials)
        report["phase"] = "finished"
        save()
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        report["phase"] = "failed"
        save()
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            from isaaclab.sim import SimulationContext
            sim = SimulationContext.instance()
            if sim is not None:
                sim.clear_all_callbacks()
                sim.clear_instance()
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
