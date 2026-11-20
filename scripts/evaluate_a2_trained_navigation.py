"""训练后导航部署检查；源自 evaluate_a2_valid_spawn.py，增加显式导航权重路径。"""

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


def make_trial_batches(manifest, split, repeats, num_envs, family=None):
    import numpy as np

    batches = []
    for repeat in range(repeats):
        for family_index, name in enumerate(("sparse", "dense", "maze")):
            group = []
            for case in manifest["cases"]:
                if case["split"] != split or case["family"] != name:
                    continue
                rng = np.random.default_rng(np.random.SeedSequence([20261117, family_index, case["variant"], repeat]))
                group.append({**case, "repeat": repeat, "trial_id": f"{case['case_id']}_r{repeat:02d}",
                              "initial_yaw": float(rng.uniform(-np.pi, np.pi)),
                              "joint_scale": rng.uniform(0.6, 1.4, 12).tolist()})
            batches.extend(group[i:i+num_envs] for i in range(0, len(group), num_envs))
    # 选择分组后保留原完整评估的批次编号，对应同一组随机种子。
    return [(index, batch) for index, batch in enumerate(batches)
            if family is None or batch[0]["family"] == family]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("/workspace/assets/evaluation/public_static_v1"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True, help="训练导出的 best_policy.pt 或 last_policy.pt")
    parser.add_argument("--family", choices=["sparse", "dense", "maze"], help="只跑一个场景分组")
    parser.add_argument("--reference", type=Path, help="原评估目录，用于运行前逐条核对初态和评估设置")
    args = parser.parse_args()
    if min(args.repeats, args.num_envs) < 1:
        parser.error("重复次数与环境数必须大于零")
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
              "repeats": args.repeats, "family": args.family or "all", "terrain_sha256": actual_hash,
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
        from a2_adapter.nav_env import configure_a2
        from a2_adapter.valid_spawn_env import GeometrySpawnA2EvaluationEnv as A2StaticEvaluationEnv

        torch.set_num_threads(4)
        batches = make_trial_batches(manifest, args.split, args.repeats, args.num_envs, args.family)
        if not batches or any(len(batch) != args.num_envs for _, batch in batches):
            raise ValueError("所选路线必须能分为完整的并行批次")
        trials = [trial for _, batch in batches for trial in batch]
        reference_initial = {}
        if args.reference is not None:
            reference_summary = json.loads((args.reference/"summary.json").read_text())
            old_trials = json.loads((args.reference/"trials.json").read_text())
            if not reference_summary["complete"] or trials != [
                    trial for trial in old_trials if args.family is None or trial["family"] == args.family]:
                raise RuntimeError("当前评估清单与已完成的原评估不一致")
            reference_initial = {row["trial_id"]: row for row in
                                 json.loads((args.reference/"initial_states.json").read_text())}
            report["paired_reference"] = str(args.reference)
            report["paired_initial_states_checked"] = 0
        report["expected_episodes"] = len(trials)
        report["original_batch_indices"] = [index for index, _ in batches]
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

        policy_path = args.policy.resolve()
        report["navigation_policy_path"] = str(policy_path)
        policy = torch.jit.load(str(policy_path), map_location="cuda:0").eval()
        for name in ("h_state", "c_state"):
            state = getattr(policy, name)
            setattr(policy, name, torch.zeros(state.shape[0], args.num_envs, state.shape[2], device="cuda:0"))
        report["models"] = {name: hashlib.sha256((Path("logs/rsl_rl/go2_lidar")/f"{name}_1/exported/policy.pt").read_bytes()).hexdigest()
                            for name in ("filter", "nav")}
        report["models"]["nav"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        report["models"]["a2_loco"] = hashlib.sha256(Path("/workspace/assets/a2/student_model_103500.pt").read_bytes()).hexdigest()
        report["robot"] = "A2"
        report["migration_scope"] = "explicit_navigation_frozen_official_filter_and_A2_MoE_valid_spawn_v1"
        report["initialization"] = env.spawn_configuration
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
                                   for name in ("evaluate_a2_trained_navigation.py", "evaluate_a2_valid_spawn.py", "static_eval_env.py", "evaluate_a2_static.py")}
        for relative in ("a2_adapter/nav_env.py", "a2_adapter/low_level.py", "a2_adapter/robot.py", "a2_adapter/valid_spawn_env.py", "a2_adapter/spawn_geometry.py"):
            report["script_sha256"][relative] = hashlib.sha256((Path("/workspace")/relative).read_bytes()).hexdigest()
        if args.reference is not None:
            fields = ("split", "repeats", "num_envs", "terrain_sha256", "source_commit", "input_source",
                      "success_distance_m", "failure_priority", "physics", "initialization", "runtime")
            for key in fields:
                if report[key] != reference_summary[key]:
                    raise RuntimeError("评估设置与原评估不同: "+key)
            for key in ("filter", "a2_loco"):
                if report["models"][key] != reference_summary["models"][key]:
                    raise RuntimeError("下层权重与原评估不同: "+key)
            for key in report["script_sha256"].keys() & reference_summary["script_sha256"].keys():
                if report["script_sha256"][key] != reference_summary["script_sha256"][key]:
                    raise RuntimeError("共用评估源码与原评估不同: "+key)
        report["startup_wall_seconds"] = time.monotonic()-started
        rollout_started = time.monotonic()
        report["phase"] = "evaluation"
        total_steps = 0
        with torch.inference_mode():
            for completed_batches, (batch_number, batch) in enumerate(batches, start=1):
                torch.manual_seed(10000+batch_number)
                observations, _ = env.set_batch(batch)
                if args.reference is not None:
                    for initial in env.recorded_initial_states[-len(batch):]:
                        if initial != reference_initial.get(initial["trial_id"]):
                            raise RuntimeError("实际物理初态与原评估不同: "+initial["trial_id"])
                    report["paired_initial_states_checked"] += len(batch)
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
                report["completed_batches"] = completed_batches
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
            if not episodes:
                continue
            counts = Counter(record["outcome"] for record in episodes)
            report["per_family"][family] = {"episodes": len(episodes), "outcomes": dict(counts),
                "success_rate": counts["success"]/len(episodes),
                "wilson_95_interval": wilson(counts["success"], len(episodes))}
        (args.out/"initial_states.json").write_text(json.dumps(env.recorded_initial_states, indent=2)+"\n")
        report["spawn_records"] = len(env.spawn_records)
        report["minimum_spawn_clearance_m"] = min(r["runtime_geometry_clearance_m"] for r in env.spawn_records)
        (args.out/"spawn_records.json").write_text(json.dumps(env.spawn_records, indent=2)+"\n")
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
