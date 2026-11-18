"""仅适配固定评估场景与回合记录；控制和传感器沿用官方 Go2NavEnv。"""

import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from go2_lidar.tasks.go2_nav_env import Go2NavEnv
from isaaclab.utils import math as math_utils


class ZeroTrainingGuide:
    """引导仅供奖励及 critic 使用，官方 actor 不读取它。"""

    def get_optimal_direction(self, x, **kwargs):
        return torch.zeros((len(x), 2), device=x.device)


class FrozenTestTerrain:
    def __init__(self, cfg, device):
        with np.load(cfg.frozen_asset) as data:
            self.terrain_mesh = trimesh.Trimesh(data["vertices"], data["faces"], process=False)
            self.terrain_origins = data["terrain_origins"].copy()
            self.flat_patches = {name[6:]: torch.as_tensor(data[name].copy(), device=device)
                                 for name in data.files if name.startswith("patch_")}
        # 兼容原环境构造时读取的训练元数据；实际起终点由固定测试路线提供。
        for prefix in ("general", "dead_end", "detour"):
            self.flat_patches[prefix + "_start"] = self.flat_patches["scatter_static_sparse_start"]
            self.flat_patches[prefix + "_goal"] = self.flat_patches["scatter_static_sparse_goal"]
            self.flat_patches[prefix + "_guide_generator"] = ZeroTrainingGuide()


class StaticEvaluationEnv(Go2NavEnv):
    def __init__(self, cfg, out, **kwargs):
        self.output_dir = Path(out)
        self.episodes = []
        self.batch_cases = []
        self.recorded_initial_states = []
        super().__init__(cfg, **kwargs)
        self.active_trials = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.minimum_distances = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._trajectory_file = (self.output_dir / "trajectory.jsonl").open("w")
        self._episodes_file = (self.output_dir / "episodes.jsonl").open("w")

    def set_batch(self, cases):
        if self.active_trials.any():
            raise RuntimeError("上一组路线尚未结束")
        if len(cases) != self.num_envs:
            raise ValueError("每组路线数必须等于并行环境数")
        if len({case["case_id"] for case in cases}) != len(cases):
            raise ValueError("同一组不能在同一物理区域放置两只机器人")
        self.batch_cases = cases
        self.active_trials[:] = True
        self.minimum_distances[:] = float("inf")
        observations, info = self.reset()
        for index, case in enumerate(cases):
            self.recorded_initial_states.append({"trial_id": case["trial_id"],
                "position": self._robot.data.root_pos_w[index].cpu().tolist(),
                "quaternion": self._robot.data.root_quat_w[index].cpu().tolist(),
                "joint_position": self._robot.data.joint_pos[index].cpu().tolist(),
                "goal": self._cmd_goal_w[index].cpu().tolist()})
        return observations, info

    def _update_curriculum(self, env_ids):
        # 固定测试路线，不按上个回合表现改变地形或目标。
        self._first_reset = False
        self._num_active_obstacles[env_ids] = 0
        for index in env_ids.tolist():
            case = self.batch_cases[index]
            self._terrain.terrain_levels[index] = case["variant"]
            self._env_terrain_cols[index] = case["column"]
            self._terrain.env_origins[index] = self._terrain.terrain_origins[case["variant"], case["column"]]
            self._episode_length_sec_per_env[index] = case["deadline_s"]

    def _reset_robot_and_cmd(self, env_ids):
        root = self._robot.data.default_root_state[env_ids].clone()
        for row, index in enumerate(env_ids.tolist()):
            case = self.batch_cases[index]
            root[row, :2] = torch.tensor(case["start"][:2], device=self.device)
            yaw = case["initial_yaw"]
            root[row, 3:7] = torch.tensor([np.cos(yaw/2), 0, 0, np.sin(yaw/2)], device=self.device)
            root[row, 7:] = 0
            self._cmd_goal_w[index, :2] = torch.tensor(case["goal"][:2], device=self.device)
            self._cmd_goal_w[index, 2] = 0
            self._robot_start_pos[index] = root[row, :3]
        self._robot.write_root_link_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(root[:, 7:], env_ids)
        self._cmd_goal_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self._robot.data.root_quat_w), self._cmd_goal_w-self._robot.data.root_pos_w)
        norm = self._cmd_goal_b.norm(dim=-1, keepdim=True)
        self._cmd_goal_dir_b = self._cmd_goal_b / norm.clamp_min(1e-6)
        self._cmd_goal_heading_b = torch.atan2(self._cmd_goal_dir_b[:, 1], self._cmd_goal_dir_b[:, 0])[:, None]

    def _reset_obstacles(self, env_ids):
        self._no_obstacle_env[env_ids] = True

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # 原范围仍为默认关节角的 0.6～1.4 倍；将每条路线的采样固定下来。
        scales = torch.tensor([self.batch_cases[index]["joint_scale"] for index in env_ids.tolist()], device=self.device)
        positions = self._robot.data.default_joint_pos[env_ids] * scales
        limits = self._robot.data.soft_joint_pos_limits[env_ids]
        positions = positions.clamp(limits[..., 0], limits[..., 1])
        self._robot.write_joint_state_to_sim(positions, self._robot.data.default_joint_vel[env_ids], env_ids=env_ids)

    def _get_dones(self):
        failed, expired = super()._get_dones()
        position = self._robot.data.root_pos_w
        distance = (self._cmd_goal_w[:, :2]-position[:, :2]).norm(dim=-1)
        if not torch.isfinite(position).all() or not torch.isfinite(distance).all():
            raise RuntimeError("评估物理状态出现非有限数值")
        upright = -self._robot.data.projected_gravity_b[:, 2]
        tilted = upright < 0.75
        contact = self._contact_sensor.data.net_forces_w[:, self._undesired_contact_body_ids_cs].norm(dim=-1).amax(dim=-1)
        base_history = self._contact_sensor.data.net_forces_w_history[:, :, self._base_id_cs].norm(dim=-1).flatten(1).amax(dim=1)
        collision = (contact > 1.0) | (base_history > 1.0)
        succeeded = (distance < 0.5) & ~failed
        ended = (failed | expired | succeeded) & self.active_trials
        self.minimum_distances[self.active_trials] = torch.minimum(
            self.minimum_distances[self.active_trials], distance[self.active_trials])

        for index in self.active_trials.nonzero().flatten().tolist():
            case = self.batch_cases[index]
            step = int(self.episode_length_buf[index])
            record = {"trial_id": case["trial_id"], "step": step, "time_s": step*self.step_dt,
                      "position": position[index].cpu().tolist(),
                      "quaternion": self._robot.data.root_quat_w[index].cpu().tolist(),
                      "goal_distance": float(distance[index]), "upright": float(upright[index]),
                      "contact_force_n": float(contact[index]), "base_history_force_n": float(base_history[index]),
                      "navigation_command": self._high_actions[index].cpu().tolist(),
                      "safe_command": self._filter_actions[index].cpu().tolist()}
            if step % 5 == 0 or bool(ended[index]):
                self._trajectory_file.write(json.dumps(record, allow_nan=False)+"\n")
            if bool(ended[index]):
                # 与用户确认的判据一致：碰撞/倾斜优先，到达优先于同一步超时。
                if bool(failed[index]):
                    outcome = "collision" if bool(collision[index]) else "tilt" if bool(tilted[index]) else "termination"
                else:
                    outcome = "success" if bool(succeeded[index]) else "timeout"
                episode = {**case, **record, "outcome": outcome,
                           "minimum_goal_distance": float(self.minimum_distances[index]),
                           "collision": bool(collision[index]), "tilted": bool(tilted[index]),
                           "recorded_before_reset": True}
                self.episodes.append(episode)
                self._episodes_file.write(json.dumps(episode, allow_nan=False)+"\n")
                self._episodes_file.flush()
                self._trajectory_file.flush()
                print(json.dumps({"trial_id": case["trial_id"], "outcome": outcome,
                                  "time_s": record["time_s"], "distance": record["goal_distance"]}), flush=True)

        self.active_trials[ended] = False
        terminated = (failed | succeeded) & ended
        timeout = expired & ended & ~terminated
        self._reset_buf[:] = terminated | timeout
        return terminated, timeout

    def close(self):
        for name in ("_trajectory_file", "_episodes_file"):
            stream = getattr(self, name, None)
            if stream is not None:
                stream.close()
        super().close()
