# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""独立 A2 导航后端。五个机器人相关方法源自固定版本的 Go2NavEnv。

其余导航、射线、奖励与终止逻辑直接继承。修改对照见 REASAN_METHOD_DIFF.patch。
"""

from __future__ import annotations

import asyncio
import copy
import math
import os

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import (
    Articulation,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
)
from isaaclab.sensors import (
    ContactSensor,
)
from isaaclab.terrains import TerrainImporter

from go2_lidar.sensor.raycaster_dynamic import RayCasterDynamic
from go2_lidar.tasks.go2_nav_env_cfg import Go2NavEnvCfg
from go2_lidar.utils.hdf5_data_collector import HDF5DatasetWriter_General, HDF5DatasetWriter_Ray


from go2_lidar.tasks.go2_nav_env import Go2NavEnv
from a2_adapter.low_level import A2LowLevel
from a2_adapter.robot import asset_path, robot_cfg, robot_contract
from scripts.static_eval_env import StaticEvaluationEnv
from pathlib import Path
import json


class A2NavEnv(Go2NavEnv):
    def __init__(self, cfg: Go2NavEnvCfg, render_mode: str | None = None, **kwargs):
        print("Initializing training environment...")
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)

        if self.cfg.use_dynamic_obstacle:
            print("Using dynamic obstacles in the environment.")
        else:
            print("Not using dynamic obstacles in the environment.")
        self._wait_for_key()

        print(f"number of dynamic obstacles: {self._num_obstacles}")
        self._wait_for_key()

        # 冻结 A2 学生策略，替换 Go2 的步态模型与状态。
        self._a2_loco = A2LowLevel(self._robot)
        self._a2_motor_strength = (torch.rand_like(self._robot.data.joint_pos) * 0.2 + 0.9)[:, self._a2_loco.joints]

        filter_policy_path = "./logs/rsl_rl/go2_lidar/filter_1/exported/policy.pt"
        print(f"Loading filter policy from: {filter_policy_path}")
        self._filter_policy = torch.jit.load(filter_policy_path)
        self._filter_policy.to(self.device).eval()
        hidden_state_shape = self._filter_policy.h_state.shape
        cell_state_shape = self._filter_policy.c_state.shape
        self._filter_policy.h_state = torch.zeros(
            hidden_state_shape[0],
            self.num_envs,
            hidden_state_shape[2],
            device=self.device,
        )
        self._filter_policy.c_state = torch.zeros(
            cell_state_shape[0],
            self.num_envs,
            cell_state_shape[2],
            device=self.device,
        )
        for param in self._filter_policy.parameters():
            param.requires_grad = False
        self._wait_for_key()

        self._ray_predictor = None
        ray_predictor_path = "./go2_lidar/ray_predictor/ray_predictor.pt"
        print(f"Loading ray estimator model from: {ray_predictor_path}")
        try:
            self._ray_predictor = torch.jit.load(ray_predictor_path)
            self._ray_predictor.to(self.device).eval()
            print("Ray estimator model loaded:")
            print(self._ray_predictor)
        except Exception:
            print("Failed to load ray estimator model.")
        self._wait_for_key()

        print(f"num ray centers: {self.cfg.num_ray_centers}")
        self._wait_for_key()

        self._randomize_mass()

        self._reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._timeout_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._init_physx_material_buffer()
        self._reset_physx_materials(torch.ones(self.num_envs, device="cpu", dtype=torch.bool))

        # 保留原增益/电机强度的 ±10% 采样分布，名义增益改用 A2 自身值。
        # 强度乘入 Kp/Kd，等价于对 PD 力矩乘强度后再限幅。
        self._a2_kp_factor = (torch.rand_like(self._robot.data.joint_pos) * 0.2 + 0.9)[:, self._a2_loco.joints]
        self._a2_kd_factor = (torch.rand_like(self._robot.data.joint_pos) * 0.2 + 0.9)[:, self._a2_loco.joints]
        self._a2_loco.kp = self._a2_loco.kp[None, :] * self._a2_kp_factor * self._a2_motor_strength
        self._a2_loco.kd = self._a2_loco.kd[None, :] * self._a2_kd_factor * self._a2_motor_strength

        self._first_reset = True

        self._current_grid = None

        self._last_dist_to_goal = torch.zeros(self.num_envs, device=self.device)

        self._high_actions = torch.zeros(self.num_envs, self.cfg.num_high_actions, device=self.device)
        self._prev_high_actions = [self._high_actions.clone()] * 5
        self._num_high_actions = self._high_actions.shape[1]

        self._filter_actions = torch.zeros(self.num_envs, 3, device=self.device)

        self._loco_actions = torch.zeros(self.num_envs, self.cfg.num_loco_actions, device=self.device)
        self._prev_loco_actions = [self._loco_actions.clone()] * 5
        self._num_loco_actions = self._loco_actions.shape[1]

        self._obs_buf = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        self._obs_actor_rays = torch.zeros(self.num_envs, 180, device=self.device)

        self._init_data_extra_buf()

        self._base_id_cs, _ = self._contact_sensor.find_bodies("base_link")
        self._feet_ids_cs, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids_cs, _ = self._contact_sensor.find_bodies(
            [
                ".*thigh",
                ".*calf",
                "base_link",
                ".*hip",
            ]
        )
        self._feet_ids_bd, _ = self._robot.find_bodies(".*foot")
        self._hip_ids_jt, _ = self._robot.find_joints(".*hip.*")

        self._episode_sums = {}
        self._step_counter = 0
        self._iter_counter = 0

        self._track_env_id = 0
        self._track_env_curriculum = self.num_terrain_rows - 1
        self._track_camera_height = 5.0
        self._reset_tracked_env = False

        self._cmd_goal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_goal_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_goal_dir_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_goal_heading_b = torch.zeros(self.num_envs, 1, device=self.device)

        self._goal_dist_range = torch.zeros(self.num_envs, 2, device=self.device)
        self._goal_dist_range[:, 0] = 1.0
        self._goal_dist_range[:, 1] = 5.0

        self._obst_dist_range = torch.zeros(self.num_envs, 2, device=self.device)
        self._obst_dist_range[:, 0] = 2.0
        self._obst_dist_range[:, 1] = 5.0

        self._vel_cmd_limits = torch.tensor([[2.5, 1.5, 3.0]], device=self.device)

        self._no_obstacle_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._obst_pos_xy_a = torch.zeros(self.num_envs, self._num_obstacles, 2, device=self.device)
        self._obst_pos_xy_b = torch.zeros(self.num_envs, self._num_obstacles, 2, device=self.device)
        self._obst_speed = torch.zeros(self.num_envs, self._num_obstacles, 1, device=self.device)
        self._obst_speed_range = self.cfg.obst_speed_range
        print(f"obstacle speed range: {self._obst_speed_range}")
        self._wait_for_key()

        self._episode_length_sec = 15.0
        self._episode_length_sec_per_env = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_length_sec_per_env[:] = self._episode_length_sec

        self._ray_directions_b = torch.zeros(self.num_envs, 180, 3, device=self.device)
        self._ttc_full = torch.ones(self.num_envs, 180, device=self.device) * 3.0

        self._guiding_vectors = torch.zeros(self.num_envs, 2, device=self.device)
        self._goal_idx_general = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_idx_dead_end = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_idx_detour = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._proximal_ray_dist = 3.0
        self._num_prev_data = 15
        self._prev_grid_buf = torch.ones(self.num_envs, self._num_prev_data, 30, 180, device=self.device) * 3.0
        self._prev_proj_gravity_buf = torch.zeros(self.num_envs, self._num_prev_data, 3, device=self.device)
        self._prev_ang_vel_buf = torch.zeros(self.num_envs, self._num_prev_data, 3, device=self.device)

        self._init_data_collection()
        self._wait_for_key()

        self.border_x = self.cfg.terrain.terrain_generator.num_rows * self.cfg.terrain.terrain_generator.size[0] * 0.5
        self.border_y = self.cfg.terrain.terrain_generator.num_cols * self.cfg.terrain.terrain_generator.size[1] * 0.5

        self._env_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_type[: self.num_envs // 2] = 1
        self._env_type[self.num_envs // 2 :] = 3

        self._flat_patches_1 = self._terrain.flat_patches["general_start"].clone()
        self._flat_patches_1_goals = self._terrain.flat_patches["general_goal"].clone()
        self._flat_patches_2 = self._terrain.flat_patches["dead_end_start"].clone()
        self._flat_patches_2_goals = self._terrain.flat_patches["dead_end_goal"].clone()
        self._flat_patches_3 = self._terrain.flat_patches["detour_start"].clone()
        self._flat_patches_3_goals = self._terrain.flat_patches["detour_goal"].clone()
        print("Flat patches for curriculum prepared.")
        print(f"patches 1 shape: {self._flat_patches_1.shape}")
        print(f"patches 1 goals shape: {self._flat_patches_1_goals.shape}")
        print(f"patches 2 shape: {self._flat_patches_2.shape}")
        print(f"patches 2 goals shape: {self._flat_patches_2_goals.shape}")
        print(f"patches 3 shape: {self._flat_patches_3.shape}")
        print(f"patches 3 goals shape: {self._flat_patches_3_goals.shape}")
        self._wait_for_key()

        self._general_env_guide = self._terrain.flat_patches["general_guide_generator"]
        print(f"general guide class: {self._general_env_guide}")
        self._wait_for_key()

        self._dead_end_env_guide = self._terrain.flat_patches["dead_end_guide_generator"]
        print(f"dead end guide class: {self._dead_end_env_guide}")
        self._wait_for_key()

        self._detour_env_guide = self._terrain.flat_patches["detour_guide_generator"]
        print(f"detour guide class: {self._detour_env_guide}")
        self._wait_for_key()

        self._robot_start_pos = torch.zeros(self.num_envs, 3, device=self.device)

        if self.cfg.use_predicted_rays:
            print("=" * 50)
            print("Using predicted rays for policy input.")
            print("=" * 50)
            self._wait_for_key()

        self._init_debug_viz()
        self._update_debug_draw()

        asyncio.ensure_future(self.setup_ui())


    def _randomize_mass(self):
        asset = self._robot

        env_ids = torch.arange(self.num_envs, device="cpu")

        body_ids, _ = self._robot.find_bodies("base_link")
        body_ids = torch.tensor(body_ids, dtype=torch.int, device="cpu")
        assert body_ids.shape == (1,), "Mass randomization is only supported for the base body."

        masses = asset.root_physx_view.get_masses()
        print("*" * 50)
        print(f"masses before randomization: {masses[env_ids, body_ids]}")
        print("*" * 50)

        masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()

        masses[env_ids[:, None], body_ids] += math_utils.sample_uniform(
            -1.0, 2.0, (masses.shape[0], body_ids.shape[0]), device=masses.device
        )

        asset.root_physx_view.set_masses(masses, env_ids)

        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            inertias[env_ids[:, None], body_ids] = (
                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            )
        else:
            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
        asset.root_physx_view.set_inertias(inertias, env_ids)

        new_masses = asset.root_physx_view.get_masses()
        print("*" * 50)
        print(f"randomized masses: {new_masses[env_ids, body_ids]}")
        print("*" * 50)


    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape[1] != self.cfg.num_high_actions:
            raise ValueError(f"Expected number of actions {self.cfg.num_high_actions}, but got {actions.shape}")
        self._step_counter += 1
        self._iter_counter = self._step_counter // 24

        self._update_debug_draw()

        if self._step_counter < 1e10 and self.cfg.is_play_env and self.viewport_camera_controller is not None:
            lookat_pos = self._robot.data.root_pos_w[self._track_env_id].cpu()
            eye_pos = lookat_pos.clone()
            eye_pos[0] += 1.0
            eye_pos[1] += 1.0
            eye_pos[2] += self._track_camera_height
            self.viewport_camera_controller.update_view_location(eye=eye_pos, lookat=lookat_pos)

        self._prev_high_actions.append(self._high_actions.clone())
        self._prev_high_actions.pop(0)
        self._high_actions = actions.clone()

        self._cmd_goal_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self._robot.data.root_quat_w), self._cmd_goal_w - self._robot.data.root_pos_w
        )
        cmd_goal_dir_norm = torch.norm(self._cmd_goal_b, dim=-1)
        self._cmd_goal_dir_b = self._cmd_goal_b / (cmd_goal_dir_norm[:, None] + 1e-6)
        self._cmd_goal_dir_b[cmd_goal_dir_norm < 0.1] = 0.0
        self._cmd_goal_heading_b = torch.atan2(self._cmd_goal_dir_b[:, 1], self._cmd_goal_dir_b[:, 0]).unsqueeze(-1)

        filter_proprio_obs = torch.cat(
            [
                self._robot.data.root_com_ang_vel_b * 0.25,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self._high_actions,  # 3 (velocity command)
                self._filter_actions,  # 3 (previous filter action)
            ],
            dim=-1,
        )
        self._a2_filter_input = filter_proprio_obs.clone()
        self._filter_actions = self._filter_policy(filter_proprio_obs, self._obs_actor_rays)

        self._prev_loco_actions.append(self._loco_actions.clone())
        self._prev_loco_actions.pop(0)
        self._a2_loco.commands[:] = self._filter_actions.clamp(min=-self._vel_cmd_limits, max=self._vel_cmd_limits)
        self._a2_observation = self._a2_loco.observation().clone()

        self._update_dynamic_obstacles()


    def _apply_action(self):
        effort = self._a2_loco.apply()
        self._a2_last_effort = effort.clone()
        self._loco_actions[:, self._a2_loco.joints] = self._a2_loco.previous_action
        if not torch.isfinite(self._a2_observation).all() or not torch.isfinite(effort).all():
            raise RuntimeError("A2 观测或 PD 力矩出现非有限数值")


    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_mask[env_ids] = True

        self._prev_grid_buf[env_mask] = 0.0
        self._prev_proj_gravity_buf[env_mask] = 0.0
        self._prev_ang_vel_buf[env_mask] = 0.0

        self._update_curriculum(env_ids)

        self._robot.reset(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)

        self._high_actions[env_ids] = 0.0
        self._filter_actions[env_ids] = 0.0
        self._loco_actions[env_ids] = 0.0
        for actions in self._prev_high_actions:
            actions[env_ids] = 0.0
        for actions in self._prev_loco_actions:
            actions[env_ids] = 0.0
        self._ttc_full[env_ids] = 3.0
        self._obs_buf[env_ids] = 0.0
        self._obs_actor_rays[env_ids] = 1.0

        self._reset_prev_extra_buf(env_ids)

        self._reset_robot_and_cmd(env_ids)

        self._last_dist_to_goal[env_ids] = torch.norm(
            self._cmd_goal_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2], dim=-1
        )

        self._reset_obstacles(env_ids)

        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        default_joint_pos *= math_utils.sample_uniform(0.6, 1.4, default_joint_pos.shape, default_joint_pos.device)
        joint_pos_limits = self._robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = default_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        self._robot.write_joint_state_to_sim(joint_pos, self._robot.data.default_joint_vel[env_ids], env_ids=env_ids)

        self._a2_loco.reset(env_ids)
        self._filter_policy.h_state[:, env_mask, :] = 0.0
        self._filter_policy.c_state[:, env_mask, :] = 0.0

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self._episode_length_sec
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Episode/average_speed"] = torch.mean(self._robot.data.root_com_lin_vel_b.norm(dim=1))
        extras["Episode/average_terrain_level"] = torch.mean(self._terrain.terrain_levels.to(torch.float))
        extras["Episode/max_terrain_level"] = torch.max(self._terrain.terrain_levels.to(torch.float))
        extras["Episode/mean_active_obstacles"] = torch.mean(self._num_active_obstacles.to(torch.float))
        extras["Episode/mean_max_episode_length"] = torch.mean(self._episode_length_sec_per_env)
        extras["Episode/env_type_1"] = torch.sum(self._env_type == 1).item()
        extras["Episode/env_type_2"] = torch.sum(self._env_type == 2).item()
        extras["Episode/env_type_3"] = torch.sum(self._env_type == 3).item()
        self.extras["log"] = extras



def configure_a2(cfg):
    """仅替换机器人与机身路径；保留原传感器图样及环境事件分布。"""
    from isaaclab.managers import SceneEntityCfg
    cfg.robot = robot_cfg(asset_path()).replace(prim_path=cfg.robot.prim_path)
    for sensor in (cfg.raycaster, cfg.raycaster_measure):
        sensor.prim_path = sensor.prim_path.rsplit("/", 1)[0]+"/base_link"
    for name in ("base_external_force_torque", "base_com"):
        getattr(cfg.events, name).params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
    return cfg


class A2StaticEvaluationEnv(StaticEvaluationEnv, A2NavEnv):
    """复用既有固定路线、回合判据及记录；通过继承接入 A2 后端。"""

    def __init__(self, cfg, out, **kwargs):
        self.selective_reset_checks = 0
        self.control_period_checks = 0
        self.command_records = 0
        self.command_outside_candidate = 0
        super().__init__(cfg, out, **kwargs)
        self._a2_control_file = (Path(out)/"a2_control.jsonl").open("w")
        assert len(self._robot.joint_names) == 12 and len(self._robot.body_names) == 17
        assert self._contact_sensor.body_names[self._base_id_cs[0]] == "base_link"
        for actuator in self._robot.actuators.values():
            if torch.count_nonzero(actuator.stiffness) or torch.count_nonzero(actuator.damping):
                raise RuntimeError("A2 显式 PD 不应叠加非零的隐式增益")

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        untouched = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        untouched[env_ids] = False
        fields = ("history", "previous_action", "target", "commands", "phase", "physics_steps", "inference_calls")
        snapshot = {name: getattr(self._a2_loco, name)[untouched].clone() for name in fields}
        filter_states = {name: getattr(self._filter_policy, name)[:, untouched].clone() for name in ("h_state", "c_state")}
        super()._reset_idx(env_ids)
        for name, before in snapshot.items():
            if not torch.equal(getattr(self._a2_loco, name)[untouched], before):
                raise RuntimeError("A2 复位影响了其他环境: "+name)
        for name, before in filter_states.items():
            if not torch.equal(getattr(self._filter_policy, name)[:, untouched], before):
                raise RuntimeError("安全网络复位影响了其他环境: "+name)
        if untouched.any():
            self.selective_reset_checks += len(env_ids)

    def _get_dones(self):
        active = self.active_trials.clone()
        ended, expired = super()._get_dones()
        low = self._a2_loco
        # 每次控制周期恰有一次学生推理、四次物理步。
        if active.any():
            if not torch.equal(low.inference_calls[active], self.episode_length_buf[active]):
                raise RuntimeError("A2 推理次数与导航步数不一致")
            if not torch.equal(low.physics_steps[active], 4*self.episode_length_buf[active]):
                raise RuntimeError("A2 物理步调度不一致")
            if torch.count_nonzero(low.phase[active]):
                raise RuntimeError("控制周期结束后 A2 相位未回到零")
            expected = self._filter_actions.clamp(min=-self._vel_cmd_limits, max=self._vel_cmd_limits)
            if not torch.equal(low.commands[active], expected[active]):
                raise RuntimeError("安全层到 A2 的速度指令发生额外转换")
            if not torch.allclose(self._robot.data.applied_torque[active], self._a2_last_effort[active], atol=1e-5, rtol=1e-6):
                raise RuntimeError("A2 实际执行器输出与显式 PD 目标不一致")
            self.control_period_checks += int(active.sum())
            outside = (low.commands[active].abs() > torch.tensor([2., 1., 2.], device=self.device)).any(dim=-1)
            self.command_records += int(active.sum())
            self.command_outside_candidate += int(outside.sum())
        for index in active.nonzero().flatten().tolist():
            step = int(self.episode_length_buf[index])
            terminal = bool(ended[index] or expired[index])
            if step % 5 and not terminal:
                continue
            contact = self._contact_sensor.data.net_forces_w[index, self._undesired_contact_body_ids_cs]
            largest = int(contact.norm(dim=-1).argmax())
            row = {"trial_id": self.batch_cases[index]["trial_id"], "step": step, "time_s": step*self.step_dt,
                "terminal": terminal, "navigation_command": self._high_actions[index].cpu().tolist(),
                "filter_command_raw": self._filter_actions[index].cpu().tolist(),
                "navigation_observation": self._a2_navigation_input[index].cpu().tolist(),
                "filter_observation": self._a2_filter_input[index].cpu().tolist(),
                "rays_to_navigation_and_filter": self._a2_navigation_rays[index].cpu().tolist(),
                "locomotion_command": low.commands[index].cpu().tolist(),
                "locomotion_observation": self._a2_observation[index].cpu().tolist(),
                "action_policy_order": low.previous_action[index].cpu().tolist(),
                "effort_policy_order": self._a2_last_effort[index, low.joints].cpu().tolist(),
                "history_norm": float(low.history[index].norm()),
                "inference_calls": int(low.inference_calls[index]), "physics_steps": int(low.physics_steps[index]),
                "largest_current_contact_body": self._contact_sensor.body_names[self._undesired_contact_body_ids_cs[largest]],
                "largest_current_contact_force": contact[largest].cpu().tolist()}
            self._a2_control_file.write(json.dumps(row, allow_nan=False)+"\n")
        return ended, expired

    def close(self):
        stream = getattr(self, "_a2_control_file", None)
        if stream is not None:
            stream.close()
        super().close()
