"""冻结的原A2学生步态：20ms推理、5ms显式PD，独立于导航采样周期。"""
from pathlib import Path
import torch
from a2_migration.lab_robot import ROOT, robot_contract


class A2LowLevel:
    physics_dt = .005
    inference_dt = .020
    physics_steps_per_inference = 4

    def __init__(self, robot, policy_path=None):
        self.robot = robot
        self.contract = c = robot_contract()
        self.device = robot.device
        self.num_envs = robot.data.joint_pos.shape[0]
        self.joints = [robot.joint_names.index(name) for name in c['joint_names']]
        if len(robot.joint_names) != 12:
            raise ValueError('A2要求原12个关节，必须先验收资产')
        self.default = robot.data.default_joint_pos[:, self.joints].clone()
        self.kp = torch.tensor([c['control']['stiffness'][name.split('_', 1)[1]]
                               for name in c['joint_names']], device=self.device)
        self.kd = torch.tensor([c['control']['damping'][name.split('_', 1)[1]]
                               for name in c['joint_names']], device=self.device)
        self.limits = torch.tensor([float(c['joint_limits'][name]['effort'])
                                   for name in c['joint_names']], device=self.device)
        self.command_scale = torch.tensor([2., 2., .25], device=self.device)
        path = Path(policy_path) if policy_path else ROOT/'assets/a2/student_model_103500.pt'
        self.policy = torch.jit.load(str(path), map_location=self.device).eval()
        self.history = torch.zeros(self.num_envs, 5, 45, device=self.device)
        self.previous_action = torch.zeros(self.num_envs, 12, device=self.device)
        self.target = self.default.clone()
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.physics_steps = torch.zeros_like(self.phase)
        self.inference_calls = torch.zeros_like(self.phase)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

    def observation(self):
        r, c = self.robot, self.contract
        scales = c['normalization']['obs_scales']
        return torch.cat((r.data.root_ang_vel_b*scales['ang_vel'], r.data.projected_gravity_b,
            self.commands*self.command_scale, (r.data.joint_pos[:, self.joints]-self.default)*scales['dof_pos'],
            r.data.joint_vel[:, self.joints]*scales['dof_vel'], self.previous_action), -1).clamp(
                -c['normalization']['clip_observations'], c['normalization']['clip_observations'])

    def reset(self, env_ids):
        """只清所选环境；下一物理步立即推理，不改变其他机器人的相位。"""
        self.history[env_ids] = 0
        self.previous_action[env_ids] = 0
        self.commands[env_ids] = 0
        self.target[env_ids] = self.default[env_ids]
        self.phase[env_ids] = 0
        self.physics_steps[env_ids] = 0
        self.inference_calls[env_ids] = 0

    @torch.no_grad()
    def apply(self):
        """每个物理步之前调用一次；保持最近命令，20ms之间保持最近关节目标。"""
        due = (self.phase == 0).nonzero().flatten()
        if due.numel():
            action, history = self.policy(self.observation()[due], self.history[due])
            clip = self.contract['normalization']['clip_actions']
            action = action.clamp(-clip, clip)
            self.history[due] = history
            self.previous_action[due] = action
            self.target[due] = self.default[due]+self.contract['control']['action_scale']*action
            self.inference_calls[due] += 1
        r = self.robot
        tau = (self.kp*(self.target-r.data.joint_pos[:, self.joints])
               -self.kd*r.data.joint_vel[:, self.joints]).clamp(-self.limits, self.limits)
        effort = torch.zeros_like(r.data.joint_pos)
        effort[:, self.joints] = tau
        r.set_joint_effort_target(effort)
        self.phase.add_(1).remainder_(self.physics_steps_per_inference)
        self.physics_steps += 1
        return effort
