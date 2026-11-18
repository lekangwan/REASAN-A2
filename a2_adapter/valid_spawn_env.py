"""A2 合法出生高度的独立评估版本；原环境与原始评估入口保持可用。"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from a2_adapter.nav_env import A2StaticEvaluationEnv
from a2_adapter.spawn_geometry import ROOT,VERSION,SpawnGeometry,quaternion_rotation


class GeometrySpawnA2EvaluationEnv(A2StaticEvaluationEnv):
    def __init__(self,cfg,out,**kwargs):
        self.spawn_records=[]
        self._spawn_cache={}
        super().__init__(cfg,out,**kwargs)
        from pxr import PhysxSchema
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        ground=stage.GetPrimAtPath('/World/ground/terrain/mesh')
        offset=PhysxSchema.PhysxCollisionAPI(ground).GetContactOffsetAttr().Get()
        description=ROOT/'assets/a2/spawn_geometry_v1/colliders.json'
        self.spawn_geometry=SpawnGeometry(json.loads(description.read_text()),cfg.terrain.terrain_generator.frozen_asset)
        self.spawn_configuration={'version':VERSION,'colliders_sha256':hashlib.sha256(description.read_bytes()).hexdigest(),
            'robot_contact_offset_m':cfg.robot.spawn.collision_props.contact_offset,
            'ground_contact_offset_m':float(offset) if offset is not None and np.isfinite(offset) else 'automatic',
            'clearance_definition':'fixed geometric gap; automatic ground contactOffset is left unchanged',
            'total_clearance_m':self.spawn_geometry.contact_clearance,
            'change':'raise z only; same xy, yaw, q sample, controllers, termination and clocks'}
        (Path(out)/'spawn_configuration.json').write_text(json.dumps(self.spawn_configuration,indent=2)+'\n')

    def _reset_idx(self,env_ids):
        if env_ids is None:env_ids=self._robot._ALL_INDICES
        untouched=torch.ones(self.num_envs,dtype=torch.bool,device=self.device);untouched[env_ids]=False
        other_pose=self._robot.data.root_link_pose_w[untouched].clone()
        other_q=self._robot.data.joint_pos[untouched].clone()
        super()._reset_idx(env_ids)
        original=self._robot.data.root_link_pose_w[env_ids].clone()
        q=self._robot.data.joint_pos[env_ids].clone()
        changed=original.clone()
        plans=[]
        for row,index in enumerate(env_ids.tolist()):
            case=self.batch_cases[index]
            pose=original[row].cpu().numpy();values=q[row].cpu().numpy()
            signature=(tuple(pose),tuple(values))
            key=case['trial_id']
            if key not in self._spawn_cache:
                plan=self.spawn_geometry.plan(pose,dict(zip(self._robot.joint_names,values)))
                self._spawn_cache[key]=(signature,plan)
            previous,plan=self._spawn_cache[key]
            if previous!=signature:raise RuntimeError('同一路线的出生状态发生未预期改变')
            changed[row,2]=plan['root_height_m'];plans.append(plan)
        # 只写入位姿，不推进物理、等待落地或跳过终止检查。
        self._robot.write_root_link_pose_to_sim(changed,env_ids=env_ids)
        self._robot_start_pos[env_ids]=changed[:,:3]
        from isaaclab.utils import math as math_utils
        goal=math_utils.quat_apply_inverse(math_utils.yaw_quat(changed[:,3:7]),self._cmd_goal_w[env_ids]-changed[:,:3])
        self._cmd_goal_b[env_ids]=goal
        self._cmd_goal_dir_b[env_ids]=goal/goal.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        self._cmd_goal_heading_b[env_ids]=torch.atan2(goal[:,1],goal[:,0])[:,None]
        self.sim.forward()
        if torch.count_nonzero(self._a2_loco.physics_steps[env_ids]) or torch.count_nonzero(self._a2_loco.inference_calls[env_ids]):
            raise RuntimeError('出生高度修正推进了物理或策略时钟')
        if not torch.equal(self._robot.data.joint_pos[env_ids],q):raise RuntimeError('高度修正改变了关节采样')
        if not torch.allclose(self._robot.data.root_link_pose_w[untouched],other_pose,atol=1e-7,rtol=1e-6):raise RuntimeError('高度修正影响了其他环境位置')
        if not torch.equal(self._robot.data.joint_pos[untouched],other_q):raise RuntimeError('高度修正影响了其他环境关节')
        actual=self._robot.root_physx_view.get_link_transforms()[env_ids].cpu().numpy()
        for row,index in enumerate(env_ids.tolist()):
            root=changed[row].cpu().numpy();values=q[row].cpu().numpy()
            fk=self.spawn_geometry.forward_kinematics(root,dict(zip(self._robot.joint_names,values)))
            live={}
            error=0.
            for b,name in enumerate(self._robot.body_names):
                t=np.eye(4);t[:3,3]=actual[row,b,:3]
                t[:3,:3]=quaternion_rotation(actual[row,b,[6,3,4,5]])
                error=max(error,float(np.abs(t-fk[name]).max()));live[name]=t
            if error>2e-5:raise RuntimeError('URDF 与实际 USD 关节几何不一致: '+str(error))
            bounds=self.spawn_geometry.bounds(live)
            low=min(float(b['min'][2]) for b in bounds)
            clearance=low-plans[row]['floor_max_z_m']
            if clearance<self.spawn_geometry.contact_clearance-2e-5:raise RuntimeError('修正后出生碰撞几何余量不足')
            self.spawn_records.append({'trial_id':self.batch_cases[index]['trial_id'],
                'active':bool(self.active_trials[index]),**plans[row],
                'original_root_pose':original[row].cpu().tolist(),'corrected_root_pose':root.tolist(),
                'joint_position':values.tolist(),'runtime_geometry_clearance_m':clearance,
                'kinematics_max_error':error,'physics_steps':int(self._a2_loco.physics_steps[index]),
                'inference_calls':int(self._a2_loco.inference_calls[index])})

    def close(self):
        if hasattr(self,'output_dir') and hasattr(self,'spawn_records'):
            (self.output_dir/'spawn_records.json').write_text(json.dumps(self.spawn_records,indent=2)+'\n')
        super().close()
