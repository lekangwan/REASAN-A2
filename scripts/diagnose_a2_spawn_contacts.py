"""在独立短跑中记录 A2 出生接触对；保留原评估实现与设置。"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'results/evaluation/20261117_210209_reasan_a2_static'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda:0'])
    p.add_argument('--batches',type=int,default=1)
    p.add_argument('--source-result',type=Path,default=BASELINE)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device=args.device,kit_args=(
        '--/plugins/carb.tasking.plugin/threadCount=4 --/persistent/isaac/asset_root/cloud=/asset-cache')).app
    env=None
    result={'complete':False,'device':args.device,'physics_backend_matches_formal':args.device=='cuda:0',
            'kind':'short diagnostic with contact-pair readout only; excluded from evaluation statistics'}
    try:
        import numpy as np
        import torch
        import go2_lidar.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from go2_lidar.terrain.train_terrain_cfg import GO2_TEST_TERRAIN_CFG
        from scripts.static_eval_env import FrozenTestTerrain
        from a2_adapter.nav_env import A2StaticEvaluationEnv,configure_a2
        torch.set_num_threads(4)
        contacts=[]
        class DiagnosticEnv(A2StaticEvaluationEnv):
            def _get_dones(self):
                if hasattr(self,'pair_views'):
                    for index,view in enumerate(self.pair_views):
                        if not self.active_trials[index]:continue
                        forces=view.get_contact_force_matrix(.005).cpu().numpy()
                        net=view.get_net_contact_forces(.005).cpu().numpy()
                        detail=view.get_contact_data(.005)
                        f,points,normals,separations,counts,starts=[x.cpu().numpy() for x in detail]
                        pairs=[]
                        for a,b in np.argwhere(np.linalg.norm(forces,axis=-1)>.05):
                            start,count=int(starts[a,b]),int(counts[a,b]);sl=slice(start,start+count)
                            pairs.append({'sensor':view.sensor_paths[a],'filter':view.filter_paths[a][b],
                                'force':forces[a,b].tolist(),'contact_count':count,
                                'positions':points[sl].tolist(),'normals':normals[sl].tolist(),
                                'separations':separations[sl].flatten().tolist()})
                        contacts.append({'trial_id':self.batch_cases[index]['trial_id'],
                            'step':int(self.episode_length_buf[index]),'time_s':float(self.episode_length_buf[index])*self.step_dt,
                            'pairs':pairs,'net_forces':net.tolist(),'sensor_paths':list(view.sensor_paths),
                            'body_positions':self._robot.data.body_pos_w[index].cpu().tolist(),
                            'pair_sum_minus_net_max_n':float(np.abs(forces.sum(axis=1)-net).max())})
                return super()._get_dones()
        cfg=parse_env_cfg('Unitree-Go2-Navigation',device=args.device,num_envs=2)
        cfg.seed=0;cfg.is_play_env=True;cfg.wait_for_key=False
        cfg.use_dynamic_obstacle=False;cfg.use_predicted_rays=False;cfg.events.push_robot=None
        cfg.set_raycaster_measure_pattern('11x')
        cfg.sim.physx.gpu_max_rigid_patch_count=2**18
        cfg.terrain.terrain_generator=GO2_TEST_TERRAIN_CFG.copy()
        cfg.terrain.terrain_generator.class_type=FrozenTestTerrain
        cfg.terrain.terrain_generator.frozen_asset=str(ROOT/'assets/evaluation/public_static_v1/terrain.npz')
        configure_a2(cfg)
        env=DiagnosticEnv(cfg=cfg,out=args.out)
        env.pair_views=[]
        filters=[]
        for index in range(2):
            own=f'/World/envs/env_{index}/Robot/'
            candidates=['/World/ground/terrain/mesh']+[own+body for body in env._robot.body_names]+[f'/World/envs/env_{1-index}/Robot/'+body for body in env._robot.body_names]
            view=env._contact_sensor._physics_sim_view.create_rigid_contact_view(own+'*',filter_patterns=candidates,max_contact_data_count=8192)
            env.pair_views.append(view)
            filters.append({'sensors':list(view.sensor_paths),'filters':view.filter_paths,'shape':[view.sensor_count,view.filter_count]})
        result['contact_view_paths']=filters
        policy=torch.jit.load(str(ROOT/'upstream/REASAN/training/logs/rsl_rl/go2_lidar/nav_1/exported/policy.pt'),map_location=args.device).eval()
        for name in ('h_state','c_state'):
            state=getattr(policy,name);setattr(policy,name,torch.zeros(state.shape[0],2,state.shape[2],device=args.device))
        # 张量过滤矩阵在此版本可能不完整；额外订阅原始 PhysX 接触报告，直接记录双方碰撞体。
        import omni.physx
        from pxr import PhysicsSchemaTools
        raw=[]
        def on_contact(headers,data):
            for header in headers:
                actors=[str(PhysicsSchemaTools.intToSdfPath(header.actor0)),str(PhysicsSchemaTools.intToSdfPath(header.actor1))]
                indices=[i for i in range(2) if any(f'/env_{i}/Robot/' in a for a in actors)]
                if not indices or not env.batch_cases:continue
                active=[i for i in indices if bool(env.active_trials[i])]
                if not active:continue
                points=[]
                for j in range(header.contact_data_offset,header.contact_data_offset+header.num_contact_data):
                    d=data[j]
                    points.append({'position':list(d.position),'normal':list(d.normal),'impulse':list(d.impulse),
                                   'separation':float(d.separation),'face_index0':int(d.face_index0),'face_index1':int(d.face_index1)})
                raw.append({'trials':[env.batch_cases[i]['trial_id'] for i in active],
                    'physics_steps':[int(env._a2_loco.physics_steps[i]) for i in active],
                    'actors':actors,'colliders':[str(PhysicsSchemaTools.intToSdfPath(header.collider0)),str(PhysicsSchemaTools.intToSdfPath(header.collider1))],
                    'type':str(header.type),'points':points})
        subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
        all_trials=json.loads((args.source_result/'trials.json').read_text())
        if not 0<args.batches*2<=len(all_trials):raise ValueError('诊断批次数超出来源清单')
        initial=[]
        with torch.inference_mode():
            for batch_number in range(args.batches):
                # 前面批次只运行起步部分，诊断后主动复位；不将其当成正式测试。
                env.active_trials[:]=False
                torch.manual_seed(10000+batch_number)
                observations,_=env.set_batch(all_trials[batch_number*2:batch_number*2+2])
                initial.extend(env.recorded_initial_states[-2:])
                policy.h_state.zero_();policy.c_state.zero_()
                for _ in range(5):
                    obs=observations['policy'];p0,p1=cfg.obs_ranges['proprio'];r0,r1=cfg.obs_ranges['actor_ray']
                    env._a2_navigation_input=obs[:,p0:p1].clone();env._a2_navigation_rays=obs[:,r0:r1].clone()
                    action=policy(obs[:,p0:p1],obs[:,r0:r1]);action[~env.active_trials]=0
                    observations,_,ended,expired,_=env.step(action)
                    done=ended|expired
                    policy.h_state[:,done]=0;policy.c_state[:,done]=0
                    if not env.active_trials.any():break
        result.update(complete=True,episodes=env.episodes,initial_states=initial,
            control_period_checks=env.control_period_checks,
            self_collisions=cfg.robot.spawn.articulation_props.enabled_self_collisions,
            recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        original_initial={r['trial_id']:r for r in json.loads((args.source_result/'initial_states.json').read_text())}
        for state in initial:
            for key in ('position','quaternion','joint_position','goal'):
                np.testing.assert_allclose(state[key],original_initial[state['trial_id']][key],atol=1e-6)
        result['initial_states_match_source']=True
        result['source_result']=str(args.source_result)
        (args.out/'contacts.json').write_text(json.dumps(contacts,indent=2,allow_nan=False)+'\n')
        (args.out/'raw_contacts.json').write_text(json.dumps(raw,indent=2,allow_nan=False)+'\n')
        result['raw_contact_events']=len(raw)
        subscription=None
        print('DIAGNOSTIC COMPLETE',[(e['trial_id'],e['outcome'],e['time_s']) for e in env.episodes],flush=True)
    except Exception:
        result['error']=traceback.format_exc();traceback.print_exc()
    finally:
        (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if env is not None:env.close()
        from isaaclab.sim import SimulationContext
        sim=SimulationContext.instance()
        if sim is not None:sim.clear_all_callbacks();sim.clear_instance()
        app.close(wait_for_replicator=False)
    if not result['complete']:raise SystemExit(1)


if __name__=='__main__':main()
