"""按首批正式初态重跑两例并保存姿态，再独立渲染录像；不改评估入口。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT/'results/evaluation/20261117_210209_reasan_a2_static'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'render'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rendering = args.mode == 'render'
    original = json.loads((BASELINE/'summary.json').read_text())
    for relative, expected in original['script_sha256'].items():
        path = ROOT/(relative if '/' in relative else 'scripts/'+relative)
        assert digest(path) == expected, str(path)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=rendering, device='cuda:0', kit_args=(
        '--/plugins/carb.tasking.plugin/threadCount=4 '
        '--/persistent/isaac/asset_root/cloud=/asset-cache')).app
    env = None
    try:
        import numpy as np
        import torch
        import go2_lidar.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from go2_lidar.terrain.train_terrain_cfg import GO2_TEST_TERRAIN_CFG
        from scripts.static_eval_env import FrozenTestTerrain
        from a2_adapter.nav_env import A2StaticEvaluationEnv, configure_a2
        torch.set_num_threads(4)
        trials = json.loads((BASELINE/'trials.json').read_text())[:2]
        assert [c['trial_id'] for c in trials] == ['sparse_02_r00', 'sparse_03_r00']
        tracks = [[], []]

        class CaptureEnv(A2StaticEvaluationEnv):
            def _get_dones(self):
                # 在原自动复位前保存每个有效控制周期的真实关节和机身姿态。
                if not rendering:
                    for index in self.active_trials.nonzero().flatten().tolist():
                        tracks[index].append(np.concatenate((
                            [float(self.episode_length_buf[index])*self.step_dt],
                            self._robot.data.root_link_pose_w[index].cpu().numpy(),
                            self._robot.data.joint_pos[index].cpu().numpy())))
                return super()._get_dones()

        cfg = parse_env_cfg('Unitree-Go2-Navigation', device='cuda:0', num_envs=2)
        cfg.seed = 0
        cfg.is_play_env = True
        cfg.wait_for_key = False
        cfg.use_dynamic_obstacle = False
        cfg.use_predicted_rays = False
        cfg.events.push_robot = None
        cfg.set_raycaster_measure_pattern('11x')
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
        cfg.terrain.terrain_generator = GO2_TEST_TERRAIN_CFG.copy()
        cfg.terrain.terrain_generator.class_type = FrozenTestTerrain
        cfg.terrain.terrain_generator.frozen_asset = str(ROOT/'assets/evaluation/public_static_v1/terrain.npz')
        cfg.viewer.resolution = (1280, 720)
        configure_a2(cfg)
        logdir = args.out/('render_logs' if rendering else 'replay')
        logdir.mkdir(exist_ok=True)
        env = CaptureEnv(cfg=cfg, out=logdir, render_mode='rgb_array' if rendering else None)
        torch.manual_seed(10000)
        with torch.inference_mode():
            observations, _ = env.set_batch(trials)
            if not rendering:
                for index in range(2):
                    tracks[index].append(np.concatenate(([0.],
                        env._robot.data.root_link_pose_w[index].cpu().numpy(),
                        env._robot.data.joint_pos[index].cpu().numpy())))
                policy_path = ROOT/'upstream/REASAN/training/logs/rsl_rl/go2_lidar/nav_1/exported/policy.pt'
                policy = torch.jit.load(str(policy_path), map_location='cuda:0').eval()
                for name in ('h_state', 'c_state'):
                    state = getattr(policy, name)
                    setattr(policy, name, torch.zeros(state.shape[0], 2, state.shape[2], device='cuda:0'))
                for _ in range(751):
                    obs = observations['policy']
                    p0,p1 = cfg.obs_ranges['proprio']; r0,r1 = cfg.obs_ranges['actor_ray']
                    env._a2_navigation_input = obs[:,p0:p1].clone()
                    env._a2_navigation_rays = obs[:,r0:r1].clone()
                    assert torch.equal(env._a2_navigation_rays, env._obs_actor_rays)
                    action = policy(obs[:,p0:p1], obs[:,r0:r1])
                    action[~env.active_trials] = 0
                    observations,_,terminated,truncated,_ = env.step(action)
                    done = terminated | truncated
                    for name in ('h_state','c_state'):
                        getattr(policy,name)[:,done] = 0
                    if not env.active_trials.any():
                        break
                assert len(env.episodes) == 2
                originals = {e['trial_id']:e for e in map(json.loads,(BASELINE/'episodes.jsonl').read_text().splitlines())}
                comparisons = []
                for e in env.episodes:
                    prior = originals[e['trial_id']]
                    assert e['outcome'] == prior['outcome']
                    comparisons.append({'trial_id':e['trial_id'],'outcome':e['outcome'],
                        'original_time_s':prior['time_s'],'replay_time_s':e['time_s'],
                        'terminal_position_difference_m':float(np.linalg.norm(np.array(e['position'])-prior['position']))})
                np.savez_compressed(args.out/'poses.npz', **{c['trial_id']:np.array(tracks[i]) for i,c in enumerate(trials)})
                (logdir/'initial_states.json').write_text(json.dumps(env.recorded_initial_states,indent=2)+'\n')
                initial_original = {e['trial_id']:e for e in json.loads((BASELINE/'initial_states.json').read_text())}
                for state in env.recorded_initial_states:
                    for key in ('position','quaternion','joint_position','goal'):
                        np.testing.assert_allclose(state[key],initial_original[state['trial_id']][key],atol=1e-6)
                metadata = {'source_evaluation':str(BASELINE.relative_to(ROOT)), 'kind':'new replay for recording; excluded from formal statistics',
                    'models':original['models'],'terrain_sha256':original['terrain_sha256'],
                    'recorder_sha256':digest(Path(__file__)), 'comparisons':comparisons,
                    'control_period_checks':env.control_period_checks,'joint_names':env._robot.joint_names}
                (args.out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
                print('CAPTURE COMPLETE',json.dumps(comparisons),flush=True)
            else:
                render_videos(env, trials, args.out, torch, np)
    finally:
        if env is not None:
            env.close()
        from isaaclab.sim import SimulationContext
        sim = SimulationContext.instance()
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        app.close(wait_for_replicator=False)


def render_videos(env, trials, out, torch, np):
    """只写入已保存的姿态并渲染，不推进物理或重新计算策略。"""
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    from isaaclab.markers import VisualizationMarkers
    import isaaclab.sim as sim_utils
    # 离线资产缺少原地形的 MDL 纹理，仅在姿态回放中替换可视材质。
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.38,.41,.45), roughness=.85)
    material.func('/World/VideoGroundMaterial',material)
    sim_utils.bind_visual_material('/World/ground/terrain','/World/VideoGroundMaterial')
    # 原白色机器人在强环境光下对比不足；回放时采用深蓝显示材质。
    robot_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.045,.09,.16),roughness=.65)
    robot_material.func('/World/VideoRobotMaterial',robot_material)
    for index in range(2):
        sim_utils.bind_visual_material(f'/World/envs/env_{index}/Robot','/World/VideoRobotMaterial')
    import omni.usd
    from pxr import UsdShade
    stage = omni.usd.get_context().get_stage()
    for prim in list(stage.Traverse()):
        if '/Robot' in str(prim.GetPath()) and prim.IsInstance():
            prim.SetInstanceable(False)
    # 导入模型可能有 full-purpose 材质绑定，逐个可视网格覆盖该显示绑定。
    surface = UsdShade.Material(stage.GetPrimAtPath('/World/VideoRobotMaterial'))
    material_count = 0
    for prim in stage.Traverse():
        if '/Robot/' in str(prim.GetPath()) and prim.GetTypeName() in ('Mesh','GeomSubset'):
            binding=UsdShade.MaterialBindingAPI.Apply(prim)
            for purpose in (UsdShade.Tokens.allPurpose,UsdShade.Tokens.full,UsdShade.Tokens.preview):
                binding.Bind(surface,UsdShade.Tokens.strongerThanDescendants,purpose)
            material_count += 1
    print('VIDEO ROBOT DISPLAY MESHES',material_count,flush=True)
    stage.GetPrimAtPath('/World/skyLight').GetAttribute('inputs:intensity').Set(900.)
    for value in vars(env).values():
        if isinstance(value, VisualizationMarkers):
            value.set_visibility(False)
    goals=[]
    for i,c in enumerate(trials):
        sphere = sim_utils.SphereCfg(radius=.13, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.85,.08,.12)))
        sphere.func('/World/VideoGoal'+str(i),sphere,translation=(*c['goal'],.18))
    data = np.load(out/'poses.npz')
    with np.load(ROOT/'assets/evaluation/public_static_v1/terrain.npz') as terrain:
        vertices, faces = terrain['vertices'], terrain['faces']
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,21) if Path(fontpath).exists() else ImageFont.load_default(size=21)
    small=ImageFont.truetype(fontpath,17) if Path(fontpath).exists() else ImageFont.load_default(size=17)
    episodes={e['trial_id']:e for e in map(json.loads,(out/'replay/episodes.jsonl').read_text().splitlines())}
    reports=[]
    for index,case in enumerate(trials):
        track=data[case['trial_id']]
        episode=episodes[case['trial_id']]
        outcome=episode['outcome']
        basename=('success_' if outcome=='success' else 'failure_')+case['trial_id']
        video=out/(basename+'.mp4')
        # 25 fps 原速回放；初态停留 1 秒、终止停留 2 秒，不延长仿真。
        selected=list(range(0,len(track),2))
        if selected[-1] != len(track)-1:
            selected.append(len(track)-1)
        xmin,xmax,ymin,ymax=case['bounds']
        mapw,maph=340,184
        base=Image.new('RGB',(mapw,maph),(248,248,248))
        draw=ImageDraw.Draw(base)
        def xy(pos):
            return (int(10+(pos[0]-xmin)/(xmax-xmin)*(mapw-20)),int(maph-10-(pos[1]-ymin)/(ymax-ymin)*(maph-20)))
        inside=((vertices[:,0]>=xmin)&(vertices[:,0]<=xmax)&(vertices[:,1]>=ymin)&(vertices[:,1]<=ymax)&(vertices[:,2]>.10))
        triangles=vertices[faces[np.any(inside[faces],axis=1)]]
        for tri in triangles[triangles[:,:,2].mean(axis=1)>.10]:
            draw.polygon([xy(v) for v in tri], fill=(160,164,170))
        start,goal=xy(case['start']),xy(case['goal'])
        draw.ellipse((start[0]-4,start[1]-4,start[0]+4,start[1]+4), fill=(25,135,65))
        draw.ellipse((goal[0]-5,goal[1]-5,goal[0]+5,goal[1]+5), fill=(210,25,45))
        writer=imageio.get_writer(str(video),fps=25,codec='libx264',quality=8,macro_block_size=8,
            ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'])
        frames=0; keyframes=[]
        for sequence,k in enumerate(selected):
            state=track[k]
            ids=torch.tensor([index],device=env.device)
            env._robot.write_root_link_pose_to_sim(torch.tensor(state[1:8],dtype=torch.float32,device=env.device)[None],ids)
            env._robot.write_root_com_velocity_to_sim(torch.zeros((1,6),device=env.device),env_ids=ids)
            q=torch.tensor(state[8:],dtype=torch.float32,device=env.device)[None]
            env._robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
            env.sim.forward()
            pos=state[1:4]
            target=(float(pos[0]+1.2),float(pos[1]),.45)
            env.sim.set_camera_view(eye=(float(pos[0]-4.2),float(pos[1]-6.2),5.7),target=target)
            # 初帧等待离屏渲染产品就绪；随后双次渲染消除镜头上一帧残留。
            for _ in range(12 if sequence==0 else 2):
                rgb=env.render()
            assert rgb.shape==(720,1280,3) and rgb.std()>5, rgb.shape
            frame=Image.fromarray(rgb)
            mini=base.copy(); d=ImageDraw.Draw(mini)
            path=[xy(p) for p in track[:k+1,1:3]]
            if len(path)>1: d.line(path,fill=(25,105,210),width=3)
            now=xy(pos);d.ellipse((now[0]-4,now[1]-4,now[0]+4,now[1]+4),fill=(20,95,225))
            frame.paste(mini,(920,16))
            d=ImageDraw.Draw(frame)
            terminal=k==len(track)-1
            text=f"t={state[0]:.2f}s"+(f"  {outcome}" if terminal else '')
            d.rectangle((16,16,445,51),fill=(25,30,38));d.text((26,23),text,font=font,fill='white')
            if terminal and outcome!='success':
                d.rectangle((16,58,615,90),fill=(25,30,38))
                d.text((26,64),'calf contact > 1 N; episode terminated',font=small,fill='white')
            array=np.asarray(frame)
            repeat=25 if sequence==0 else 1
            if terminal: repeat=50
            for _ in range(repeat):writer.append_data(array);frames+=1
            for label,chosen in [('start',0),('middle',len(selected)//2),('end',len(selected)-1)]:
                if sequence==chosen:
                    name=basename+'_'+label+'.jpg';frame.save(out/name,quality=92);keyframes.append(name)
            if sequence%50==0: print(f'RENDER {basename}: {sequence+1}/{len(selected)}',flush=True)
        writer.close()
        reports.append({'trial_id':case['trial_id'],'outcome':outcome,'video':video.name,'fps':25,'frames':frames,
                        'duration_s':frames/25,'simulation_terminal_time_s':float(track[-1,0]),
                        'initial_hold_s':1,'terminal_hold_s':2,'size_bytes':video.stat().st_size,'sha256':digest(video),'keyframes':keyframes})
    (out/'videos.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('VIDEOS COMPLETE',json.dumps(reports),flush=True)


if __name__=='__main__':
    main()
