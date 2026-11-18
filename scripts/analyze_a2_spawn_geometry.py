"""按保存初态和原 URDF 前向运动学，检查出生碰撞盒与冻结地形关系。"""
import argparse,itertools,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def rotation(axis,angle):
    a=np.asarray(axis,dtype=float);a/=np.linalg.norm(a);x,y,z=a
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)


def transform(origin):
    t=np.eye(4)
    if origin is None:return t
    t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
    r,p,y=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
    t[:3,:3]=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r)
    return t


def triangle_hits(center,r,half,tri):
    if len(tri)==0:return np.zeros(0,dtype=bool)
    v=(tri-center)@r
    e=np.stack([v[:,1]-v[:,0],v[:,2]-v[:,1],v[:,0]-v[:,2]],axis=1)
    normal=np.cross(e[:,0],e[:,1])
    cross=np.cross(e[:,:,None,:],np.eye(3)[None,None,:,:]).reshape(len(v),9,3)
    axes=np.concatenate([normal[:,None,:],cross],axis=1)
    proj=np.einsum('nvc,nac->nva',v,axes)
    radius=np.abs(axes)@half
    separated=((proj.min(axis=1)>radius+1e-8)|(proj.max(axis=1)<-radius-1e-8)).any(axis=1)
    separated|=((v.min(axis=1)>half+1e-8)|(v.max(axis=1)<-half-1e-8)).any(axis=1)
    return ~separated


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    initial=json.loads((args.result/'initial_states.json').read_text())
    summary=json.loads((args.result/'summary.json').read_text())
    joint_names=summary['runtime']['joint_names']
    episodes={r['trial_id']:r for r in map(json.loads,(args.result/'episodes.jsonl').read_text().splitlines())}
    urdf=ET.parse(ROOT/'assets/a2/usd_v1/a2_import.urdf').getroot()
    with np.load(ROOT/'assets/evaluation/public_static_v1/terrain.npz') as data:
        vertices=data['vertices'];faces=data['faces']
    triangles=vertices[faces];tmin=triangles.min(axis=1);tmax=triangles.max(axis=1)
    corners=np.array(list(itertools.product([-1,1],repeat=3)))
    output=[]
    for case in initial:
        q=dict(zip(joint_names,case['joint_position']))
        transforms={'base_link':np.eye(4)}
        root=transforms['base_link'];root[:3,3]=case['position']
        w,x,y,z=case['quaternion'];root[:3,:3]=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        pending=list(urdf.findall('joint'))
        while pending:
            for j in pending[:]:
                parent=j.find('parent').get('link');child=j.find('child').get('link')
                if parent not in transforms:continue
                t=transform(j.find('origin'))
                if j.get('type') in ('revolute','continuous'):
                    t[:3,:3]=t[:3,:3]@rotation(np.fromstring(j.find('axis').get('xyz'),sep=' '),q[j.get('name')])
                transforms[child]=transforms[parent]@t;pending.remove(j)
        shapes=[];bounds=[]
        for link in urdf.findall('link'):
            name=link.get('name')
            for index,c in enumerate(link.findall('collision')):
                t=transforms[name]@transform(c.find('origin'));g=c.find('geometry')[0]
                if g.tag=='box':half=np.fromstring(g.get('size'),sep=' ')/2
                elif g.tag=='sphere':half=np.full(3,float(g.get('radius')))
                elif g.tag=='cylinder':
                    # USD 将原圆柱替换为胶囊，额外加半径，形成保守的包围盒。
                    radius=float(g.get('radius'));half=np.array([radius,radius,float(g.get('length'))/2+radius])
                else:raise ValueError(g.tag)
                world=(corners*half)@t[:3,:3].T+t[:3,3]
                bounds.append(world)
                if g.tag=='box':shapes.append((name,index,t,half,world))
        all_points=np.concatenate(bounds)
        lo,hi=all_points.min(axis=0),all_points.max(axis=0)
        # 外扩 2 厘米，比接触偏移留出更多空间；若这里无高障碍面，则所有碰撞形状均不在障碍内。
        nearby=((tmax[:,:2]>=lo[:2]-.02)&(tmin[:,:2]<=hi[:2]+.02)).all(axis=1)
        obstacle_candidates=nearby&(tmax[:,2]>.10)
        floor_candidates=nearby&(tmax[:,2]<=.10)
        local_floor=triangles[floor_candidates]
        local_triangles=triangles[nearby]
        local_min=tmin[nearby];local_max=tmax[nearby]
        body={}
        for name,index,t,half,world in shapes:
            nearby_shape=((local_max>=world.min(axis=0)-.02)&(local_min<=world.max(axis=0)+.02)).all(axis=1)
            hit=triangle_hits(t[:3,3],t[:3,:3],half,local_triangles[nearby_shape])
            hit_triangles=local_triangles[nearby_shape][hit]
            d=body.setdefault(name,{'minimum_box_z_m':float('inf'),'intersecting_floor_faces':0,'intersecting_high_obstacle_faces':0})
            d['minimum_box_z_m']=min(d['minimum_box_z_m'],float(world[:,2].min()))
            d['intersecting_floor_faces']+=int((hit_triangles[:,:,2].max(axis=1)<=.10).sum())
            d['intersecting_high_obstacle_faces']+=int((hit_triangles[:,:,2].max(axis=1)>.10).sum())
        e=episodes.get(case['trial_id'],{})
        output.append({'trial_id':case['trial_id'],'outcome':e.get('outcome'),'time_s':e.get('time_s'),
          'collision_geometry_bounds':[lo.tolist(),hi.tolist()],
          'high_obstacle_faces_within_expanded_xy_bounds':int(obstacle_candidates.sum()),
          'floor_z_range_m':[float(local_floor[:,:,2].min()),float(local_floor[:,:,2].max())] if len(local_floor) else None,
          'body_boxes':body})
    args.out.write_text(json.dumps({'scope':'saved initial poses and source URDF geometry only; no dynamics or parameter changes',
        'terrain_note':'high obstacle face means triangle max z > 0.10 m; capsule bounds conservatively expanded',
        'cases':output},indent=2)+'\n')
    failures=[c for c in output if c['outcome']=='collision' and c['time_s']<=.1]
    print('cases',len(output),'early failures',len(failures),'with nearby high obstacle faces',sum(c['high_obstacle_faces_within_expanded_xy_bounds']>0 for c in failures))
    for c in failures:
        calf={k:round(v['minimum_box_z_m'],4) for k,v in c['body_boxes'].items() if k.endswith('_calf')}
        print(c['trial_id'],calf,'floor',c['floor_z_range_m'],'obstacle_faces',c['high_obstacle_faces_within_expanded_xy_bounds'])


if __name__=='__main__':main()
