"""根据实际 USD 碰撞体和冻结地面计算合法出生高度；不包含策略或终止处理。"""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VERSION='a2_geometry_clearance_v1'
# 固定的出生几何间隙；不修改 PhysX 接触参数，也不把自动 contactOffset 当成实数。
MINIMUM_CLEARANCE_M=.031


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(axis,angle):
    a=np.asarray(axis,dtype=float);a=a/np.linalg.norm(a);x,y,z=a
    skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)


def origin_transform(origin):
    t=np.eye(4)
    if origin is None:return t
    t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
    r,p,y=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
    t[:3,:3]=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r)
    return t


def quaternion_rotation(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def extract_usd_colliders(usd_path):
    """仅在 Isaac 应用加载后调用；按刚体坐标保存实际碰撞几何。"""
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
    stage=Usd.Stage.Open(str(usd_path))
    cache=UsdGeom.XformCache()
    shapes=[]
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() not in ('Mesh','Cube','Sphere','Capsule','Cylinder'):continue
        # 导入器在 collisions Xform 上定义复合碰撞体，几何子节点不单独带 CollisionAPI。
        collider=prim
        while collider and not collider.HasAPI(UsdPhysics.CollisionAPI):collider=collider.GetParent()
        if not collider:continue
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if not body:raise RuntimeError('碰撞体未找到所属刚体: '+str(prim.GetPath()))
        local=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse()
        d={'body':body.GetName(),'prim':str(prim.GetPath()),'type':prim.GetTypeName(),
           'local_transform':np.asarray(local).T.tolist()}
        if d['type']=='Mesh':d['points']=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float).tolist()
        elif d['type']=='Cube':d['size']=float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        elif d['type']=='Sphere':d['radius']=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        elif d['type'] in ('Capsule','Cylinder'):
            shape=UsdGeom.Capsule(prim) if d['type']=='Capsule' else UsdGeom.Cylinder(prim)
            d.update(radius=float(shape.GetRadiusAttr().Get()),height=float(shape.GetHeightAttr().Get()),axis=str(shape.GetAxisAttr().Get()))
        else:raise RuntimeError('尚未支持的实际碰撞形状: '+d['type'])
        shapes.append(d)
    if not shapes:raise RuntimeError('USD 中没有碰撞形状')
    asset_manifest=json.loads((ROOT/'assets/a2/USD_SOURCE_v1.json').read_text())
    return {'version':VERSION,'usd_path':str(Path(usd_path).relative_to(ROOT)),
            'usd_files_sha256':asset_manifest['files'],
            'urdf_sha256':sha256(ROOT/'assets/a2/usd_v1/a2_import.urdf'),
            'shapes':shapes}


class SpawnGeometry:
    def __init__(self,colliders,terrain_path,minimum_clearance=MINIMUM_CLEARANCE_M):
        self.description=colliders
        for name,h in colliders['usd_files_sha256'].items():
            if sha256(ROOT/'assets/a2/usd_v1'/name)!=h:raise RuntimeError('A2 USD 来源已变化: '+name)
        urdf=ROOT/'assets/a2/usd_v1/a2_import.urdf'
        if sha256(urdf)!=colliders['urdf_sha256']:raise RuntimeError('A2 关节坐标来源已变化')
        tree=ET.parse(urdf).getroot()
        self.joints=[]
        for j in tree.findall('joint'):
            self.joints.append((j.get('name'),j.find('parent').get('link'),j.find('child').get('link'),
                origin_transform(j.find('origin')),np.fromstring(j.find('axis').get('xyz'),sep=' ') if j.get('type') in ('revolute','continuous') else None))
        with np.load(terrain_path) as data:
            self.triangles=np.asarray(data['vertices'][data['faces']],dtype=float)
        self.triangle_min=self.triangles.min(axis=1);self.triangle_max=self.triangles.max(axis=1)
        self.contact_clearance=float(minimum_clearance)
        if not np.isfinite(self.contact_clearance) or self.contact_clearance<=0:
            raise ValueError('出生几何间隙必须是有限正数')
        self.shapes=colliders['shapes']

    def forward_kinematics(self,root_pose,q):
        t=np.eye(4);t[:3,:3]=quaternion_rotation(root_pose[3:7]);t[:3,3]=root_pose[:3]
        transforms={'base_link':t};pending=self.joints.copy()
        while pending:
            old=len(pending)
            for j in pending[:]:
                name,parent,child,origin,axis=j
                if parent not in transforms:continue
                local=origin.copy()
                if axis is not None:local[:3,:3]=local[:3,:3]@rotation(axis,q[name])
                transforms[child]=transforms[parent]@local;pending.remove(j)
            if len(pending)==old:raise RuntimeError('A2 关节坐标链无法解析')
        return transforms

    def bounds(self,transforms):
        bounds=[]
        for shape in self.shapes:
            t=transforms[shape['body']]@np.asarray(shape['local_transform'])
            a=t[:3,:3];center=t[:3,3];kind=shape['type']
            if kind in ('Mesh','Cube'):
                if kind=='Mesh':points=np.asarray(shape['points'])
                else:points=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*(shape['size']/2)
                world=points@a.T+center;lo=world.min(axis=0);hi=world.max(axis=0)
            else:
                row_norm=np.linalg.norm(a,axis=1)
                if kind=='Sphere':extent=shape['radius']*row_norm
                else:
                    axis='XYZ'.index(shape['axis'])
                    radial=row_norm if kind=='Capsule' else np.sqrt(np.maximum(0,row_norm**2-a[:,axis]**2))
                    extent=abs(a[:,axis])*shape['height']/2+shape['radius']*radial
                lo=center-extent;hi=center+extent
            bounds.append({'body':shape['body'],'prim':shape['prim'],'min':lo,'max':hi})
        return bounds

    def plan(self,root_pose,q):
        transforms=self.forward_kinematics(root_pose,q);bounds=self.bounds(transforms)
        lo=np.stack([b['min'] for b in bounds]).min(axis=0);hi=np.stack([b['max'] for b in bounds]).max(axis=0)
        # 场景是已冻结的静态平地加障碍。仅有低地面时允许调整 z；遇到墙体直接报告无效配置。
        near=((self.triangle_max[:,:2]>=lo[:2]-self.contact_clearance)&(self.triangle_min[:,:2]<=hi[:2]+self.contact_clearance)).all(axis=1)
        if not near.any():raise RuntimeError('出生碰撞范围内没有地面')
        local_hi=self.triangle_max[near,2]
        if np.any(local_hi>.10):raise RuntimeError('出生碰撞范围内存在高障碍，不可仅调整出生高度')
        floor_max=float(local_hi.max());floor_min=float(self.triangle_min[near,2].min())
        shift=max(0.,floor_max+self.contact_clearance-float(lo[2]))
        return {'version':VERSION,'original_root_height_m':float(root_pose[2]),
                'root_height_m':float(root_pose[2])+shift,'height_shift_m':shift,
                'floor_min_z_m':floor_min,'floor_max_z_m':floor_max,
                'contact_clearance_m':self.contact_clearance,
                'original_lowest_collision_z_m':float(lo[2]),'lowest_collision_z_m':float(lo[2])+shift,
                'lowest_nonfoot_z_m':min(float(b['min'][2])+shift for b in bounds if not b['body'].endswith('_foot')),
                'xy_bounds':[lo[:2].tolist(),hi[:2].tolist()],
                'highest_obstacle_in_spawn_bounds_m':float(local_hi.max())}
