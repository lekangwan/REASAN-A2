"""从已验证A2名义参数创建Lab关节资产；不包含导航或训练逻辑。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def asset_path():
    return ROOT/json.loads((ROOT/'assets/a2/ACTIVE_ASSET.json').read_text())['usd']


def robot_contract():
    return json.loads((ROOT/"assets/a2/ROBOT_CONTRACT.json").read_text())


def robot_cfg(usd_path):
    import isaaclab.sim as sim
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    c = robot_contract()
    efforts = {name: float(v["effort"]) for name, v in c["joint_limits"].items()}
    velocities = {name: float(v["velocity"]) for name, v in c["joint_limits"].items()}
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/A2",
        spawn=sim.UsdFileCfg(usd_path=str(usd_path), activate_contact_sensors=True,
            rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0., angular_damping=0.,
                max_linear_velocity=1000., max_angular_velocity=1000., max_depenetration_velocity=1.,
                enable_gyroscopic_forces=False),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=.01, rest_offset=0.),
            articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=4, solver_velocity_iteration_count=0)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., c["root_height"]),
            joint_pos=c["default_joint_angles"], joint_vel={".*": 0.}),
        # 显式PD在每个物理步外部计算；此处驱动增益为0，避免重复施加PD。
        actuators={"effort": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0., damping=0.,
            effort_limit_sim=efforts, velocity_limit_sim=velocities, armature=c["armature"], friction=0.)},
        soft_joint_pos_limit_factor=1.)
