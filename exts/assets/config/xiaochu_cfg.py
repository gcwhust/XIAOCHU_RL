import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
usd_path = os.path.join(parent_dir, "usd/xiaochu/jc04.usd")

XIAOCHU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,#0.0
            angular_damping=0.0,#0.0
            max_linear_velocity=1000.0,#1000.0
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,#True
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),  # 设定高度为 0.63 m
        joint_pos={
            # Leg joints - balanced pose from trajectory data
            # From balance_trajectory_0_75deg.csv at 75 degrees delta_theta
            "right_hip_pitch": -0.6556,  # -41.574° in radians
            "right_knee_pitch": -1.2315,  # -79.582° in radians
            "left_hip_pitch": -0.6556,
            "left_knee_pitch": -1.2315,
            
            # Driving wheels - zero position
            "right_wheel_joint": 0.0,
            "left_wheel_joint": 0.0,
            
            # Arms - neutral position
            "right_shoulder_pitch": 0.0,
            "right_shoulder_roll": 0.0,
            "right_elbow_pitch": 1.5707963268,
            "left_shoulder_pitch": 0.0,
            "left_shoulder_roll": 0.0,
            "left_elbow_pitch": 1.5707963268,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                'right_hip_pitch',
                'right_knee_pitch',
                'left_hip_pitch', 
                'left_knee_pitch', 
            ],
            effort_limit_sim=200.0,#limx:effort_limit 100
            velocity_limit_sim=20.0,#limx:velocity_limit
            stiffness=1000.0,#40.0
            damping=50,#4
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_wheel_joint",
                "right_wheel_joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=10.0,
            friction=0.0,
        ), 
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_pitch",
                "right_shoulder_roll",
                "right_elbow_pitch",
                "left_shoulder_pitch",
                "left_shoulder_roll",
                "left_elbow_pitch",
            ],
            effort_limit_sim=200.0,
            velocity_limit_sim=20.0,
            stiffness=1000.0,
            damping=50.0,
            friction=0.0,
        ),
        # TODO: change to delayed implicit actuator
    },
)
