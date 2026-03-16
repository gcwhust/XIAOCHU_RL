python -m pip install -e exts/bipedal_locomotion/ --index-url https://pypi.org/simple   --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

--index-url https://pypi.org/simple --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
 PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple python -m pip install -e exts/bipedal_locomotion/ -vvv


isaacsim位置
# Isaac Sim root directory
export ISAACSIM_PATH="/home/user/IsaacSim/_build/linux-x86_64/release"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
# add IsaacSim target-deps python bin to PATH
export PATH="/home/user/IsaacSim/_build/target-deps/python/bin:$PATH"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python scripts/train.py --task=Isaac-XC-Blind-Flat-v0 --headless --video
python scripts/play.py --task=Isaac-XC-Blind-Flat-Play-v0 --video --video_length 500 --headless

tensorboard --logdir logs

IsaacLab 强化学习环境完整配置

环境基本信息

• 环境名称: Isaac-XC-Blind-Flat-v0

• 机器人型号: Xiaochu 双足机器人

• 设备: CUDA

• 种子: 42

• 实验名称: xc_flat

训练参数配置

RSL-RL PPO 配置

max_iterations = 10000
num_steps_per_env = 24
seed = 42
device = "cuda:0"
empirical_normalization = False
save_interval = 500
resume = False


网络架构

policy_network = [512, 256, 128]  # 隐藏层维度
critic_network = [512, 256, 128]  # 隐藏层维度
activation = "elu"
init_noise_std = 1.0


PPO 算法参数

value_loss_coef = 1.0
use_clipped_value_loss = True
clip_param = 0.2
entropy_coef = 0.01
num_learning_epochs = 5
num_mini_batches = 4
learning_rate = 0.001
schedule = "adaptive"
gamma = 0.99
lam = 0.95
desired_kl = 0.01
max_grad_norm = 1.0


编码器配置

encoder_output_dim = 3
encoder_hidden_dims = [256, 128]
encoder_activation = "elu"
output_detach = True
orthogonal_init = False


环境参数配置

基础设置

episode_length_s = 20.0
decimation = 4
dt = 0.005
render_interval = 8
num_envs = 4096
env_spacing = 2.5


物理仿真参数

gravity = (0.0, 0.0, -9.81)
enable_scene_query_support = False
use_fabric = True

# PhysX 配置
solver_type = 1
min_position_iteration_count = 1
max_position_iteration_count = 255
min_velocity_iteration_count = 0
max_velocity_iteration_count = 255
enable_ccd = False
enable_stabilization = True

# 物理材料
static_friction = 0.5
dynamic_friction = 0.5
restitution = 0.0


渲染配置

enable_translucency = False
enable_reflections = False
enable_global_illumination = False
antialiasing_mode = "DLSS"
samples_per_pixel = 1
enable_shadows = True
enable_ambient_occlusion = False


机器人配置

初始状态

initial_position = (0.0, 0.0, 0.966)
initial_rotation = (1.0, 0.0, 0.0, 0.0)
initial_linear_velocity = (0.0, 0.0, 0.0)
initial_angular_velocity = (0.0, 0.0, 0.0)

# 关节初始位置
joint_pos = {
    'right_hip_pitch': 0.0,
    'right_knee_pitch': 0.0, 
    'left_hip_pitch': 0.0,
    'left_knee_pitch': 0.0
}
joint_vel = {'.*': 0.0}  # 所有关节速度为0


执行器配置

腿部执行器（PD控制）

joint_names = ['right_hip_pitch', 'right_knee_pitch', 'left_hip_pitch', 'left_knee_pitch']
effort_limit = 80.0
velocity_limit = 15.0
stiffness = 40.0
damping = 2.5


轮子执行器（PD控制）

joint_names = ['left_wheel_joint', 'right_wheel_joint']
effort_limit = 80.0
velocity_limit = 15.0
stiffness = 0.0  # 零刚度
damping = 0.8


观测空间配置

策略网络观测（带噪声）

policy_observations = {
    'base_ang_vel': {'noise_std': 0.05, 'clip': (-100, 100), 'scale': 0.25},
    'proj_gravity': {'noise_std': 0.025, 'clip': (-100, 100), 'scale': 1.0},
    'joint_pos_rel_exclude_wheel': {'noise_std': 0.01, 'clip': None, 'scale': None},
    'joint_vel_rel': {'noise_std': 0.01, 'clip': (-100, 100), 'scale': 0.05},
    'last_action': {'noise_std': 0.01, 'clip': (-100, 100), 'scale': 1.0}
}


评价网络观测（无噪声）

critic_observations = {
    'base_lin_vel': {'clip': (-100, 100), 'scale': 1.0},
    'base_ang_vel': {'clip': (-100, 100), 'scale': 1.0},
    'proj_gravity': {'clip': (-100, 100), 'scale': 1.0},
    'joint_pos_rel': {'clip': (-100, 100), 'scale': 1.0},
    'joint_vel': {'clip': (-100, 100), 'scale': 1.0},
    'last_action': {'clip': (-100, 100), 'scale': 1.0},
    # 物理属性观测
    'robot_joint_torque': {},
    'robot_joint_acc': {},
    'feet_lin_vel': {'body_names': '*wheel'},
    'robot_mass': {},
    'robot_inertia': {},
    'robot_joint_pos': {},
    'robot_joint_stiffness': {},
    'robot_joint_damping': {},
    'robot_pos': {},
    'robot_vel': {},
    'robot_material_properties': {},
    'feet_contact_force': {'body_names': 'wheel_.*'}
}


历史观测

history_length = 10
history_observations = ['base_ang_vel', 'proj_gravity', 'joint_pos_rel_exclude_wheel', 'joint_vel_rel', 'last_action']


命令观测

commands_obs = {
    'velocity_commands': {'command_name': 'base_velocity'}
}


动作空间配置

关节位置动作

joint_pos_action = {
    'joints': ['right_hip_pitch', 'right_knee_pitch', 'left_hip_pitch', 'left_knee_pitch'],
    'scale': 0.25,
    'offset': 0.0
}


关节速度动作

joint_vel_action = {
    'joints': ['right_wheel_joint', 'left_wheel_joint'],
    'scale': 1.0,
    'offset': 0.0
}


奖励函数配置

正向奖励

positive_rewards = {
    'keep_balance': {'weight': 1.0},           # 保持平衡
    'rew_lin_vel_xy': {'weight': 3.0},        # 跟踪线速度
    'rew_ang_vel_z': {'weight': 1.0},         # 跟踪角速度  
    'rew_leg_symmetry': {'weight': 0.5}       # 腿部对称性
}


负向奖励（惩罚）

negative_rewards = {
    'stand_still': {'weight': -5.0},           # 静止惩罚
    'pen_lin_vel_z': {'weight': -0.3},         # Z轴速度惩罚
    'pen_ang_vel_xy': {'weight': -0.3},       # XY角速度惩罚
    'pen_joint_torque': {'weight': -0.00016}, # 关节力矩惩罚
    'pen_joint_accel': {'weight': -1.5e-07},  # 关节加速度惩罚
    'pen_action_rate': {'weight': -0.3},      # 动作变化率惩罚
    'pen_non_wheel_pos_limits': {'weight': -2.0},  # 非轮子关节位置限制
    'undesired_contacts': {'weight': -0.25},  # 非期望接触惩罚
    'pen_action_smoothness': {'weight': -0.03}, # 动作平滑性惩罚
    'pen_flat_orientation_l2': {'weight': -12.0}, # 平坦朝向惩罚
    'pen_feet_distance': {'weight': -100},    # 足部距离惩罚
    'pen_base_height': {'weight': -30.0},     # 基础高度惩罚
    'pen_joint_power_l1': {'weight': -2e-05}, # 关节功率惩罚
    'pen_joint_vel_wheel_l2': {'weight': -0.005}, # 轮子关节速度惩罚
    'pen_vel_non_wheel_l2': {'weight': -0.03} # 非轮子关节速度惩罚
}


终止条件配置

termination_conditions = {
    'time_out': {'time_out': True},  # 超时终止（20秒）
    'base_contact': {                # 基础接触地面终止
        'sensor_cfg': {'body_names': 'base_Link'},
        'threshold': 1.0,
        'time_out': False
    }
}


域随机化配置

启动时随机化

startup_randomization = {
    'add_base_mass': {  # 基础质量添加
        'body_names': 'base_Link',
        'mass_distribution_params': (-1.0, 2.0),
        'operation': 'add'
    },
    'add_link_mass': {  # 链接质量缩放
        'body_names': '.*_[LR]_Link', 
        'mass_distribution_params': (0.8, 1.2),
        'operation': 'scale'
    },
    'radomize_rigid_body_mass_inertia': {  # 质量惯性随机化
        'mass_inertia_distribution_params': (0.8, 1.2),
        'operation': 'scale'
    },
    'robot_physics_material': {  # 物理材料随机化
        'static_friction_range': (0.4, 1.2),
        'dynamic_friction_range': (0.7, 0.9), 
        'restitution_range': (0.0, 1.0),
        'num_buckets': 48
    },
    'robot_joint_stiffness_and_damping': {  # 关节刚度和阻尼
        'stiffness_distribution_params': (32, 48),
        'damping_distribution_params': (2.0, 3.0),
        'operation': 'abs',
        'distribution': 'uniform'
    },
    'robot_center_of_mass': {  # 质心随机化
        'com_distribution_params': ((-0.075, 0.075), (-0.075, 0.075), (-0.075, 0.075)),
        'operation': 'add',
        'distribution': 'uniform'
    }
}


重置时随机化

reset_randomization = {
    'reset_robot_base': {  # 重置机器人基础状态
        'pose_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'yaw': (-3.14, 3.14)},
        'velocity_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'z': (-0.5, 0.5), 
                          'roll': (-0.5, 0.5), 'pitch': (-0.5, 0.5), 'yaw': (-0.5, 0.5)}
    },
    'reset_robot_joints': {  # 重置关节状态
        'position_range': (-0.2, 0.2),
        'velocity_range': (-0.5, 0.5)
    },
    'randomize_actuator_gains': {  # 执行器增益随机化
        'stiffness_distribution_params': (0.5, 2.0),
        'damping_distribution_params': (0.5, 2.0),
        'operation': 'scale',
        'distribution': 'log_uniform'
    }
}


间隔随机化

interval_randomization = {
    'push_robot': {  # 随机推动机器人
        'body_names': 'base_Link',
        'force_range': {'x': (-500.0, 500.0), 'y': (-500.0, 500.0), 'z': (-0.0, 0.0)},
        'torque_range': {'x': (-50.0, 50.0), 'y': (-50.0, 50.0), 'z': (-0.0, 0.0)},
        'probability': 0.002
    }
}


命令生成配置

velocity_commands = {
    'resampling_time_range': (3.0, 15.0),
    'heading_command': True,
    'heading_control_stiffness': 1.0,
    'rel_standing_envs': 0.02,
    'rel_heading_envs': 1.0,
    'ranges': {
        'lin_vel_x': (-0.7, 0.7),
        'lin_vel_y': (-0.5, 0.5), 
        'ang_vel_z': (-3.14159, 3.14159),
        'heading': (-3.14159, 3.14159)
    }
}


传感器配置

接触传感器

contact_sensor = {
    'update_period': 0.005,
    'history_length': 4,
    'track_air_time': True,
    'force_threshold': 1.0
}


空值/未设置配置

empty_configs = {
    'height_scanner': None,
    'heights_observation': None,
    'terrain_levels_curriculum': None,
    'run_name': '',
    'encoder_num_input_dim': 'MISSING'
}


这个配置定义了一个完整的双足机器人强化学习训练环境，包含了观测、动作、奖励、随机化等所有必要组件。
