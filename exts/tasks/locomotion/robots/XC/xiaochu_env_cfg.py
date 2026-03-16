import math

from isaaclab.utils import configclass

from exts.assets.config.xiaochu_cfg import XIAOCHU_CFG
from exts.tasks.locomotion.robots.XC.base_env_cfg import XCEnvCfg
from exts.tasks.locomotion.robots.XC.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)

from isaaclab.sensors import RayCasterCfg, patterns
from exts.tasks.locomotion import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg


######################
# xiaochu Base Environment
######################


@configclass
class XCBaseEnvCfg(XCEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = XIAOCHU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            'right_hip_pitch': -0.6553,
            'right_knee_pitch': -1.2313,
            'left_hip_pitch': -0.6553, 
            'left_knee_pitch': -1.2313, 
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        # self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        
        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class XCBaseEnvCfg_PLAY(XCBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


############################
# XIAOCHU Blind Flat Environment
############################


@configclass
class XCBlindFlatEnvCfg(XCBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class XCBlindFlatEnvCfg_PLAY(XCBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


#############################
# XIAOCHU Blind Rough Environment
#############################


@configclass
class XCBlindRoughEnvCfg(XCBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class XCBlindRoughEnvCfg_PLAY(XCBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None
        
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG
        

##############################
# XC Blind Stairs Environment
##############################

@configclass
class XCBlindStairEnvCfg(XCBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class XCBlindStairEnvCfg_PLAY(XCBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))
        
        
#############################
# XIAOCHU Flat Environment
#############################

@configclass
class XCFlatEnvCfg(XCBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.curriculum.terrain_levels = None

@configclass
class XCFlatEnvCfg_PLAY(XCBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.curriculum.terrain_levels = None
        
        
#############################
# Xiaochu Rough Environment
#############################

@configclass
class XCRoughEnvCfg(XCBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class XCRoughEnvCfg_PLAY(XCBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG



        
        
##############################
# Xiaochu Blind Stairs Environment
##############################


@configclass
class XCStairEnvCfg(XCBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
                    clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class XCStairEnvCfg_PLAY(XCBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))
        


