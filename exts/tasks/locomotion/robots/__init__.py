import gymnasium as gym

from exts.tasks.locomotion.agents.rsl_rl_ppo_cfg import XCFlatPPORunnerCfg

from .XC import base_env_cfg, xiaochu_env_cfg

##
# Create PPO runners for RSL-RL
##
xc_blind_flat_runner_cfg = XCFlatPPORunnerCfg()



##
# Register Gym environments
##
gym.register(
    id="Isaac-XC-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": xiaochu_env_cfg.XCBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": xc_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-XC-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": xiaochu_env_cfg.XCBlindFlatEnvCfg_PLAY,#PLAY version
        "rsl_rl_cfg_entry_point": xc_blind_flat_runner_cfg,
    },
)
