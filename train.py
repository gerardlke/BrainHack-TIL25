import argparse

from omegaconf import OmegaConf
from pathlib import Path

# the 3 components we will use
from trainer import RLTrainer
from stablebaselines_gridworld import build_env

from enum import StrEnum, auto

"""
TODO additional after baseline pipeline is established:
1. Integrate ray

"""

# re-define reward names to pass into our env-builder.
class CustomRewardNames(StrEnum):
    GUARD_WINS = auto()
    GUARD_CAPTURES = auto()
    SCOUT_CAPTURED = auto()
    SCOUT_RECON = auto()
    SCOUT_MISSION = auto()
    WALL_COLLISION = auto()
    AGENT_COLLIDER = auto()
    AGENT_COLLIDEE = auto()
    STATIONARY_PENALTY = auto()
    GUARD_TRUNCATION = auto()
    SCOUT_TRUNCATION = auto()
    GUARD_STEP = auto()
    SCOUT_STEP = auto()
    SCOUT_STEP_EMPTY_TILE = auto()
    LOOKING = auto()
    FORWARD = auto()
    BACKWARD = auto()

STD_REWARDS_DICT = {
    CustomRewardNames.GUARD_CAPTURES: 50,
    CustomRewardNames.SCOUT_CAPTURED: -50,
    CustomRewardNames.SCOUT_RECON: 1,
    CustomRewardNames.SCOUT_MISSION: 5,
    CustomRewardNames.WALL_COLLISION: 0,
    CustomRewardNames.STATIONARY_PENALTY: 0,
    CustomRewardNames.SCOUT_STEP_EMPTY_TILE: 0,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Example script with config file option")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to the config file"
    )
    return parser.parse_args()

config_path = 'standard_config.yaml'

def main():
    args = parse_args()
    config_path = args.config

    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist.")
        exit(1)
    else:
        print(f"Using config file: {config_path}")
        

    config = OmegaConf.load(config_path)
    print('config', config)
    
    # # first, load in training configurations to override others
    # training_config = config.train

    env_config = config.env
    train_env_config = env_config.train
    eval_env_config = env_config.eval

    _, train_env = build_env(
        reward_names=CustomRewardNames,
        rewards_dict=STD_REWARDS_DICT,
        **train_env_config
    )
    _, eval_env = build_env(
        reward_names=CustomRewardNames,
        rewards_dict=STD_REWARDS_DICT,
        **eval_env_config
    )


    policies_config = config.policies
    


if __name__ == '__main__':
    main()
