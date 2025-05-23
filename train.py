import argparse
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path

# the 3 components we will use
from trainer import RLRolloutSimulator
from stablebaselines_gridworld import build_env

from enum import StrEnum, auto

"""
TODO additional after baseline pipeline is established:
1. Integrate ray
ok bet
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

def generate_policy_agent_indexes(n_envs, policy_mapping):
    """
    Input: A list of policy mapping each agent's index to a policy.
    E.g default [1, 0, 0, 0] maps the 0th index agent to policy with id 1,
    and 1, 2, 3 index to policy of id 0.
    From this, create a nested list of n policies long, each list has indexes
    of the vectorized environments index.
    e.g n_envs = 2, policy mapping as above.
    Output will be:
    [
        [1, 2, 3, 5, 6, 7], [0, 4]
    ].
    """
    n_policy_mapping = np.array(policy_mapping * n_envs)
    policy_indexes = [
        np.where(n_policy_mapping == polid)[0] for polid in np.unique(n_policy_mapping)
    ]

    return policy_indexes


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
    
    # first, load in training configurations to override others
    training_config = config.train

    env_config = config.env
    train_env_config = env_config.train
    eval_env_config = env_config.eval

    train_gridworld, train_env = build_env(
        reward_names=CustomRewardNames,
        rewards_dict=STD_REWARDS_DICT,
        **train_env_config
    )
    _, eval_env = build_env(
        reward_names=CustomRewardNames,
        rewards_dict=STD_REWARDS_DICT,
        **eval_env_config
    )

    policy_agent_indexes = generate_policy_agent_indexes(
        train_env_config.num_vec_envs, train_gridworld.policy_mapping
    )

    assert len(train_env.observation_space.shape) == 1
    policies_config = config.policies
    trial_code = 'test'
    trial_name = 'delete_me'
    eval_log_path = f"{training_config.root_dir}/ppo_logs/{trial_code}/{trial_name}"
    RLRolloutSimulator(
        train_env=train_env,
        policies_config=policies_config,
        policy_agent_indexes=policy_agent_indexes,
        tensorboard_log=eval_log_path,
        callbacks=None,
        n_steps=training_config.n_steps,
        verbose=1,
    )
    


if __name__ == '__main__':
    main()
