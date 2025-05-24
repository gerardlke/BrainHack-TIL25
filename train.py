import argparse
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path

# the 3 components we will use
from trainer import RLRolloutSimulator
from stablebaselines_gridworld import build_env
from custom_eval_callback import CustomEvalCallback

from enum import StrEnum, auto
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
    CallbackList,
    # ProgressBarCallback,
    StopTrainingOnNoModelImprovement,
)

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


def main():
    """
    Main function.
    The job of this function is to hold state / information to be regarded as
    component independent; that is, state that is generated and the held unchanged
    throughout the training run.

    This includes things like hyperparamter configurations, policy and role mappings, etc.
    """
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

    agent_roles = train_gridworld.role_mapping 
    policy_mapping = train_gridworld.policy_mapping

    assert len(train_env.observation_space.shape) == 1
    policies_config = config.policies
    
    root_dir = training_config.root_dir
    n_steps = training_config.n_steps
    training_iters = 10000
    trial_code = 'test'
    trial_name = 'delete_me'
    EXPERIMENT_NAME = 'test'

    eval_log_path = f"{root_dir}/ppo_logs/{trial_code}/{trial_name}"
    simulator = RLRolloutSimulator(
        train_env=train_env,
        train_env_config=train_env_config,
        policies_config=policies_config,
        policy_mapping=policy_mapping,
        tensorboard_log=eval_log_path,
        callbacks=None,
        n_steps=training_config.n_steps,
        verbose=1,
    )

    num_evals = 500
    eval_freq = int(max(training_iters / num_evals / 4, 1)) * n_steps  # cuz parallel env

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=f"{root_dir}/checkpoints/{trial_code}/{trial_name}",
        name_prefix=f"{EXPERIMENT_NAME}"
        )
    no_improvement = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=50,
        min_evals=50,
        verbose=1
    )
    above_reward = StopTrainingOnRewardThreshold(
        reward_threshold=100.0,
        verbose=1
    )
    eval_callback = CustomEvalCallback(
        in_bits=True if eval_env_type == 'binary' else False,  # TODO this is really bad code
        agent_roles=agent_roles,
        policy_mapping=policy_mapping,
        eval_env_config=eval_env_config,
        eval_env=eval_env,                    
        callback_after_eval=no_improvement,
        callback_on_new_best=above_reward,
        deterministic=True,
    )
    # progress_bar = ProgressBarCallback()

    # Combine callbacks
    callback = CallbackList([
        eval_callback,
        checkpoint_callback,
        # progress_bar,
    ])

    simulator.learn(

    )

    


if __name__ == '__main__':
    main()
