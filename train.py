import argparse
import numpy as np
import re

from collections import defaultdict
from ray import tune
from functools import partial
from omegaconf import OmegaConf
from pathlib import Path
from copy import deepcopy
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
from ray.tune.schedulers import PopulationBasedTraining

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

def interpret_search_space(cfg_node):

    typ = cfg_node.get("type")
    if typ == "uniform":
        return tune.uniform(cfg_node.min, cfg_node.max)
    elif typ == "loguniform":
        return tune.loguniform(cfg_node.min, cfg_node.max)
    elif typ == "choice":
        return tune.choice(cfg_node.choices)
    else:
        raise AssertionError('Unknown type provided')
    
def split_dict_by_prefix(source_dict):
    pattern = re.compile(r"^(\d+)/(.+)$")  # this is chatgpt'd. i have no idea of regex works.
    grouped = defaultdict(dict)

    for key, value in source_dict.items():
        match = pattern.match(key)
        if match:
            prefix, suffix = match.groups()
            grouped[int(prefix)][suffix] = value
    return dict(grouped)  # convert defaultdict to regular dict
    
def replace_and_report(base: dict, override: dict, merge=False) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in base and base[key] != value:
            print(f"Overriding key '{key}': {base[key]} -> {value}")
            merged[key] = value
        elif merge:
            print(f"Adding new key '{key}': {value}")
            merged[key] = value

    return merged
    
class CustomTrainer(tune.Trainable):
    """
    Our very own trainer class.
    Pass in the whole config during class initialization, and we'll handle the rest.
    """

    def setup(self, config: dict, base_config):
        """
        Args:
            - config: Hyperparam config passed down to us from the ray gods
            - base_config: base config from the yaml file (see tune.with_parameters) to see
                how it got passed down here
        """
        # setup call. each new iteration, take configs given to us
        # and override copies of our defaults that were defined in our init.

        self.root_dir = base_config.train.root_dir

        self._training_config = base_config.train
        self._env_config = base_config.env
        self._train_env_config = self._env_config.train
        self._eval_env_config = self._env_config.eval
        self._policies_config = base_config.policies
        
        # deepcopy to avoid imploding original configs (for whatever reason)
        training_config = deepcopy(self._training_config)
        train_env_config = deepcopy(self._train_env_config)
        eval_env_config = deepcopy(self._eval_env_config)
        policies_config = deepcopy(self._policies_config)

        print('before training_config', training_config)
        training_config = replace_and_report(training_config, config)
        print('after training_config', training_config)
        train_env_config = replace_and_report(train_env_config, config)
        # we want to be more careful for the eval_env_config, since we keep it controlled at
        # 100 iters. just hardcode replace the frame stack size

        if eval_env_config.frame_stack_size != config['frame_stack_size']:
            print('Overriding hyperparam environment frame_stack_size')
            eval_env_config.frame_stack_size = config['frame_stack_size']

        # merge policy configurations; override old with new per policy basis.
        policies_hparams = split_dict_by_prefix(config)
        assert len(policies_hparams) == len(policies_config), 'Assertion failed, mismatching number of policies as specified by tune configuration,' \
            'and number of policies under the policies section. Failing gracefully.'
        print('hparams', policies_hparams)
        print('base', policies_config)
        for polid, incoming_policy_config in policies_hparams.items():
            policies_config[polid] = replace_and_report(policies_config[polid], incoming_policy_config, merge=True)
        print('after replacement', policies_config)

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

        assert len(train_env.observation_space.shape) == 1, 'Assertion failed, your env observation space is not flattened.'
        policies_config = base_config.policies

        trial_code = 'test'
        trial_name = self.trial_name
        EXPERIMENT_NAME = 'test'

        eval_log_path = f"{self.root_dir}/ppo_logs/{trial_code}/{trial_name}"
        self.simulator = RLRolloutSimulator(
            train_env=train_env,
            train_env_config=train_env_config,
            policies_config=policies_config,
            policy_mapping=policy_mapping,
            tensorboard_log=eval_log_path,
            callbacks=None,
            n_steps=training_config.n_steps,
            verbose=1,
        )
        self.total_timesteps = training_config.training_iters * training_config.n_steps * train_env.num_envs
        eval_freq = self.total_timesteps / training_config.num_evals / train_env.num_envs
        eval_freq = int(max(eval_freq, training_config.n_steps))
        training_config.eval_freq = eval_freq

        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=f"{self.root_dir}/checkpoints/{trial_code}/{trial_name}",
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
            evaluate_policy='mp',
            in_bits=True if eval_env_config.env_type == 'binary' else False,  # TODO this is really bad code
            agent_roles=agent_roles,
            policy_mapping=policy_mapping,
            eval_env_config=eval_env_config,
            training_config=training_config,
            eval_env=eval_env,                    
            callback_after_eval=no_improvement,
            callback_on_new_best=above_reward,
            deterministic=True,
        )
        # progress_bar = ProgressBarCallback()

        # Combine callbacks
        self.callbacks = CallbackList([
            eval_callback,
            checkpoint_callback,
            # progress_bar,
        ])
        

    def step(self):  # This is called iteratively.
        self.simulator.learn(
            total_timesteps=self.total_timesteps,
            callbacks=self.callbacks,
        )

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config

    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist.")
        exit(1)
    else:
        print(f"Using config file: {config_path}")

    base_config = OmegaConf.load(config_path)

    tune_config = base_config.tune

    # first, create the hyperparam search space and population based training.
    # this will optimize the hyperparams within each run.
    # there are some that can be opted to be policy specific, and some that are environment specific.
    # things that control environment run length or steps before policies train are examples of these.
    policy_independent_hparams = {
        "n_steps": interpret_search_space(tune_config.n_steps),
        "frame_stack_size": interpret_search_space(tune_config.frame_stack_size),
        "novice": interpret_search_space(tune_config.novice),
        "num_iters": interpret_search_space(tune_config.num_iters),
        "guard_captures": interpret_search_space(tune_config.guard_captures),
        "scout_captured": interpret_search_space(tune_config.scout_captured),
        "scout_recon": interpret_search_space(tune_config.scout_recon),
        "scout_mission": interpret_search_space(tune_config.scout_mission),
        "scout_step_empty_tile": interpret_search_space(tune_config.scout_step_empty_tile),
    }
    # the following may be policy specific. hence, scale to n_policies:
    policy_dependent_hparams = [{
        f"{polid}/{k}": interpret_search_space(v) for k, v in policy_config.items()
    } for polid, policy_config in tune_config.policies.items()]

    # merge everything
    merged = {}
    [merged.update(d) for d in policy_dependent_hparams]
    merged.update(policy_independent_hparams)


    pbt = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="all_scores",
            mode="max",
            perturbation_interval=5,  # every n trials
            hyperparam_mutations=merged)
    trainable_cls = tune.with_parameters(CustomTrainer, base_config=base_config)
    tuner = tune.Tuner(
        trainable_cls,
        tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=1,
                reuse_actors=True,
                max_concurrent_trials=1,
        ),
        run_config=tune.RunConfig(
            name='test',
            storage_path=f"{base_config.train.root_dir}/ray_results",
            verbose=1
        )
    )

    results = tuner.fit()
    print('best results', results)
