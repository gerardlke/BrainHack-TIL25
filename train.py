import argparse
import numpy as np
import re
import os
import json

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
        self.experiment_name = base_config.train.experiment_name

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

        training_config = replace_and_report(training_config, config)

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
        num_policies = len(policies_hparams)
        for polid, incoming_policy_config in policies_hparams.items():
            policies_config[polid] = replace_and_report(policies_config[polid], incoming_policy_config, merge=True)
        
        print('config in setup', config)
        REWARDS_DICT = {
            CustomRewardNames.GUARD_CAPTURES: config.get('guard_captures'),
            CustomRewardNames.SCOUT_CAPTURED: config.get('scout_captured'),
            CustomRewardNames.SCOUT_RECON: config.get('scout_recon'),
            CustomRewardNames.SCOUT_MISSION: config.get('scout_mission'),
            CustomRewardNames.WALL_COLLISION: config.get('wall_collision'),
            CustomRewardNames.STATIONARY_PENALTY: config.get('stationary_penalty'),
            CustomRewardNames.SCOUT_STEP_EMPTY_TILE: config.get('scout_step_empty_tile'),
            CustomRewardNames.LOOKING: config.get('looking'),
        }
        # disable collisions for now and enable false eval for now
        # train_env_config['collisions'] = False
        print('train_env_config', train_env_config)
        _, train_env = build_env(
            reward_names=CustomRewardNames,
            rewards_dict=REWARDS_DICT,
            **train_env_config
        )
        # eval_env_config['collisions'] = False
        # eval_env_config['eval'] = False
        _, eval_env = build_env(
            reward_names=CustomRewardNames,
            rewards_dict=STD_REWARDS_DICT,
            **eval_env_config
        )
        print('UNWRAPPED??????', train_env.unwrapped)
        print('UNWRAPPED??????', train_env.unwrapped.compute_mask)
        print('UNWRAPPED??????', train_env.unwrapped.compute_mask('hi'))

        self.agent_roles = list(base_config.agent_roles)
        self.policy_mapping = list(base_config.policy_mapping)
        trial_name = self.trial_name
        trial_code = trial_name[:-6]

        self.eval_log_path = f"{self.root_dir}/ppo_logs/{trial_code}/{trial_name}"
        self.simulator = RLRolloutSimulator(
            train_env=train_env,
            train_env_config=train_env_config,
            policies_config=policies_config,
            policy_mapping=self.policy_mapping,
            tensorboard_log=self.eval_log_path,
            callbacks=None,
            n_steps=training_config.n_steps,
            verbose=1,
        )
        self.total_timesteps = training_config.training_iters * train_env.num_envs
        eval_freq = int(self.total_timesteps / training_config.num_evals / train_env.num_envs)
        print('self.total_timesteps', self.total_timesteps)
        eval_freq = int(max(eval_freq, training_config.n_steps))
        print('eval_freq after max', eval_freq)
        training_config.eval_freq = eval_freq

        checkpoint_callbacks = [
            CheckpointCallback(
                save_freq=eval_freq,
                save_path=f"{self.root_dir}/checkpoints/{trial_code}/{trial_name}/polid_{policy}",
                name_prefix=f"{self.experiment_name}"
            ) for policy in range(num_policies)
        ]
        no_improvement = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=training_config.no_improvement,
            min_evals=int(training_config.num_evals) * 0.25,
            verbose=1
        )
        above_reward = StopTrainingOnRewardThreshold(
            reward_threshold=100.0,
            verbose=1
        )
        eval_callback = CustomEvalCallback(
            evaluate_policy='mp',
            in_bits=True if eval_env_config.binary == 'binary' else False,  # TODO this is really bad code
            log_path=self.eval_log_path,
            agent_roles=self.agent_roles,
            policy_mapping=self.policy_mapping,
            eval_env_config=eval_env_config,
            training_config=training_config,
            eval_env=eval_env,                    
            callback_after_eval=no_improvement,
            callback_on_new_best=above_reward,
            deterministic=False,
        )

        self.callbacks = checkpoint_callbacks
        self.eval_callback = eval_callback
        

    def step(self):  # This is called iteratively.
        self.simulator.learn(
            total_timesteps=self.total_timesteps,
            callbacks=self.callbacks,
            eval_callback=self.eval_callback,
        )

        logging_dict = defaultdict()
        mean_policy_scores = []

        for polid in range(len(set(self.policy_mapping))):
            path = os.path.join(self.eval_log_path, "evaluations", f"policy_id_{polid}.npz")
            thing = np.load(path)
            mean_scores = np.mean(thing['results'], axis=-1)
            max_mean_eval = np.max(mean_scores)
            max_idx = np.argmax(mean_scores)
            mean_policy_scores.append(max_mean_eval)
            best_timestep = thing['timesteps'][max_idx]
            logging_dict.setdefault(f'policy_{polid}_best_result', max_mean_eval)
            logging_dict.setdefault(f'policy_{polid}_best_timestep', best_timestep)

        for polid in range(len(set(self.agent_roles))):
            path = os.path.join(self.eval_log_path, "evaluations", f"role_id_{polid}.npz")
            thing = np.load(path)
            mean_scores = np.mean(thing['results'], axis=-1)
            max_mean_eval = np.max(mean_scores)
            max_idx = np.argmax(mean_scores)
            best_timestep = thing['timesteps'][max_idx]
            logging_dict.setdefault(f'role_{polid}_best_result', max_mean_eval)
            logging_dict.setdefault(f'role_{polid}_best_timestep', best_timestep)

        all_policy_scores = sum(mean_policy_scores)
        logging_dict.setdefault('all_policy_scores', all_policy_scores)
        self.logging_dict = logging_dict
        self.logging_dict.update({'step': 1})

        return logging_dict
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        path = os.path.join(tmp_checkpoint_dir, "state.npz")
        print('path??', path)
        np.savez(
            path,
            **self.logging_dict
        )
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_file):
        # print('tmp_checkpoint_file??', tmp_checkpoint_file)
        state = np.load(tmp_checkpoint_file)
        self.iter = state["step"]
        self.all_policy_scores = state['all_policy_scores']

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config

    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist.")
        exit(1)
    else:
        print(f"Using config file: {config_path}")

    base_config = OmegaConf.load(config_path)
    experiment_name = base_config.train.experiment_name

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
        "stationary_penalty": interpret_search_space(tune_config.stationary_penalty),
        "looking": interpret_search_space(tune_config.looking),
        "wall_collision": interpret_search_space(tune_config.wall_collision),
        "distance_penalty": interpret_search_space(tune_config.distance_penalty),
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
            metric="all_policy_scores",
            mode="max",
            perturbation_interval=10,  # every n trials
            hyperparam_mutations=merged)
    trainable_cls = tune.with_parameters(CustomTrainer, base_config=base_config)
    tuner = tune.Tuner(
        tune.with_resources(trainable_cls, resources={"cpu": 5}),
        tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=1000,
                max_concurrent_trials=1,
        ),
        run_config=tune.RunConfig(
            name='test',
            storage_path=f"{base_config.train.root_dir}/ray_results/{experiment_name}",
            verbose=1,
            stop={"training_iteration": 1},
        )
    )

    results = tuner.fit()
    print('get best results', results.get_best_results())
