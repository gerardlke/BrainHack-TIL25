import argparse
import numpy as np
import re
import os
import shutil
import json
import hashlib
import uuid

from collections import defaultdict
from ray import tune
from functools import partial
from omegaconf import OmegaConf
from pathlib import Path
from copy import deepcopy
# the 3 components we will use
from selfplay_trainer import RLRolloutSimulator
from self_play_env import build_env
from custom_callbacks import CustomEvalCallback, CustomCheckpointCallback

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

from rl.db.db import RL_DB

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

def generate_8char_hash():
    unique_str = str(uuid.uuid4())  # Generate a unique string
    hash_object = hashlib.sha256(unique_str.encode())  # Hash it
    short_hash = hash_object.hexdigest()[:8]  # Take the first 8 characters
    return short_hash

class SelfPlayOrchestrator:
    """
    A wrapper around ray tune for training. This orchestrates agent selection and policy loading,
    continuing to loop until some stopping condition (TODO decide this). 

    While stopping condition has not passed:
    1. Select an agent, load in its corresponding policy.
    2. Initialize ray tune and trainable. Tune allows for us to try how different hyperparameters fare in a trial.
    3. Obtain the best result, checkpoint it into the DB, and loop back to 1.

    Each orchestrator will generate its own unique 8-size hash, and that will be the working directory
    of all checkpoints, ray results, and tensorboard logs.
    """

    def __init__(self, config):
        """
        Initialize orchestration.
        Most important configurations are what agents we are training, and what agents are part of the environment.
        """
        self.config = config
        assert len(self.config.agent_roles) == len(self.config.policy_mapping), 'Assertion failed. agent_roles in the config '\
            'must have the same length as policy_mapping in the config.'
        self.policy_mapping = config.policy_mapping
        self.hash = generate_8char_hash()
        self.config.train.root_dir = os.path.join(self.config.train.root_dir, f'Orchestrator_{self.hash}')
        if not os.path.exists(self.config.db_name):
            db_path = os.path.join(self.config.train.root_dir, self.config.db_name)
        else:
            db_path = self.config.db_name
        # if path to db is not in our root directory, copy the db file into it.
        file_path = Path(db_path).resolve()
        parent_dir = Path(self.config.train.root_dir).resolve()
        parent_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_path.relative_to(parent_dir)
        except ValueError:
            # do a copy
            db_path = parent_dir / file_path.name
            shutil.copy2(file_path, db_path)

        self.config.db_path = db_path
        self.num_loops = self.config.train.num_loops * len(self.config.agent_roles)
        self.agent_roles = self.config.agent_roles

    def commence(self):
        """
        Commences training.
        For all collections of selectable policies, build the trainable and call ray tune to optimize it.
        """
        # we loop over agent_roles to know how many agents there are. no stopping condition for now.
        for _ in range(self.num_loops):
            for polid in list(set(self.policy_mapping)):
                tmp_config = deepcopy(self.config)  # deepcopy to avoid mutating underlying config, for whatever reason.

                # boolean mask of npcs: 0 represents the selected agent, 1 represents an agent controlled by the environment.
                policies_controlled_here = [polid] # integer indexes of policy controlled here ( for now, just one. )
                num_vec_envs = self.policy_mapping.count(polid)
                tmp_config.env.train.num_vec_envs = num_vec_envs
                tmp_config.env.eval.num_vec_envs = num_vec_envs

                # for the policies not controlled by the orchestrator-simulator, mark None.
                self.simulator_policies = deepcopy(self.policy_mapping)
                for idx, polid in enumerate(self.policy_mapping):
                    if polid not in policies_controlled_here:
                        self.simulator_policies[idx] = None

                print('self.simulator_policies Nonified?', self.simulator_policies)
                tmp_config.simulator_policies = self.simulator_policies
                trainable = create_trainable()

                experiment_name = tmp_config.train.experiment_name

                tune_config = tmp_config.tune

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
                # extract only the hparams relevant to the policy(ies) we are training.
                tune_config.policies = {
                    polid: v for polid, v in tune_config.policies.items() if polid in policies_controlled_here
                }
                policy_dependent_hparams = [{
                    f"{polid}/{k}": interpret_search_space(v) for k, v in policy_config.items()
                } for polid, policy_config in tune_config.policies.items()]

                # prune those configurations
                tmp_config.policies = {
                    polid: v for polid, v in tmp_config.policies.items() if polid in policies_controlled_here
                }

                print('policies_controlled_here', policies_controlled_here)
                print('tmp_config', tmp_config)

                # merge everything
                merged = {}
                [merged.update(d) for d in policy_dependent_hparams]
                merged.update(policy_independent_hparams)

                pbt = PopulationBasedTraining(
                        time_attr="training_iteration",
                        metric="all_policy_scores",
                        mode="max",
                        perturbation_interval=4,  # every n trials
                        hyperparam_mutations=merged)
                
                trainable_cls = tune.with_parameters(trainable, base_config=tmp_config) # this is where the edited config
                # gets passed to the trainable. environment is initialized within trainable.
            
                tuner = tune.Tuner(
                    tune.with_resources(trainable_cls, resources={"cpu": 5}),
                    tune_config=tune.TuneConfig(
                            scheduler=pbt,
                            num_samples=8,
                            max_concurrent_trials=4,
                    ),
                    run_config=tune.RunConfig(
                        name='test',
                        storage_path=f"{tmp_config.train.root_dir}/ray_results/{experiment_name}",
                        verbose=1,
                        stop={"training_iteration": 1},
                    )
                )

                results = tuner.fit()
                # print('get best results', results.get_best_result())


def create_trainable():
    class CustomTrainer(tune.Trainable):
        """
        Our very own policy trainer class.
        Pass in the whole ray_config during class initialization, and we'll handle the rest.
        """

        def setup(self, ray_config: dict, base_config):
            """
            Sets up training. Merges base and ray config, replacing default hyperparams and specifications from base
            with whatever ray generates.

            
                            npcs = [1 if agent != agent_role else 0 for agent in self.agent_roles]
            Args:
                - ray_config: Hyperparam ray_config passed down to us from the ray gods
                - base_config: base ray_config from the yaml file (see tune.with_parameters to see
                    how it got passed down here).
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

            training_config = replace_and_report(training_config, ray_config)

            train_env_config = replace_and_report(train_env_config, ray_config)
            # we want to be more careful for the eval_env_config, since we keep it controlled at
            # 100 iters. just hardcode replace the frame stack size

            if eval_env_config.frame_stack_size != ray_config['frame_stack_size']:
                print('Overriding hyperparam environment frame_stack_size')
                eval_env_config.frame_stack_size = ray_config['frame_stack_size']

            # merge policy configurations; override old with new per policy basis.
            policies_hparams = split_dict_by_prefix(ray_config)
            assert len(policies_hparams) == len(policies_config), 'Assertion failed, mismatching number of policies as specified by tune configuration,' \
                'and number of policies under the policies section. Failing gracefully.'

            for polid, incoming_policy_config in policies_hparams.items():
                policies_config[polid] = replace_and_report(policies_config[polid], incoming_policy_config, merge=True)
            self.policies_config = policies_config

            print('ray_config in setup', ray_config)
            REWARDS_DICT = {
                CustomRewardNames.GUARD_CAPTURES: ray_config.get('guard_captures'),
                CustomRewardNames.SCOUT_CAPTURED: ray_config.get('scout_captured'),
                CustomRewardNames.SCOUT_RECON: ray_config.get('scout_recon'),
                CustomRewardNames.SCOUT_MISSION: ray_config.get('scout_mission'),
                CustomRewardNames.WALL_COLLISION: ray_config.get('wall_collision'),
                CustomRewardNames.STATIONARY_PENALTY: ray_config.get('stationary_penalty'),
                CustomRewardNames.SCOUT_STEP_EMPTY_TILE: ray_config.get('scout_step_empty_tile'),
                CustomRewardNames.LOOKING: ray_config.get('looking'),
            }
            
            # initialize the environment configurations, but throw in some other things from the main
            # configuration part that are required too.
            print('train_env_config', train_env_config)
            print('eval_env_config', eval_env_config)
            _, train_env = build_env(
                reward_names=CustomRewardNames,
                rewards_dict=REWARDS_DICT,
                policy_mapping=base_config.policy_mapping,
                agent_roles=base_config.agent_roles,
                self_play=base_config.self_play,
                db_path=base_config.db_path,
                env_config=train_env_config,
                num_iters=train_env_config.num_iters,
            )
            print('done building train env.')
            _, eval_env = build_env(
                reward_names=CustomRewardNames,
                rewards_dict=STD_REWARDS_DICT,
                policy_mapping=base_config.policy_mapping,
                agent_roles=base_config.agent_roles,
                self_play=base_config.self_play,
                db_path=base_config.db_path,
                env_config=eval_env_config,
                num_iters=eval_env_config.num_iters,
            )
            print('done building eval env.')
            
            self.agent_roles = list(base_config.agent_roles)
            self.simulator_policies = list(base_config.simulator_policies)
            trial_name = self.trial_name
            trial_code = trial_name[:-6]

            self.eval_log_path = f"{self.root_dir}/ppo_logs/{trial_code}/{trial_name}"
            self.simulator = RLRolloutSimulator(
                selfplay=True,
                db_path=base_config.db_path,
                train_env=train_env,
                train_env_config=train_env_config,
                policies_config=self.policies_config,
                policy_mapping=self.simulator_policies,
                tensorboard_log=self.eval_log_path,
                callbacks=None,
                n_steps=training_config.n_steps,
                use_action_masking=training_config.use_action_masking,
                verbose=0,
            )
            self.total_timesteps = training_config.training_iters * train_env.num_envs
            eval_freq = int(self.total_timesteps / training_config.num_evals / train_env.num_envs)
            eval_freq = int(max(eval_freq, training_config.n_steps))
            training_config.eval_freq = eval_freq

            checkpoint_callbacks = [
                CustomCheckpointCallback(
                    save_freq=eval_freq,
                    polid=polid,
                    db_path=base_config.db_path,
                    save_path=f"{self.root_dir}/checkpoints/{trial_code}/{trial_name}",
                    name_prefix=f"{self.experiment_name}"
                ) for polid in self.policies_config
            ]
            no_improvement = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=training_config.no_improvement,
                min_evals=int(training_config.num_evals) * 0.25,
                verbose=1
            )

            eval_callback = CustomEvalCallback(
                in_bits=True if eval_env_config.binary == 'binary' else False,  # TODO this is really bad code
                log_path=self.eval_log_path,
                agent_roles=self.agent_roles,
                policy_mapping=self.simulator_policies,
                eval_env_config=eval_env_config,
                training_config=training_config,
                eval_env=eval_env,                    
                callback_after_eval=no_improvement,
                deterministic=False,
            )

            self.callbacks = checkpoint_callbacks
            self.eval_callback = eval_callback
            

        def step(self):
            self.simulator.learn(
                total_timesteps=self.total_timesteps,
                callbacks=self.callbacks,
                eval_callback=self.eval_callback,
            )

            logging_dict = defaultdict()
            mean_policy_scores = []

            for polid in self.policies_config:
                path = os.path.join(self.eval_log_path, "evaluations", f"policy_id_{polid}.npz")
                thing = np.load(path)
                mean_scores = np.mean(thing['results'], axis=-1)
                max_mean_eval = np.max(mean_scores)
                max_idx = np.argmax(mean_scores)
                mean_policy_scores.append(max_mean_eval)
                best_timestep = thing['timesteps'][max_idx]
                logging_dict.setdefault(f'policy_{polid}_best_result', max_mean_eval)
                logging_dict.setdefault(f'policy_{polid}_best_timestep', best_timestep)

            # for polid in range(len(set(self.agent_roles))):
            #     path = os.path.join(self.eval_log_path, "evaluations", f"role_id_{polid}.npz")
            #     thing = np.load(path)
            #     mean_scores = np.mean(thing['results'], axis=-1)
            #     max_mean_eval = np.max(mean_scores)
            #     max_idx = np.argmax(mean_scores)
            #     best_timestep = thing['timesteps'][max_idx]
            #     logging_dict.setdefault(f'role_{polid}_best_result', max_mean_eval)
            #     logging_dict.setdefault(f'role_{polid}_best_timestep', best_timestep)

            all_policy_scores = sum(mean_policy_scores)
            logging_dict.setdefault('all_policy_scores', all_policy_scores)
            self.logging_dict = logging_dict
            self.logging_dict.update({'step': 1})

            return logging_dict
        
        def save_checkpoint(self, tmp_checkpoint_dir):
            path = os.path.join(tmp_checkpoint_dir, "state.npz")
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

    return CustomTrainer

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config

    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist.")
        exit(1)
    else:
        print(f"Using config file: {config_path}")

    base_config = OmegaConf.load(config_path)
    orchestrator = SelfPlayOrchestrator(config=base_config)
    orchestrator.commence()
