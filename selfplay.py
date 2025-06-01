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
from selfplay_trainer import RLRolloutSimulator
from self_play_env import build_env
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

class SelfPlayOrchestrator:
    """
    A wrapper around ray tune for training. This orchestrates agent selection and policy loading,
    continuing to loop until some stopping condition (TODO decide this). 

    While stopping condition has not passed:
    1. Select an agent, load in its corresponding policy.
    2. Initialize ray tune and trainable. Tune allows for us to try how different hyperparameters fare in a trial.
    3. Obtain the best result, checkpoint it into the DB, and loop back to 1.

    """

    def __init__(self, config):
        """
        Initialize orchestration.
        Most important configurations are what agents we are training, and what agents are part of the environment.
        """
        self.config = config

    def commence(self):
        """
        Commences training.
        For all collections of selectable agents, build the trainable and call ray tune to optimize it.
        """
        # we loop over agent_roles to know how many agents there are.
        tmp_config = deepcopy(self.config)
        agent_roles = tmp_config.agent_roles

        for agent_role in agent_roles:
            tmp_config.selected_agent = agent_role
            print('-------------------SELECTED AGENT---------------------')
            # for now, made for the case of agent_roles being unique.
            npcs = [1 if agent != agent_role else 0 for agent in agent_roles]
            # boolean mask of npcs: 0 represents the selected agent, 1 represents an agent controlled by the environment.
            policies_controlled_here = [agent_role] # integer indexes of policy controlled here ( for now, just one. )
            policies_env = [agent for agent in agent_roles if agent != agent_role] # integer indexes of policies controlled here
            # apply a mask over specific policies in the config
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
            tmp_config.npcs = npcs
            print(tmp_config)

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
            
            trainable_cls = tune.with_parameters(trainable, base_config=tmp_config) # this is where the edited config
            # gets passed to the trainable. environment is initialized within trainable.
        
            tuner = tune.Tuner(
                tune.with_resources(trainable_cls, resources={"cpu": 5}),
                tune_config=tune.TuneConfig(
                        scheduler=pbt,
                        num_samples=10,
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
            print('policies_hparams', policies_hparams)
            print('policies_config', policies_config)
            assert len(policies_hparams) == len(policies_config), 'Assertion failed, mismatching number of policies as specified by tune configuration,' \
                'and number of policies under the policies section. Failing gracefully.'
            num_policies = len(policies_hparams)
            for polid, incoming_policy_config in policies_hparams.items():
                policies_config[polid] = replace_and_report(policies_config[polid], incoming_policy_config, merge=True)
            
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
            _, train_env = build_env(
                reward_names=CustomRewardNames,
                rewards_dict=REWARDS_DICT,
                policy_mapping=base_config.policy_mapping,
                agent_roles=base_config.agent_roles,
                self_play=base_config.self_play,
                npcs=base_config.npcs,
                db_path=base_config.db_path,
                env_config=train_env_config,
            )

            _, eval_env = build_env(
                reward_names=CustomRewardNames,
                rewards_dict=STD_REWARDS_DICT,
                policy_mapping=base_config.policy_mapping,
                agent_roles=base_config.agent_roles,
                self_play=base_config.self_play,
                npcs=base_config.npcs,
                db_path=base_config.db_path,
                env_config=eval_env_config,
            )

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
                use_action_masking=training_config.use_action_masking,
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
            

        def step(self):
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
