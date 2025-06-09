from re import L
from utils import generate_policy_agent_indexes
import time
import inspect
from collections import deque
from typing import Any, Dict, List, Optional, Type, Union, TypeVar
from copy import deepcopy
from omegaconf import OmegaConf
from collections import defaultdict
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
import numpy as np
import torch
from operator import itemgetter
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (configure_logger, obs_as_tensor,
                                            safe_mean)
from stable_baselines3.common.vec_env import DummyVecEnv
from pettingzoo import ParallelEnv
from rl.db.db import RL_DB
from stable_baselines3.common.utils import safe_mean
from einops import rearrange
from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from otherppos import ModifiedPPO
from otherppos import ModifiedMaskedPPO

class DummyGymEnv(gymnasium.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class RLRolloutSimulator(OnPolicyAlgorithm):
    """
    Acts as a wrapper around multiple policies and environments, calling them during rollouts
    and calling the policies train method afterwards.

    TODO: only supports PPO right now, because stepping is different for off and on-policy models.
    Can we generalize this?
    """

    def __init__(
        self,
        db_path,
        train_env,
        train_env_config,
        policies_config,
        policy_mapping,
        callbacks,
        n_steps,
        tensorboard_log,
        verbose,
        use_action_masking,
        selfplay: bool = False,
    ):
        """
        All that is required is the configuration file.
        From there we will parse it into the appropriate components.
        Also, action masking is done here instead of in the environment. Mask function is local
        to policy, so we hold no warranties.

        This simulator will train one given policy at any given time. 
        This policy may map to several different roles. We would like to train them simultaneously (but not against each other at the same time, instead
        only facing prior checkpoints of themselves), such that despite training the same policy for different roles, they still share the same 
        parameters and are checkpointed as such. For this to occur, our train/eval environment will have to have n vector envs
        equivalent to the number of different agents each policy controls. An example should clarify:

        E.g policy mapping [0, 1, 1, 1]. We are currently training policy 1, and would like it to play-off against static opponents,
        hence two agents under still-learning policies cannot interact in the same environment. Hence, vectorize envs by 3, resulting in:
        [static, 1, static, static],
        [static, static, 1, static],
        [static, static, static, static],
        where static denotes a policy loaded in from DB (or random).

        This is honestly very scuffed, and requires that we pass in to the envs a None action for agents we do not control.
        The env will handle loading from there.
         
        Init flows as follows:
        -> Define policy(ies), which utilize the environment observation space
        -> Define policy-agent indexes, which are used to map the various indexes of the observation
            during rollout collection.
        -> Define callbacks

        Rollout flows as follows:
        -> Reset environment to recieve initial observation.
        -> Index using policy agent indexes to obtain policy-specific observations.
        -> Per policy, do an action
        -> Aggregate all actions and step in the environment, rinsing and repeating.

        Args:
            - db_path: Path to database where checkpoint pointers are stored.
            - train_env: Training environment to call step
            - train_env_config: Configuration of the training env.
            - policies_config: Dictionary from omegaconf, policies can be access by 
                string of their policy id, e.g policies_config['0']
            - policy_mapping: Maps policies to env indexes.
                Each element may be integer, representing the associated policy id, or None, which
                    represents a policy we are not in control of.
            - n_steps: Number of steps to take and to fill buffer up with, before training commences
            - use_action_masking: Boolean on whether or not action masking is being used.

        """
        self.selfplay = selfplay
        print('-----------------------SELFPLAY MODE IN SIMULATOR IS SET TO TRUE------------------') if self.selfplay else None
        self.db_path = db_path
        self.env = train_env
        self.total_envs = self.env.num_envs  # number of vec envs times number of agents from parallelization
        self.vec_envs = train_env_config.num_vec_envs
        self.policies_config = policies_config
        self.policy_mapping = policy_mapping
        self.action_masking = use_action_masking

        # check if policy_mapping and policy config have the same number of policies
        all_policies_there = len(policies_config) == len(set(self.policy_mapping))
        
        if not all_policies_there:
            print(f'WARNING: Main configuration suggests that there are {len(set(self.policy_mapping))} ' \
                f'total policies, but recieved policy configuration has {len(policies_config)} policies. ' \
                    'This mismatch should only be the case if self-play is being conducted.')
        
        self.policy_agent_indexes = generate_policy_agent_indexes(
            n_envs=self.vec_envs, policy_mapping=self.policy_mapping, selfplay=self.selfplay
        )
        self.callbacks = callbacks

        self.n_steps = n_steps  # we forcibly standardize this across all agents, because if one collects fewer n_steps
        # than the other, that would be kinda wack

        # observation_space is flattened per agent, then concatenated across all agent.
        # this means that if the original flattened dict space is 36

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        self._vec_normalize_env = None

        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        self.dummy_envs = [DummyVecEnv([env_fn] * len(policy_index)) for _, policy_index in self.policy_agent_indexes.items()]
        # this is a wrapper class, so it will not hold any states like
        # buffer_size, or action_noise. pass those straight to DQN.

        self.policies = {}
        self._policies_config = deepcopy(policies_config)
        
        for polid, policy_config in self._policies_config.items():
            algo_type = eval(policy_config.algorithm)  # this will fail if the policy specified in the config
            # has not yet been imported. TODO do a registry if we aren't lazy?
            _policy_config = deepcopy(policy_config)
            del _policy_config.algorithm  # so it doesnt interfere with init
            if hasattr(_policy_config, 'n_steps'):
                assert _policy_config.n_steps == self.n_steps, 'Assertion failed.' \
                    f'You passed in different n_steps for polid {polid} when you '
            else:
                _policy_config.n_steps = self.n_steps
            if hasattr(_policy_config, 'path'):
                policy = algo_type.load(
                    env = self.dummy_envs[polid],
                    tensorboard_log=self.tensorboard_log,
                    verbose=self.verbose,
                    **_policy_config
                )
            else:
                # try to ping db to get best checkpoint for this policy and this role.
                self.db = RL_DB(db_file=self.db_path, num_roles=len(self.policy_mapping))
                self.db.set_up_db(timeout=100)
                checkpoints = self.db.get_checkpoint_by_policy(policy=polid, shuffle=False)

                if len(checkpoints) == 0:  # train from scratch.
                    policy = algo_type(
                        env = self.dummy_envs[polid],
                        tensorboard_log=self.tensorboard_log,
                        verbose=self.verbose,
                        **_policy_config
                    )
                else:
                    # EXPLODE FOR NOW TODO CODE FOR MEAN SCORE GRAB
                    checkpoint_scores = {}
                    for checkpoint in checkpoints:
                        scores = [v for k, v in dict(checkpoint).items() if 'score' in k]
                        checkpoint_scores[checkpoint] = sum(scores) / len(scores)

                    best_checkpoint = max(checkpoint_scores, key=checkpoint_scores.get)
                    filepath = best_checkpoint['filepath']
                    print('best checkpoint fp', filepath)
                    policy = algo_type.load(
                        env = self.dummy_envs[polid],
                        tensorboard_log=self.tensorboard_log,
                        verbose=self.verbose,
                        path=filepath,
                        **_policy_config
                    )

            self.policies[polid] = policy

        self.num_policies = len(self.policies)

        
    def learn(
        self,
        total_timesteps: int,
        callbacks: Optional[list[MaybeCallback]] = None,
        eval_callback: Optional[MaybeCallback] = None,
        log_interval: int = 1,
        tb_log_name: str = "RLRolloutSimulator",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Main learn function. Mainly goes as follows:

        target: collect rollbacks and train up till total_timesteps.
        """
        print(f'NOTE: TOTAL TIMESTEPS {total_timesteps} INCLUDES NUMBER OF AGENT * VEC ENVIRONMENTS (currently {self.total_envs}), AND IS NOT A PER-ENV BASIS')
        if callbacks is not None:
            assert len(callbacks) == self.num_policies, 'callbacks must a list of num_policies number of nested lists'
            
        self.num_timesteps = 0
        all_total_timesteps = []
        
        self._logger = configure_logger(
            self.verbose,
            self.tensorboard_log,
            tb_log_name,
            reset_num_timesteps,
        )
        logdir = self._logger.dir

        # Setup for each policy. Reset things, setup timestep tracking things.
        # replace certain agent attributes
        for idx, (polid, policy) in enumerate(self.policies.items()):
            policy.start_time = time.time()
            if policy.ep_info_buffer is None or reset_num_timesteps:
                policy.ep_info_buffer = deque(maxlen=policy._stats_window_size)
                policy.ep_success_buffer = deque(maxlen=policy._stats_window_size)

            if policy.action_noise is not None:
                policy.action_noise.reset()

            if reset_num_timesteps:
                policy.num_timesteps = 0
                policy._episode_num = 0
                all_total_timesteps.append(total_timesteps)
                policy._total_timesteps = total_timesteps
            else:
                # make sure training timestamps are ahead of internal counter
                all_total_timesteps.append(total_timesteps + policy.num_timesteps)
                policy._total_timesteps = total_timesteps + policy.num_timesteps

            policy._logger = configure_logger(
                policy.verbose,
                logdir,
                f"policy_{polid}",
                reset_num_timesteps,
            )

            callbacks[idx] = policy._init_callback(callbacks[idx])

        if eval_callback is not None:
            eval_callback = self._init_callback(eval_callback)
        # Call all callbacks back to call all backs, callbacks
        for callback in callbacks:
            callback.on_training_start(locals(), globals())
        if eval_callback is not None:
            eval_callback.on_training_start(locals(), globals())

        # self.env returns a dict, where each key is (M * N, ...), M is number of envs, N is number of agents.
        # we determine number of envs based on the output shape (should find a better way to do this)

        last_obs = self.env.reset()
        last_obs_buffer = None

        n_rollout_steps = self.n_steps * self.total_envs
        print('n_rollout_steps', self.n_steps, self.total_envs)
        
        while self.num_timesteps < total_timesteps:
            # environment sampling. has to be done in this particular way because of
            # gridworld's perculiarities
            start = time.time()
            total_rewards, rollout_timesteps, continue_training, last_obs, last_obs_buffer = self.collect_rollouts(
                last_obs=last_obs,
                n_rollout_steps=n_rollout_steps,  # rollout increments timesteps by number of envs
                callbacks=callbacks,
                eval_callback=eval_callback,
                last_obs_buffer=last_obs_buffer,
            )
            if not continue_training:
                break  # early stopping
            self.num_timesteps += rollout_timesteps

            # agent training.
            for polid, policy in self.policies.items():
                policy.num_timesteps += rollout_timesteps
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps  # 
                )
                if log_interval is not None and policy.num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    policy_config = self._policies_config[polid]
                    [policy.logger.record(
                        k, v
                    ) for k, v in policy_config.items()]
                    policy.logger.record("polid", polid, exclude="tensorboard")
                    policy.logger.record(
                        "time/iterations", policy.num_timesteps, exclude="tensorboard"
                    )
                    concat = np.concatenate(total_rewards[polid])
                    mean_policy_reward = (np.sum(concat) / len(concat)).item()
                    policy.logger.record(
                        f"rollout/mean_policy_reward_polid_{polid}", mean_policy_reward,
                    )
                    policy.logger.record("time/fps", fps)
                    policy.logger.record(
                        "time/time_elapsed",
                        int(time.time() - policy.start_time),
                        exclude="tensorboard",
                    )
                    policy.logger.record(
                        "time/total_timesteps",
                        policy.num_timesteps,
                        exclude="tensorboard",
                    )
                    policy.logger.dump(step=policy.num_timesteps)
                start = time.time()
                policy.train()
                print(f'------------POLICY {polid} TRAIN TIME-------------')
                print(time.time() - start)

        for callback in callbacks:
            callback.on_training_end()
        if eval_callback is not None:
            eval_callback.on_training_end()

    def collect_rollouts(
            self,
            last_obs,
            n_rollout_steps: int,
            eval_callback,
            callbacks: list = [],
            last_obs_buffer = None
        ):
        """
        Helper function to collect rollouts (sample the env and feed observations into the agents, vice versa)
        This function will be fed by an exiting last_observation; that is the one generated by self.env.reset.

        Trouble arises when we deal with ActorCriticPolicy (dictionary of inputs) since this code was normally not built for it.
        Lets break it down.

        1. self.env.reset() should return dict[str, np.ndarray], where each array has (num_envs * num_agents, ...) shape.
        2. We must split these into each agents respective observations. Split into list of dict[str, np.ndarray], where length of list is equal to
            num_envs, and np.ndarrays are now only (num_agents, ...). This is what each agent accepts.
        3. Agent outputs a list of actions. Concat each of these actions into a (num_envs * num_agents) length list of actions.
        4. We step in the env and get dict[str, np.ndarray] again.
        5. When caching steps into each agent's replaybuffer, we repeat step 2's formatting.
        
        """
        # temporary lists to hold things before saved into history_buffer of agent.
        print('--------in rollouts-------------')
        continue_training = True
        all_last_episode_starts = {}
        all_clipped_actions = {}
        all_rewards = {}
        all_dones = {}
        all_infos = {}
        all_actions = {}
        all_values = {}
        all_log_probs = {}
        all_action_masks = {}
        last_values = {}
        total_rewards = {polid: [] for polid, policy in self.policies.items()}
        
        n_steps = 0
        step_actions = np.full(self.total_envs, -1)
        the_first_key = list(self.policies.keys())[0]  # idk what keys self.policies will have sooo just get any one

        if last_obs_buffer is None:
            last_obs_buffer = self.format_env_returns(last_obs, self.policy_agent_indexes, to_tensor=False)
            last_obs = self.format_env_returns(last_obs, self.policy_agent_indexes, device=self.policies[the_first_key].device, to_tensor=True)
        # iterate over policies, and do pre-rollout setups.
        for idx, ((_, policy), (_, policy_index)) in enumerate(zip(self.policies.items(), self.policy_agent_indexes.items())):
            num_envs = len(policy_index)
            policy.policy.set_training_mode(False)
            policy.n_steps = 0
            policy.rollout_buffer.reset()

            if policy.use_sde:
                policy.policy.reset_noise(num_envs)  # type: ignore[operator]

            [callback.on_rollout_start() for callback in callbacks]
            eval_callback.on_rollout_start()
            policy._last_episode_starts = np.ones((num_envs,), dtype=bool)
            all_last_episode_starts[idx] = policy._last_episode_starts

        # do rollout
        while n_steps < n_rollout_steps:
            with torch.no_grad():
                for polid, policy in self.policies.items():
                    if 'action_masks' not in inspect.signature(policy.policy.forward).parameters:
                        (
                            all_actions[polid],
                            all_values[polid],
                            all_log_probs[polid],
                        ) = policy.policy.forward(last_obs[polid])
                    else:
                        action_masks = policy.get_action_masks(last_obs[polid])
                        all_action_masks[polid] = action_masks
                        # print('action_masks', action_masks)
                        (
                            all_actions[polid],
                            all_values[polid],
                            all_log_probs[polid],
                        ) = policy.policy.forward(last_obs[polid], action_masks=action_masks)
  
                    if hasattr(all_actions[polid], 'cpu'):
                        all_actions[polid] = all_actions[polid].cpu().numpy()
                    clipped_actions = all_actions[polid]
                    if isinstance(self.action_space, Box):
                        clipped_actions = np.clip(
                            clipped_actions,
                            self.action_space.low,
                            self.action_space.high,
                        )
                    elif isinstance(self.action_space, Discrete):
                        # get integer from numpy array
                        clipped_actions = np.array(
                            [action.item() for action in clipped_actions]
                        )
                    all_clipped_actions[polid] = clipped_actions
            
            # TODO: SETTLE HOW WE PASS TO STEP
            for polid, actions in all_clipped_actions.items():
                policy_agent_index = self.policy_agent_indexes[polid]
                # [step_actions.__setitem__(env_idx, actions[i]) for i, env_idx in enumerate(policy_agent_index)]
                step_actions[policy_agent_index] = actions
            # actually step in the environment
            obs, rewards, dones, infos = self.env.step(step_actions)

            all_curr_obs = self.format_env_returns(obs, policy_agent_indexes=self.policy_agent_indexes, to_tensor=True, device=self.policies[the_first_key].device)
            all_curr_obs_buffer = self.format_env_returns(obs, policy_agent_indexes=self.policy_agent_indexes, to_tensor=False)
            all_rewards = self.format_env_returns(rewards, policy_agent_indexes=self.policy_agent_indexes, to_tensor=False)
            all_dones = self.format_env_returns(dones, policy_agent_indexes=self.policy_agent_indexes, to_tensor=False)
            all_infos = self.format_env_returns(infos, policy_agent_indexes=self.policy_agent_indexes, to_tensor=False)

            for polid, policy in self.policies.items():
                policy.n_steps += self.total_envs

            [callback.update_locals(locals()) for callback in callbacks]
            # this will break if you pass in any other callback or make this into a CallbackList
            # TODO: dont be lazy.
            
            eval_callback.update_locals(locals())
            thing, policy_episode_rewards = eval_callback.on_step()  # dict of {polid: [role_idx_score, ...] * n_episodes}
            if isinstance(policy_episode_rewards, dict):

                trainable_policy_episode_reward = list(policy_episode_rewards.values())[0]  # can only train one policy at a time moment
                score_dict = {i: np.mean(trainable_policy_episode_reward[i::4]).item() for i in range(4)}

            else:
                score_dict = None
            
            [
                callback.on_step(
                    hparams=dict(self._policies_config[polid]),
                    score_dict=score_dict,
            ) for callback, polid in zip(callbacks, self.policies)]
            if not thing:  # early stopping from StopTrainingOnNoModelImprovement
                continue_training = False

            n_steps += self.total_envs
            # start = time.time()
            # add data to the rollout buffers
            for polid, policy in self.policies.items():
                policy._update_info_buffer(all_infos[polid])
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                if hasattr(all_actions[polid], 'cpu'):
                    all_actions[polid] = all_actions[polid].cpu().numpy()
                if 'action_masks' not in inspect.signature(policy.rollout_buffer.add).parameters:
                    policy.rollout_buffer.add(
                        obs=deepcopy(last_obs_buffer[polid]),
                        action=deepcopy(all_actions[polid]),
                        reward=deepcopy(all_rewards[polid]),
                        episode_start=deepcopy(policy._last_episode_starts),
                        value=deepcopy(all_values[polid]),
                        log_prob=deepcopy(all_log_probs[polid]),
                    )
                else:
                    policy.rollout_buffer.add(
                        obs=deepcopy(last_obs_buffer[polid]),
                        action=deepcopy(all_actions[polid]),
                        reward=deepcopy(all_rewards[polid]),
                        episode_start=deepcopy(policy._last_episode_starts),
                        value=deepcopy(all_values[polid]),
                        log_prob=deepcopy(all_log_probs[polid]),
                        action_masks=deepcopy(all_action_masks[polid])
                    )
                total_rewards[polid].append(all_rewards[polid])
                policy._last_obs = all_curr_obs_buffer[polid]
                policy._last_episode_starts = all_dones[polid]

            last_obs = all_curr_obs
            last_obs_buffer = all_curr_obs_buffer  # APPARENTLY I WAS JUST ADDING THE SAME OBSERVATION AGAIN AND AGAIN NO WONDER CCB
            all_last_episode_starts = all_dones
            
        with torch.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            for polid, policy in self.policies.items():
                values = policy.policy.predict_values(last_obs[polid])  # type: ignore[arg-type]
                policy.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=all_last_episode_starts[polid])

        [callback.on_rollout_end() for callback in callbacks]
        eval_callback.on_rollout_end()
        return total_rewards, n_steps, continue_training, last_obs, last_obs_buffer

    def load_policy_id(
        self,
        path,
        policy_id: int,
        **kwargs
    ):
        assert policy_id < len(self.policies), f'policy_id {policy_id} passed in does not exist in already initialized algorithm'
        self.policies[policy_id].load(
            policy='MultiInputLstmPolicy',
            path=path,
            env=self.dummy_envs[policy_id],
            n_steps=self.n_steps,
            **kwargs
            )
    
    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    
    @staticmethod
    def format_env_returns(
        env_returns: dict[str, np.ndarray] | np.ndarray | list[dict],
        policy_agent_indexes: dict,
        to_tensor=True,
        device=None,
    ):
        """
        Helper function to format returns based on if they are a dict of arrays or just arrays.
        We expect the first dimension of these arrays to be (num_envs * num_agents).

        The flow is as follows:
        1. Use indexes to extract the appropriate observations per policy.
        Thats it

        Args:
            env_returns: dict[str, np.ndarray] | np.ndarray | list. We expect the first dimension of these arrays / length of list to be (num_envs * num_agents).
            policy_agent_indexes: dict of int -> list of integer, demoninating the indexes of the policy (corresponding to the list index)
                e.g {
                    0: [1, 2, 3, 5, 6, 7], 
                    1: [0, 4],
                }  # first 6 map to a policy 0, second maps to a policy 1
            to_tensor: Return as tensors
            device: what device to map tensor to
        Returns:
            to_agents: list[np.ndarray] | list[dict[str, np.ndarray]], where first dimension of array is (num_envs.) and list is of length (num_agents).

        """
        # num_policies = len([k for k in policy_agent_indexes.keys() if k is not None])
        if to_tensor:
            assert device is not None, 'Assertion failed. format_env_returns function expects device to be stated if you want to run obs_as_tensor mutation.'
            mutate_func = obs_as_tensor
            kwargs = {'device': device}
        else:
            mutate_func = lambda x: x  # noqa: E731
            kwargs = {}
        # 1. appropriate indexing
        if isinstance(env_returns, dict):
            to_policies = {
                    key: {k: mutate_func(np.take(v, policy_agent_indexes[key], axis=0), **kwargs) 
                        for k, v in env_returns.items()}
                for key in policy_agent_indexes.keys() if key is not None
            }
        elif isinstance(env_returns, np.ndarray):
            to_policies = {
                key: mutate_func(np.take(env_returns, policy_agent_indexes[key], axis=0), **kwargs)
                for key in policy_agent_indexes.keys() if key is not None
            }
        elif isinstance(env_returns, list):
            to_policies = {
                key: list(itemgetter( *(policy_agent_indexes[key]) )(env_returns))
                if len(policy_agent_indexes[key]) > 1
                else [itemgetter( *(policy_agent_indexes[key]) )(env_returns)]
                for key in policy_agent_indexes.keys() if key is not None
            }
        else:
            raise AssertionError(f'Assertion failed. format_env_returns recieved unexpected type {type(env_returns)}. \
                Expected dict[str, np.ndarray] or np.ndarray or list.')

        return to_policies


    @staticmethod
    def get_2_policy_agent_indexes_from_obs(last_obs, case, is_binarized):
        """
        A rather hardcoded, inflexible function. Currently built to handle for the following cases:

        1. Flattened dictionary  (will need to access hardcoded indexes according to shape)
        2. Flattened viewcone

        We will also do some checks on stacking, for safety.

        Dimension notation:
            - A: Env dimension / ParallelEnv * Agent dimension
            - S: Arbitrary stacking dimension.
            - B: Binary dimension of viewcone, if the observation converted to bits (should be 8)
            - C: Column dimension of viewcone
            - R: Row dimension of viewcone
            - D: Flattened dictionary dimension of arbitrary size (we assume viewcone is at the end of each flattened dict list)
        """
        # hardcoded for viewcone. if this ever fails, goodluck!
        # TODO: maybe pass all the way from source environment?
        r = 5
        c = 7
        b = 8
        assert isinstance(is_binarized, bool)
        assert case in ['flat_dict', 'flat_viewcone'], 'Assertion failed.' \
            'Case specified is not in the above list of supported observation types.'
        if case == 'flat_dict':
            # shape should be A (S D)
            if is_binarized:
                last_obs = rearrange(last_obs, 
                    'A (S D) -> A S D', 
                    )
                is_scout = last_obs[:, 0, 5, 2, 2]
        if len(last_obs.shape) == 2:
            last_obs = rearrange(last_obs, 
                'A (S C R B) -> A S B C R', 
                R=5, C=7, B=8)
            is_scout = last_obs[:, 0, 5, 2, 2]
        else:
            bits = np.unpackbits(last_obs[:, :, 2, 2][np.newaxis, :, :].astype(np.uint8), axis=0)
            is_scout = bits[5, :, 0]

        policy_agent_indexes = [
            np.where(is_scout == 0)[0], np.where(is_scout == 1)[0]
        ]

        return policy_agent_indexes

    @classmethod
    def load(
        cls,
        path: str,
        n_steps: int,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: GymEnv,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ) -> "IndependentRecurrentPPO":
        """
        Lazy method to init class and load in weights to each policy.
        For individual policy loading see load_guard and load_scout methods.
        """
        model = cls(
            policy=policy,
            num_agents=num_agents,
            n_steps=n_steps,
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs,
        )
        for polid in range(model.num_policies):
            model.policies[polid] = RecurrentPPO.load(
                policy='MultiInputLstmPolicy',
                path=path + f"/policy_{polid}/model",
                env=model.dummy_envs[polid],
                n_steps=n_steps,
                **kwargs
            )
        return model

    def save(self, path: str) -> None:
        for polid in range(self.num_policies):
            self.policies[polid].save(path=path + f"/policy_{polid}/model")


    def get_policy_agent_indexes_from_scout(self, last_obs):
        """
        Helper func to decode which indexes of observations of shape (self.total_envs * self.num_agents, ...)
        map to which policies.
        """
        policy_mapping = last_obs['scout']
        policy_agent_indexes = {}
        for polid in range(self.num_policies):
            policy_agent_indexes[polid] = np.where(policy_mapping == polid)[0]

        return policy_agent_indexes
