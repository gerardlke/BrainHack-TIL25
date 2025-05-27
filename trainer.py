from re import L
import time
import inspect
from collections import deque
from typing import Any, Dict, List, Optional, Type, Union, TypeVar
from copy import deepcopy
from omegaconf import OmegaConf
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
from otherppos import ModifiedPPO
from stable_baselines3.common.utils import safe_mean
from einops import rearrange
from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor

from sb3_contrib.ppo_mask import MaskablePPO

class DummyGymEnv(gymnasium.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class RLRolloutSimulator(OnPolicyAlgorithm):
    """
    Acts as a wrapper around multiple policies and environments, calling them during rollouts
    and calling the policies train method afterwards.

    TODO: only supports PPO right now, because stepping is different for off and on-policy models
    """

    def __init__(
        self,
        train_env,
        train_env_config,
        policies_config,
        policy_mapping,
        callbacks,
        n_steps,
        tensorboard_log,
        verbose,
        use_action_masking,
    ):
        """
        All that is required is the configuration file.
        From there we will parse it into the appropriate components.
        Also, action masking is done here instead of in the environment

        Init flows as follows:
        -> Initialize training and evaluation environments.
        -> Define policy(ies), which utilize the environment observation space
        -> Define callbacks

        Rollout flows as follows:
        -> Reset environment to recieve initial observation.
        -> Index using policy agent indexes to obtain policy-specific observations.
        -> Per policy, do an action
        -> Aggregate all actions and step in the environment, rinsing and repeating.

        Args:
            - policies_config: Dictionary from omegaconf, policies can be access by 
                string of their policy id, e.g policies_config['0']
            - 

        """
        self.env = train_env
        self.num_vec_envs = self.env.num_envs
        self.policies_config = policies_config
        self.policy_mapping = policy_mapping
        self.action_masking = use_action_masking

        # check if policy_mapping and policy config have the same number of policies
        assert len(policies_config) == len(np.unique(np.array(self.policy_mapping))), f'Assertion failed. Environment suggests that there are {len(np.unique(self.policy_mapping))} ' \
            f'total policies, but recieved policy config only has {len(policies_config)} policies.'
        self.policy_agent_indexes = self.generate_policy_agent_indexes(
            n_envs=train_env_config.num_vec_envs, policy_mapping=self.policy_mapping
        )
        self.callbacks = callbacks

        self.n_steps = n_steps  # we forcibly standardize this across all agents, because if one collects fewer n_steps
        # than the other, that would be kinda wack

        # observation_space is flattened per agent, then concatenated across all agent.
        # this means that if the original flattened dict space is 36

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        print('self.observation_space', self.observation_space)

        self._vec_normalize_env = None

        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        self.dummy_envs = [DummyVecEnv([env_fn] * len(policy_index)) for policy_index in self.policy_agent_indexes]
        # this is a wrapper class, so it will not hold any states like
        # buffer_size, or action_noise. pass those straight to DQN.

        self.policies = []
        self._policies_config = deepcopy(policies_config)
        
        for polid, policy_config in self._policies_config.items():
            algo_type = eval(policy_config.algorithm)  # this will fail if the policy specified in the config
            # has not yet been imported. TODO do a registry if we aren't lazy?
            del policy_config.algorithm
            if hasattr(policy_config, 'n_steps'):
                assert policy_config.n_steps == self.n_steps, 'Assertion failed.' \
                    f'You passed in different n_steps for polid {polid} when you '
            else:
                policy_config.n_steps = self.n_steps
            if hasattr(policy_config, 'path'):
                policy = algo_type.load(
                    env = self.dummy_envs[polid],
                    tensorboard_log=self.tensorboard_log,
                    verbose=self.verbose,
                    **policy_config
                )
            else:
                policy = algo_type(
                    env = self.dummy_envs[polid],
                    tensorboard_log=self.tensorboard_log,
                    verbose=self.verbose,
                    **policy_config
                )

            self.policies.append(
                policy
            )

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
        print(f'NOTE: TOTAL TIMESTEPS {total_timesteps} INCLUDES NUMBER OF AGENT * VEC ENVIRONMENTS (currently {self.num_vec_envs}), AND IS NOT A PER-ENV BASIS')
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
        for polid, policy in enumerate(self.policies):
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

            callbacks[polid] = policy._init_callback(callbacks[polid])

        if eval_callback is not None:
            eval_callback = self._init_callback(eval_callback)
        # Call all callbacks back to call all backs, callbacks
        for callback in callbacks:
            callback.on_training_start(locals(), globals())
        if eval_callback is not None:
            eval_callback.on_training_start(locals(), globals())

        # self.env returns a dict, where each key is (M * N, ...), M is number of envs, N is number of agents.
        # we determine number of envs based on the output shape (should find a better way to do this)

        reset_obs = self.env.reset()
        n_rollout_steps = self.n_steps * self.num_vec_envs

        while self.num_timesteps < total_timesteps:
            # environment sampling. has to be done in this particular way because of
            # gridworld's perculiarities
            start = time.time()
            total_rewards, rollout_timesteps, continue_training = self.collect_rollouts(
                last_obs=reset_obs,
                n_rollout_steps=n_rollout_steps,  # rollout increments timesteps by number of envs
                callbacks=callbacks,
                eval_callback=eval_callback,
            )
            if not continue_training:
                break  # early stopping
            self.num_timesteps += rollout_timesteps

            # agent training.
            for polid, policy in enumerate(self.policies):
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
        continue_training = True
        all_last_episode_starts = [None] * self.num_policies
        all_clipped_actions = [None] * self.num_policies
        all_rewards = [None] * self.num_policies
        all_dones = [None] * self.num_policies
        all_infos = [None] * self.num_policies
        all_actions = [None] * self.num_policies
        all_values = [None] * self.num_policies
        all_log_probs = [None] * self.num_policies
        total_rewards = [[] for _ in range(self.num_policies)] 
        
        n_steps = 0
        # before formatted, last_obs is the direct return of self.env.reset()
        step_actions = np.empty(self.num_vec_envs, dtype=np.int64)

        last_obs_buffer = self.format_env_returns(last_obs, self.policy_agent_indexes, to_tensor=False)
        last_obs = self.format_env_returns(last_obs, self.policy_agent_indexes, device=self.policies[0].device, to_tensor=True)

        # iterate over policies, and do pre-rollout setups.
        # start = time.time()
        for polid, (policy, policy_index) in enumerate(zip(self.policies, self.policy_agent_indexes)):
            num_envs = len(policy_index)
            policy.policy.set_training_mode(False)
            policy.n_steps = 0
            policy.rollout_buffer.reset()

            if policy.use_sde:
                policy.policy.reset_noise(num_envs)  # type: ignore[operator]

            [callback.on_rollout_start() for callback in callbacks]
            eval_callback.on_rollout_start()
            policy._last_episode_starts = np.ones((num_envs,), dtype=bool)
            all_last_episode_starts[polid] = policy._last_episode_starts
        # print('------------PRE ROLLOUT TIME-------------')
        # print(time.time() - start)


        # do rollout
        while n_steps < n_rollout_steps:
            with torch.no_grad():
                for polid, policy in enumerate(self.policies):
                    if 'action_masks' not in inspect.signature(policy.policy.forward).parameters:
                        (
                            all_actions[polid],
                            all_values[polid],
                            all_log_probs[polid],
                        ) = policy.policy.forward(last_obs[polid])
                    else:
                        action_masks = self.get_action_masks(last_obs[polid])
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
                    # reshape the clipped actions
                    all_clipped_actions[polid] = clipped_actions
                # print('------------POLICY FORWARD TIME-------------')
                # print(time.time() - start)

            for polid, policy_agent_index in enumerate(self.policy_agent_indexes):
                step_actions[policy_agent_index] = all_clipped_actions[polid]

            # start = time.time()
            # actually step in the environment
            obs, rewards, dones, infos = self.env.step(step_actions)
            # print('obs', obs.shape)

            # print('policy_agent_indexes', policy_agent_indexes)
            all_curr_obs = self.format_env_returns(obs, policy_agent_indexes=self.policy_agent_indexes, to_tensor=True, device=self.policies[0].device)
            all_curr_obs_buffer = self.format_env_returns(obs, policy_agent_indexes=self.policy_agent_indexes, to_tensor=False)
            all_rewards = self.format_env_returns(rewards, policy_agent_indexes=self.policy_agent_indexes, to_tensor=False)
            all_dones = self.format_env_returns(dones, policy_agent_indexes=self.policy_agent_indexes, to_tensor=False)
            all_infos = self.format_env_returns(infos, policy_agent_indexes=self.policy_agent_indexes, to_tensor=False)

            for policy in self.policies:
                policy.num_timesteps += self.num_vec_envs

            [callback.update_locals(locals()) for callback in callbacks]
            [callback.on_step() for callback in callbacks]
            eval_callback.update_locals(locals())
            thing = eval_callback.on_step()
            if not thing:  # early stopping from StopTrainingOnNoModelImprovement
                continue_training = False

            n_steps += self.num_vec_envs
            # start = time.time()
            # add data to the rollout buffers
            for polid, policy in enumerate(self.policies):
                policy._update_info_buffer(all_infos[polid])
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                if hasattr(all_actions[polid], 'cpu'):
                    all_actions[polid] = all_actions[polid].cpu().numpy()
                # all_obs[polid] is a list[dict[str, np.ndarray]], but add only wants per dict. hence this loop
                policy.rollout_buffer.add(
                    obs=deepcopy(last_obs_buffer[polid]),
                    action=deepcopy(all_actions[polid]),
                    reward=deepcopy(all_rewards[polid]),
                    episode_start=deepcopy(policy._last_episode_starts),
                    value=deepcopy(all_values[polid]),
                    log_prob=deepcopy(all_log_probs[polid]),
                )
                policy._last_obs = all_curr_obs_buffer[polid]
                policy._last_episode_starts = all_dones[polid]
                total_rewards[polid].append(all_rewards[polid])
            # print('------------BUFFER ADDED TIME-------------')
            # print(time.time() - start)

            last_obs = all_curr_obs
            all_last_episode_starts = all_dones

        [callback.on_rollout_end() for callback in callbacks]
        eval_callback.on_rollout_end()

        return total_rewards, n_steps, continue_training

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

    def get_action_masks(self, observation):
        """
        action indexes:
        0 - move forward
        1 - move backward
        2 - turn left
        3 - turn right
        4 - stationary

        if agent is facing right:
        ACTION SPACE (absolute viewpoint as north 0) -> AGENT POV
        0 -> 2  'to move cardinal forward, agent must turn left'
        1 -> 3  'to move cardinal backward, agent must turn right'
        2 -> 1  'to move (turn) cardinal left, agent must move backwards'
        3 -> 0  'to move (turn) cardinal right, agent must move forward'

        if agent is facing left:
        0 -> 3
        1 -> 2
        2 -> 0
        3 -> 1

        if agent is facing up:
        # everything is normal

        if agent is facing down:
        0 -> 1
        1 -> 0
        2 -> 3
        3 -> 2

        observation bit indexes:
        Value of last 2 bits (tile & 0b11)	Meaning
        0	No vision
        1	Empty tile
        2	Recon (1 point)
        3	Mission (5 points)
        Bit index (least significant bit lowest)	Tile contains a...
        2	Scout
        3	Guard
        # the following are WITH RESPECT TO AGENT POV, NOT THE ABSOLUTE VIEW
        4	Right wall -> index 3 of bit split
        5	Bottom wall -> index 2 of bit split
        6	Left wall -> index 1 of bit split
        7	Top wall -> index 0 of bit split

        direction of facing:
        0 - right
        1 - bottom
        2 - left
        3 - top

        """
        viewcone = observation['viewcone']
        
        action_masks = np.zeros((viewcone.shape[0], self.action_space.n))
        test = rearrange(viewcone, 'A (S R C B) -> A S B R C', B=8, R=7, C=5)
        test = test[:, -1, :, :, :]

        # 1. Disable the corresponding action if there is a wall there.
        # step a: Rearrange the order of directions to fit action space order
        curr_tile_walls = test[:, :4, 2, 2]
        ### IN HERE, LETS SAY YOU GET [0, 0, 1, 1]. This translates to 
        # left of the agent, behind the agent, right of the agent, forward of agent. AGENT POV
        # i,e top and right of agent have a wall. why did ryan make it this way???
        # anyways lbrf -> fblr
        action_wall_indexes = np.array([3, 1, 0, 2])  # i.e index 3 of wall obs (top wall) maps to index 0 of which 
        # action is influenced by its presence
        enabled_actions = np.where(curr_tile_walls == 1, 0, 1)
        enabled_actions = enabled_actions[:, action_wall_indexes]
        # now that we've permuted from obs to action arrangement, permute according to
        # to agent's facing direction.

       
        # do other stuff like masking for scout if need to. later compile all information
        # to build this action mask. for now its just wall-enabled actions
        action_masks[:, :4] = enabled_actions

        # 0: only enable stationary action if all of them are disabled.
        null_action_rows = np.all(action_masks == 0, axis=1)
        action_masks[null_action_rows, -1] = 1

        return action_masks

    @staticmethod
    def format_env_returns(
        env_returns: dict[str, np.ndarray] | np.ndarray | list[dict],
        policy_agent_indexes: np.ndarray,
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
            policy_agent_indexes: list of list of integer, demoninating the indexes of the policy (corresponding to the list index)
                e.g [[1, 2, 3, 5, 6, 7], [0, 4]]  # first 6 map to a policy 0, second maps to a policy 1
            to_tensor: Return as tensors
            device: what device to map tensor to
        Returns:
            to_agents: list[np.ndarray] | list[dict[str, np.ndarray]], where first dimension of array is (num_envs.) and list is of length (num_agents).

        """
        num_policies = len(policy_agent_indexes)
        if to_tensor:
            assert device is not None, 'Assertion failed. format_env_returns function expects device to be stated if you want to run obs_as_tensor mutation.'
            mutate_func = obs_as_tensor
            kwargs = {'device': device}
        else:
            mutate_func = lambda x: x  # noqa: E731
            kwargs = {}
        # 1. appropriate indexing
        if isinstance(env_returns, dict):
            to_policies = [
                    {k: mutate_func(np.take(v, policy_agent_indexes[polid], axis=0), **kwargs) 
                        for k, v in env_returns.items()}
                for polid in range(num_policies)
            ]
        elif isinstance(env_returns, np.ndarray):
            # print('role indexes[polid]', [
            #     np.take(env_returns, policy_agent_indexes[polid], axis=0) for polid in range(num_policies)
            # ])
            # print('role indexes[polid]', [
            #     policy_agent_indexes[polid] for polid in range(num_policies)
            # ])
            to_policies = [
                mutate_func(np.take(env_returns, policy_agent_indexes[polid], axis=0), **kwargs) for polid in range(num_policies)
            ]
        elif isinstance(env_returns, list):
            # print('env_returns', env_returns)
            # print('role indexes[polid]', [
            #     policy_agent_indexes[polid] for polid in range(num_policies)
            # ])
            # print('role indexes[polid] tolist?', [
            #     policy_agent_indexes[polid].tolist() for polid in range(num_policies)
            # ])
            to_policies = [
                list(itemgetter( *(policy_agent_indexes[polid].tolist()) )(env_returns))
                if len(policy_agent_indexes[polid]) > 1
                else [itemgetter( *(policy_agent_indexes[polid].tolist()) )(env_returns)]
                for polid in range(num_policies)
            ]
        else:
            raise AssertionError(f'Assertion failed. format_env_returns recieved unexpected type {type(env_returns)}. \
                Expected dict[str, np.ndarray] or np.ndarray or list.')

        return to_policies
    
    @staticmethod
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
        Helper func to decode which indexes of observations of shape (self.num_vec_envs * self.num_agents, ...)
        map to which policies.
        """
        policy_mapping = last_obs['scout']
        policy_agent_indexes = [None] * self.num_policies
        for polid in range(self.num_policies):
            policy_agent_indexes[polid] = np.where(policy_mapping == polid)[0]

        return policy_agent_indexes

    @staticmethod
    def get_scout_from_obs(observation, already_bits=False, is_flattened=True, frame_stack_dim=0):
        """
        Overall handler function to do everything for us.
        Observation may be a dictionary or simple numpy array.
        It may be frame-stacked. It may also have been vectorized.
        Whatever way it is, we will rearrange it into an N, (C R B) array, where C R B values are hardcoded.
        """

        if isinstance(observation, dict):
            if 'scout' in observation:
                print('SCOUT IN OBS????', observation['scout'])
                eghtyjreg
            else:
                observation = observation['viewcone']

        if is_flattened:  # have to rely on user given frame_stack_dim to know along which dim was stacked
            if frame_stack_dim == -1:
                observation = rearrange(observation,
                    '(C R B N) -> (N C R B)',
                R=5, C=7, B=8)


        # we expect by this point, some (N, ...) array. the next code will flatten it into
        if not already_bits:
            observation = np.unpackbits(observation.astype(np.uint8))

        # by this point, we expect it to be a 
        print('observation before binary_obs', observation, observation.shape)
        binary_obs = rearrange(observation, 
            '(N C R B) -> N B C R', 
            R=5, C=7, B=8)
        is_scout = binary_obs[0, 5, 2, 2]

        return is_scout
    