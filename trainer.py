from re import L
import time
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
from modifiedppo import ModifiedPPO
from stable_baselines3.common.utils import safe_mean
from einops import rearrange
from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv


from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor

class DummyGymEnv(gymnasium.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class RLRolloutSimulator:
    """
    Acts as a wrapper around multiple policies and environments, calling them during rollouts
    and calling the policies train method afterwards.

    TODO: only supports PPO right now, because stepping is different for off and on-policy models
    """

    def __init__(
        self,
        train_env,
        policies_config,
        policy_agent_indexes,
        callbacks,
        n_steps,
        tensorboard_log,
        verbose,
    ):
        """
        All that is required is the configuration file.
        From there we will parse it into the appropriate components.

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
        self.train_env = train_env
        self.policies_config = policies_config
        self.policy_agent_indexes = policy_agent_indexes
        self.callbacks = callbacks

        self.n_steps = n_steps  # we forcibly standardize this across all agents, because if one collects fewer n_steps
        # than the other, that would be kinda wack

        assert len(self.train_env.observation_space.shape) == 1, 'Assertion failed. We only support training environments' \
            'where everything is flattened, just to standardize things.'
        # observation_space is flattened per agent, then concatenated across all agent.
        # this means that if the original flattened dict space is 36

        self.observation_space = self.train_env.observation_space
        self.action_space = self.train_env.action_space
        self._vec_normalize_env = None

        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        self.dummy_envs = [DummyVecEnv([env_fn] * len(policy_index)) for policy_index in self.policy_agent_indexes]
        # this is a wrapper class, so it will not hold any states like
        # buffer_size, or action_noise. pass those straight to DQN.

        self.policies = []
        for polid, policy_config in policies_config.items():
            policy_type = eval(policy_config.policy)  # this will fail if the policy specified in the config
            # has not yet been imported. TODO do a registry if we aren't lazy?
            print('policy_type??', policy_type)
            print('policies_config', policies_config)
            # policies_config = OmegaConf.to_container(policies_config, resolve=True)
            # print('policies_config after dump', policies_config)

            self.policies.append(
                policy_type(
                    env = self.dummy_envs[polid],
                    **policy_config
                )
            )

        
    def learn(
        self,
        total_timesteps: int,
        eval_callback: Optional[List[List[MaybeCallback]]] = None,
        callbacks: Optional[List[List[MaybeCallback]]] = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Main learn function. Mainly goes as follows:

        target: collect rollbacks and train up till total_timesteps.
        """
        print(f'NOTE: TOTAL TIMESTEPS {total_timesteps} INCLUDES NUMBER OF ENVIRONMENTS (currently {self.num_vec_envs}), AND IS NOT A PER-ENV BASIS')
        if eval_callback is not None:
            eval_callback = self._init_callback(eval_callback)

        if callbacks is not None:
            assert len(callbacks) == self.num_policies, 'callbacks must a list of num_policies number of nested lists'
            assert all(isinstance(callback, list) for callback in callbacks), 'callbacks must a list of num_policies number of nested lists'

        self.num_timesteps = 0
        all_total_timesteps = []
        
        self._logger = configure_logger(
            self.verbose,
            self.tensorboard_log,
            tb_log_name,
            reset_num_timesteps,
        )
        logdir = self.logger.dir

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

        # Call all callbacks back to call all backs, callbacks
        for callback in callbacks:
            callback.on_training_start(locals(), globals())

        # self.env returns a dict, where each key is (M * N, ...), M is number of envs, N is number of agents.
        # we determine number of envs based on the output shape (should find a better way to do this)
        reset_obs = self.env.reset()
        n_rollout_steps = self.n_steps * self.num_vec_envs

        while self.num_timesteps < total_timesteps:
            # environment sampling. has to be done in this particular way because of
            # gridworld's perculiarities
            start = time.time()
            total_rewards, rollout_timesteps = self.collect_rollouts(
                last_obs=reset_obs,
                n_rollout_steps=n_rollout_steps,  # rollout increments timesteps by number of envs
                callbacks=callbacks,
                eval_callback=eval_callback,
            )
            self.num_timesteps += rollout_timesteps

            # agent training.
            for polid, policy in enumerate(self.policies):
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps  # 
                )
                if log_interval is not None and policy.num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
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

    def eval(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Main eval function. Mainly acts as a wrapper around self.collect_rollouts
        """
        print(f'NOTE: TOTAL TIMESTEPS {total_timesteps} INCLUDES NUMBER OF ENVIRONMENTS (currently {self.num_vec_envs}), AND IS NOT A PER-ENV BASIS')

        self._logger = configure_logger(
            self.verbose,
            self.tensorboard_log,
            tb_log_name,
            reset_num_timesteps,
        )
        logdir = self.logger.dir

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
                policy._total_timesteps = total_timesteps
            else:
                # make sure training timestamps are ahead of internal counter
                policy._total_timesteps = total_timesteps + policy.num_timesteps

            policy._logger = configure_logger(
                policy.verbose,
                logdir,
                f"policy_{polid}",
                reset_num_timesteps,
            )

        # self.env returns a dict, where each key is (M * N, ...), M is number of envs, N is number of agents.
        # we determine number of envs based on the output shape (should find a better way to do this)
        reset_obs = self.env.reset()
        n_rollout_steps = total_timesteps * self.num_vec_envs
        total_rewards = self.collect_rollouts(
            last_obs=reset_obs,
            n_rollout_steps=n_rollout_steps,  # rollout increments timesteps by number of envs
            callbacks=[],
        )
        return total_rewards

        # agent training.
        # for polid, policy in enumerate(self.policies):
        #     policy._update_current_progress_remaining(
        #         policy.num_timesteps, total_timesteps  # 
        #     )
        #     if log_interval is not None and num_timesteps % log_interval == 0:
        #         fps = int(policy.num_timesteps / (time.time() - policy.start_time))
        #         policy.logger.record("polid", polid, exclude="tensorboard")
        #         policy.logger.record(
        #             "time/iterations", num_timesteps, exclude="tensorboard"
        #         )
        #         if (
        #             len(policy.ep_info_buffer) > 0
        #             and len(policy.ep_info_buffer[0]) > 0
        #         ):
        #             print('rollout/ep_rew_mean', 
        #             safe_mean(
        #                     [ep_info["r"] for ep_info in policy.ep_info_buffer]
        #                 ),)
        #             print('rollout/ep_len_mean', 
        #             safe_mean(
        #                     [ep_info["l"] for ep_info in policy.ep_info_buffer]
        #                 ),)
        #             policy.logger.record(
        #                 "rollout/ep_rew_mean",
        #                 safe_mean(
        #                     [ep_info["r"] for ep_info in policy.ep_info_buffer]
        #                 ),
        #             )
        #             policy.logger.record(
        #                 "rollout/ep_len_mean",
        #                 safe_mean(
        #                     [ep_info["l"] for ep_info in policy.ep_info_buffer]
        #                 ),
        #             )
        #         policy.logger.record("time/fps", fps)
        #         policy.logger.record(
        #             "time/time_elapsed",
        #             int(time.time() - policy.start_time),
        #             exclude="tensorboard",
        #         )
        #         policy.logger.record(
        #             "time/total_timesteps",
        #             policy.num_timesteps,
        #             exclude="tensorboard",
        #         )
        #         policy.logger.dump(step=policy.num_timesteps)

            # policy.train()

        # for callback in callbacks:
        #     callback.on_training_end()

    def collect_rollouts(
            self,
            last_obs,
            n_rollout_steps: int,
            callbacks: list = [],
            eval_callback: Optional[MaybeCallback] = None,
            eval: bool = False,
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
        all_last_episode_starts = [None] * self.num_policies
        all_clipped_actions = [None] * self.num_policies
        all_rewards = [None] * self.num_policies
        all_dones = [None] * self.num_policies
        all_infos = [None] * self.num_policies
        all_actions = [None] * self.num_policies
        all_lstm_states = [None] * self.num_policies
        all_values = [None] * self.num_policies
        all_log_probs = [None] * self.num_policies
        total_rewards = [[] for _ in range(self.num_policies)] 
        
        n_steps = 0
        # before formatted, last_obs is the direct return of self.env.reset()
        step_actions = np.empty(self.num_agents * self.num_vec_envs, dtype=np.int64)
        policy_agent_indexes = self.get_policy_agent_indexes_from_viewcone(last_obs=last_obs)
        # the boolean mask to apply on each observation, mapping each to each policy.
        # as a sanity check, this should be [1 0 0 0] * self.num_vec_envs since self.env is freshly initialized.
        last_obs_buffer = self.format_env_returns(last_obs, policy_agent_indexes, to_tensor=False)
        last_obs = self.format_env_returns(last_obs, policy_agent_indexes, device=self.policies[0].device, to_tensor=True)

        # iterate over policies, and do pre-rollout setups.
        # start = time.time()
        for polid, (policy, num_envs) in enumerate(zip(self.policies, [self.num_guard_envs, self.num_scout_envs])):
            policy.policy.set_training_mode(False)
            policy.n_steps = 0
            policy.rollout_buffer.reset()

            if policy.use_sde:
                policy.policy.reset_noise(num_envs)  # type: ignore[operator]

            [callback.on_rollout_start() for callback in callbacks]
            policy._last_episode_starts = np.ones((num_envs,), dtype=bool)
            all_last_episode_starts[polid] = policy._last_episode_starts
        # print('------------PRE ROLLOUT TIME-------------')
        # print(time.time() - start)


        # do rollout
        while n_steps < n_rollout_steps:
            # print('n_steps', n_steps)
            with torch.no_grad():
                # start = time.time()
                for polid, policy in enumerate(self.policies):
                    episode_starts = torch.tensor(policy._last_episode_starts, dtype=torch.float32, device=policy.device)
                    (
                        all_actions[polid],
                        all_values[polid],
                        all_log_probs[polid],
                    ) = policy.policy.forward(last_obs[polid])
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

            for polid, policy_agent_index in enumerate(policy_agent_indexes):
                step_actions[policy_agent_index] = all_clipped_actions[polid]

            # start = time.time()
            # actually step in the environment
            obs, rewards, dones, infos = self.env.step(step_actions)
            # print('rewards', rewards)
            policy_agent_indexes = self.get_policy_agent_indexes_from_viewcone(last_obs=obs)
            # print('policy_agent_indexes', policy_agent_indexes)
            all_curr_obs = self.format_env_returns(obs, policy_agent_indexes=policy_agent_indexes, to_tensor=True, device=self.policies[0].device)
            all_curr_obs_buffer = self.format_env_returns(obs, policy_agent_indexes=policy_agent_indexes, to_tensor=False)
            all_rewards = self.format_env_returns(rewards, policy_agent_indexes=policy_agent_indexes, to_tensor=False)
            all_dones = self.format_env_returns(dones, policy_agent_indexes=policy_agent_indexes, to_tensor=False)
            all_infos = self.format_env_returns(infos, policy_agent_indexes=policy_agent_indexes, to_tensor=False)
            # print('all_curr_obs', all_curr_obs)
            # print('all_rewards', all_rewards)
            # print('------------STEP FORWARD TIME-------------')

            for policy in self.policies:
                policy.num_timesteps += self.num_vec_envs

            for callback in callbacks:
                callback.update_locals(locals())
            
            # must-have for checkpointing
            [callback.on_step() for callback in callbacks]
            # specific callback for eval
            if eval_callback is not None:
                eval_callback.on_step()


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


        for callback in callbacks:
            callback.on_rollout_end()

        return total_rewards, n_steps

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


    def format_env_returns(
        self,
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
        Returns:
            to_agents: list[np.ndarray] | list[dict[str, np.ndarray]], where first dimension of array is (num_envs.) and list is of length (num_agents).
        
        Right now, very hardcoded to two agents
        """
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
                for polid in range(self.num_policies)
            ]
        elif isinstance(env_returns, np.ndarray):
            to_policies = [
                mutate_func(np.take(env_returns, policy_agent_indexes[polid], axis=0), **kwargs) for polid in range(self.num_policies)
            ]
        elif isinstance(env_returns, list):
            # for now only 'info' fits in here. dont need to mutate?
            to_policies = [
                list(itemgetter( *(policy_agent_indexes[polid].tolist()) )(env_returns))
                if len(policy_agent_indexes[polid]) > 1 else [itemgetter( *(policy_agent_indexes[polid].tolist()) )(env_returns)]
                for polid in range(self.num_policies)
            ]
            # for obs_list in to_policies:  # i love triple nested 4 loops because im too lazy to optimize
            #     for d in obs_list:
            #         for k, v in d.items():
            #             d[k] = mutate_func(v, **kwargs)
        else:
            raise AssertionError(f'Assertion failed. format_env_returns recieved unexpected type {type(env_returns)}. \
                Expected dict[str, np.ndarray] or np.ndarray or list.')

        return to_policies


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

    def get_policy_agent_indexes_from_viewcone(self, last_obs):
        # very hardcoded solution for now.
        # if last_obs is 2 dim, the 1st is likely the env-agent dimension, and the latter is the frame-stacked * original version
        # elif last_obs is 4dim, the 1st is as above, second is the frame-stack dim, and the third is 7, 5 (viewcone dim)
        # TODO: ALL HARDCODED! PLEASE CHANGE THIS (i definitely wont)
        """
        """
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
    