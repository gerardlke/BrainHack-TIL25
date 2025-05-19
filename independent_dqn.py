import time
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Type, Union
from copy import deepcopy
from operator import itemgetter
import gymnasium
from gymnasium import spaces
import numpy as np
import torch as th
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.common.utils import (configure_logger, obs_as_tensor,
                                            safe_mean)
from stable_baselines3.common.vec_env import DummyVecEnv
from pettingzoo import ParallelEnv
from stable_baselines3.common.type_aliases import TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.buffers import ReplayBuffer
from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from einops import rearrange

class DummyGymEnv(gymnasium.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class IndependentDQN(OffPolicyAlgorithm):
    """
    Acts as a wrapper of Multi-agent implementation of DQN, specific to gridworld's case.
    Gridworld has an unintuitive environment mechanism, hence the yapping here.

    Expects the environment to be a ParallelEnv to simplify observation splitting and action
    concatenation. MarkovVectorEnv also works fine lol, we just split whatever the output is
    by the number of agents, so there can be an arbitrary number of envs.

    To be clear, gridworld's step flow follows as such.
    For n agents, during the first n-1 calls to step(), actions are cached.
    During the nth call to step(), all actions are executed in the environment.
    Then, all agents observe the environment, and these observations are cached.

    These observations are retrieved by the environment's `last` method, where each call
    changes the agent whose observations are being returned. This is why (unintuitively)
    you will see many `last` calls.

    Inherits from OffPolicyAlgorithm, but changes who owns and does what.
    Anything related to the scheduling of the iterations, steps, episodes, etc. are handled
    in this class. Anything else, will be defaulted onto agent attributes / agent handling.
    hence why you see many for loops (bad programming but oh well), and agent.`this_and_that`
    everywhere

    NOTE: Discovering how the internal player indexes of the environment change, and how that affects the index of the reward
    that is passed to us, we will take the approach of training only two models; scout and guard. Because in the field there are 3 other guards,
    we can stack the other two's observations along the batch dimension during inference, result in one comprehensive guard model (that isn't orchestrating
    all three of them, mind you) and one scout model.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: ParallelEnv | MarkovVectorEnv,
        learning_rate: Union[float, Schedule] = 1e-4,
        num_policies: int = 2,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.env = env
        self.num_agents = num_agents
        self.num_policies = num_policies
        assert num_agents == 4
        assert num_policies == 2


        self.num_envs = env.num_envs // num_agents
        self.num_scout_envs = self.num_envs * 1  # i love hardcoding!
        self.num_guard_envs = self.num_envs * 3
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print('self.obs space', self.observation_space)
        print('self.action_space', self.action_space)
        print('self.num_scout_envs, self.num_guard_envs', self.num_scout_envs, self.num_guard_envs)
        self._vec_normalize_env = None
        print('self._vec_normalize_env', self._vec_normalize_env)
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        super()._convert_train_freq()

        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        self.dummy_envs = [DummyVecEnv([env_fn] * self.num_guard_envs), DummyVecEnv([env_fn] * self.num_scout_envs)]
        # this is a wrapper class, so it will not hold any states like
        # buffer_size, or action_noise. pass those straight to DQN.

        # we keep self.agents as a list, where the first is always the scout and the second is always the guard model.
        # for now do not 
        self.policies = [
            DQN(
                policy=policy,
                env=self.dummy_envs[i],
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs,
                optimize_memory_usage=optimize_memory_usage,
                target_update_interval=target_update_interval,
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                max_grad_norm=max_grad_norm,
                stats_window_size=stats_window_size,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                seed=seed,
                device="auto",
                _init_setup_model=True,
            )
            for i in range(self.num_policies)
        ]
        print('do policies?', self.policies[0]._vec_normalize_env)

        assert self.num_policies == 2, 'Right now only supports 2 num_policies; 1 scout and 1 unified guard'

    def learn(
        self,
        total_timesteps: int,
        policy_grad_steps: int,
        callbacks: Optional[List[List[MaybeCallback]]] = None,
        eval_callback: Optional[MaybeCallback] = None,
        log_interval: int = 1,
        tb_log_name: str = "IndependentDQN",
        reset_num_timesteps: bool = True,
    ):
        """
        Main learn function. Mainly goes as follows:

        target: collect rollbacks and train up till total_timesteps.
        """
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
                policy.ep_info_buffer = deque(maxlen=100)
                policy.ep_success_buffer = deque(maxlen=100)

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

        for policy in self.policies:
            policy._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

        while self.num_timesteps < total_timesteps:
            # environment sampling. has to be done in this particular way because of
            # gridworld's perculiarities
            total_rewards, rollout_timesteps = self.collect_rollouts(
                last_obs=reset_obs,
                callbacks=callbacks,
                eval_callback=eval_callback,
                train_freq=self.train_freq,
                learning_starts=self.learning_starts,
            )
            self.num_timesteps += rollout_timesteps

            # agent training.
            for polid, policy in enumerate(self.policies):
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps  # 
                )
                if log_interval is not None and num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    policy.logger.record("polid", polid, exclude="tensorboard")
                    policy.logger.record(
                        "time/iterations", num_timesteps, exclude="tensorboard"
                    )
                    # print(f'np.concatenate(total_rewards[{polid}])')
                    # print(
                    #       np.unique(np.concatenate(total_rewards[polid]))
                    # )
                    mean_policy_reward = (np.sum(np.concatenate(total_rewards[polid])) / len(total_rewards[polid])).item()
                    policy.logger.record(
                        "rollout/mean_policy_reward", mean_policy_reward,
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

                policy.train(policy_grad_steps, policy.batch_size)

        for callback in callbacks:
            callback.on_training_end()

    def collect_rollouts(
            self,
            last_obs,
            train_freq,
            learning_starts: int = 0,
            eval_callback: Optional[MaybeCallback] = None,
            callbacks: list = [],
            run_on_step: bool = True,
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
        total_rewards = [[] for _ in range(self.num_policies)] 
        all_dones = [None] * self.num_policies
        all_infos = [None] * self.num_policies
        all_actions = [None] * self.num_policies
        all_buffer_actions = [None] * self.num_policies
        placeholder_actions = np.empty(self.num_agents * self.num_envs, dtype=np.int64)
        num_collected_steps, num_collected_episodes = 0, 0

        # current way of inferring which is the scout based off viewcone
        policy_agent_indexes = self.get_policy_agent_indexes_from_viewcone(last_obs=last_obs)
        
        # this was for the time i was feeding in dictionary inputs, which is not supported by supersuit's frame stacking environment
        # policy_agent_indexes = self.get_policy_agent_indexes_from_dict_obs(last_obs=last_obs)

        # before formatted, last_obs is the direct return of self.env.reset()
        last_obs_buffer = self.format_env_returns(last_obs, policy_agent_indexes, device=self.policies[0].device, to_tensor=False)
        last_obs = self.format_env_returns(last_obs, policy_agent_indexes, device=self.policies[0].device)

        # iterate over agents, and do pre-rollout setups.
        for polid, policy in enumerate(self.policies):
            policy.policy.set_training_mode(False)

            assert train_freq.frequency > 0, "Should at least collect one step or episode."

            if self.num_envs > 1:
                assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

            if policy.use_sde:
                policy.actor.reset_noise(self.num_envs)  # type: ignore[operator]

            [callback.on_rollout_start() for callback in callbacks]
            all_last_episode_starts[polid] = policy._last_episode_starts

        # loop over action -> step -> observation -> policy -> action ... loop
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            with th.no_grad():
                for polid, (policy, n_envs) in enumerate(zip(self.policies, [self.num_guard_envs, self.num_scout_envs])):
                    # sample random action or according to policy.
                    actions, buffer_actions = self._sample_action(
                        agent=policy,
                        obs=last_obs[polid],
                        learning_starts=learning_starts,
                        n_envs=n_envs,
                        random=not eval,
                    )
                    all_actions[polid] = actions
                    all_buffer_actions[polid] = buffer_actions
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

            for polid, policy_agent_index in enumerate(policy_agent_indexes):
                placeholder_actions[policy_agent_index] = all_clipped_actions[polid]

            # actually step in the environment
            obs, rewards, dones, infos = self.env.step(placeholder_actions)   

            # policy_agent_indexes = self.get_policy_agent_indexes_from_dict_obs(last_obs=obs)
            policy_agent_indexes = self.get_policy_agent_indexes_from_viewcone(last_obs=obs)

            all_curr_obs = self.format_env_returns(obs, policy_agent_indexes, device=self.policies[0].device)
            all_curr_obs_buffer = self.format_env_returns(obs, policy_agent_indexes, device=self.policies[0].device, to_tensor=False)
            all_rewards = self.format_env_returns(rewards, policy_agent_indexes, device=self.policies[0].device, to_tensor=False)
            all_dones = self.format_env_returns(dones, policy_agent_indexes, device=self.policies[0].device, to_tensor=False)
            all_infos = self.format_env_returns(infos, policy_agent_indexes, device=self.policies[0].device, to_tensor=False)
            # if any(np.any(arr) for arr in all_rewards[0]):
            #     print('obs', obs)
            #     print('rewards', rewards)
            #     print('dones', dones)
            #     print('infos', infos)
            #     print('policy_agent_indexes', policy_agent_indexes)
            #     print('all_curr_obs', all_curr_obs)
            #     print('all_rewards', all_rewards)
            #     print('all_dones', all_dones)
            #     print('all_infos', all_infos)
                # raise AssertionError('shi don fucked up')
            # if print_next_after_dones:
            #     print('----------------NEXT AFTER DONES-----------------')
            #     print_next_after_dones = False
            #     print_next_next_after_dones = True
            #     print('policy_agent_indexes', policy_agent_indexes)
            #     print('all_curr_obs', all_curr_obs)
            #     print('rewards', rewards)
            #     print('all_rewards', all_rewards)
            #     print('all_dones', all_dones)
            #     print('all_infos', all_infos)
            # if any((1 in arr) for arr in all_dones):
            #     print('obs', obs)
            #     print('before policy_agent_indexes', policy_agent_indexes)
            #     print_next_after_dones = True
            #     print('---------------------DONES--------------------')
            # if print_next_next_after_dones:
            #     print_next_next_after_dones = False
            #     print('----------------NEXT NEXT AFTER DONES----------------')
            #     print('policy_agent_indexes', policy_agent_indexes)
            #     print('all_curr_obs', all_curr_obs)
            #     print('rewards', rewards)
            #     print('all_rewards', all_rewards)
            #     print('all_dones', all_dones)
            #     print('all_infos', all_infos)

            for policy in self.policies:
                policy.num_timesteps += self.num_envs
            

            for callback in callbacks:
                callback.update_locals(locals())
            
            # must-have for checkpointing
            [callback.on_step() for callback in callbacks]
            # specific callback for eval
            if eval_callback is not None:
                eval_callback.on_step()

            for polid, policy in enumerate(self.policies):
                policy._update_info_buffer(all_infos[polid])

            num_collected_steps += self.num_envs

            # add data to the replay buffers

            for polid, policy in enumerate(self.policies):
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                if hasattr(all_actions[polid], 'cpu'):
                    all_actions[polid] = all_actions[polid].cpu().numpy()
                # all_obs[polid] is a list[dict[str, np.ndarray]], but add only wants per dict. hence this loop
                policy.replay_buffer.add(
                    obs=deepcopy(last_obs_buffer[polid]),
                    next_obs=deepcopy(all_curr_obs_buffer[polid]),
                    action=deepcopy(all_buffer_actions[polid]),
                    reward=deepcopy(all_rewards[polid]),
                    done=deepcopy(all_dones[polid]),
                    infos=deepcopy(all_infos[polid]),
                )
                total_rewards[polid].append(all_rewards[polid])

            last_obs = all_curr_obs
            all_last_episode_starts = all_dones
            # if any(np.any(done) for done in dones):
            #     last_obs = self.env.reset()
            #     last_obs = self.format_env_returns(last_obs, device=self.agents[0].device)

        # special dqn thing.
        if run_on_step:
            for polid, policy in enumerate(self.policies):
                if hasattr(policy, '_on_step'):
                    policy._on_step()

        for callback in callbacks:
            callback.on_rollout_end()

        for polid, policy in enumerate(self.policies):
            policy._last_episode_starts = all_last_episode_starts[polid]

        return total_rewards, num_collected_steps

    @classmethod
    def load(
        cls,
        path: str,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        num_policies: int,
        env: GymEnv,
        learning_starts: int,
        train_freq: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ) -> "IndependentDQN":
        model = cls(
            policy=policy,
            num_policies=num_policies,
            num_agents=num_agents,
            env=env,
            learning_starts=learning_starts,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            train_freq=train_freq,
            **kwargs,
        )
        for polid in range(model.num_policies):
            model.policies[polid] = DQN.load(
                policy=policy,
                path=path + f"/policy_{polid}/model",
                env=model.dummy_envs[polid],
                learning_starts=learning_starts,
                train_freq=train_freq,
                print_system_info=True,
                **kwargs
            )
        return model

    def save(self, path: str) -> None:
        for polid in range(self.num_policies):
            self.policies[polid].save(path=path + f"/policy_{polid}/model")

    def _sample_action(
        self,
        agent,
        obs,
        learning_starts: int,
        n_envs: int = 1,
        random = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if random and (agent.num_timesteps < learning_starts and not (agent.use_sde and agent.use_sde_at_warmup)):
            # Warmup phase
            print('RANDOM SAMPLING HAPPENING, AGENT PREDICT IS NOT BEING CALLED')
            unscaled_action = np.array([agent.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = agent.predict(obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(agent.action_space, spaces.Box):
            scaled_action = agent.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if agent.action_noise is not None:
                scaled_action = np.clip(scaled_action + agent.action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = agent.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action

        return action, buffer_action

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

    def get_policy_agent_indexes_from_dict_obs(self, last_obs):
        """
        Helper func to decode which indexes of observations of shape (self.num_envs * self.num_agents, ...)
        map to which policies.
        """
        policy_mapping = last_obs['scout']
        policy_agent_indexes = [None] * self.num_policies
        for polid in range(self.num_policies):
            policy_agent_indexes[polid] = np.where(policy_mapping == polid)[0]

        return policy_agent_indexes

    def get_policy_agent_indexes_from_viewcone(self, last_obs):
        bits = np.unpackbits(last_obs[:, :, 2, 2][np.newaxis, :, :].astype(np.uint8), axis=0)
        is_scout = bits[5, :, 0]
        policy_agent_indexes = [
            np.where(is_scout == 0)[0], np.where(is_scout == 1)[0]
        ]

        return policy_agent_indexes

    
    def eval(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Main eval function. Mainly acts as a wrapper around self.collect_rollouts
        """
        print(f'NOTE: TOTAL TIMESTEPS {total_timesteps} INCLUDES NUMBER OF ENVIRONMENTS (currently {self.num_envs}), AND IS NOT A PER-ENV BASIS')

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
        n_rollout_steps = total_timesteps * self.num_envs
        total_rewards, rollout_timesteps = self.collect_rollouts(
            last_obs=reset_obs,
            callbacks=[],
            train_freq=self.train_freq,
            learning_starts=self.learning_starts,
            eval=True,
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