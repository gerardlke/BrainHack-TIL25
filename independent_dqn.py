import time
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
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

from stable_baselines3.common.buffers import ReplayBuffer
from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv

class DummyGymEnv(gym.Env):
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

    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: ParallelEnv | MarkovVectorEnv,
        learning_rate: Union[float, Schedule] = 1e-4,
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
        self.num_envs = env.num_envs // num_agents
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        self.agents = [
            DQN(
                policy=policy,
                env=dummy_env,
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
            for _ in range(self.num_agents)
        ]

    def learn(
        self,
        total_timesteps: int,
        callbacks: Optional[List[MaybeCallback]] = None,
        log_interval: int = 1,
        tb_log_name: str = "IndependentDQN",
        reset_num_timesteps: bool = True,
    ):
        base_callbacks = [
            'something'
        ] * self.num_agents
        num_timesteps = 0
        all_total_timesteps = []
        if not callbacks:
            callbacks = [None] * self.num_agents
        self._logger = configure_logger(
            self.verbose,
            self.tensorboard_log,
            tb_log_name,
            reset_num_timesteps,
        )
        logdir = self.logger.dir

        # Setup for each policy. Reset things, setup timestep tracking things
        for agent_id, agent in enumerate(self.agents):
            agent.start_time = time.time()
            if agent.ep_info_buffer is None or reset_num_timesteps:
                agent.ep_info_buffer = deque(maxlen=100)
                agent.ep_success_buffer = deque(maxlen=100)

            if agent.action_noise is not None:
                agent.action_noise.reset()

            if reset_num_timesteps:
                agent.num_timesteps = 0
                agent._episode_num = 0
                all_total_timesteps.append(total_timesteps)
                agent._total_timesteps = total_timesteps
            else:
                # make sure training timestamps are ahead of internal counter
                all_total_timesteps.append(total_timesteps + agent.num_timesteps)
                agent._total_timesteps = total_timesteps + agent.num_timesteps

            agent._logger = configure_logger(
                agent.verbose,
                logdir,
                "policy",
                reset_num_timesteps,
            )

            callbacks[agent_id] = agent._init_callback(callbacks[agent_id])

        # Call all callbacks back to call all backs, callbacks
        for callback in callbacks:
            callback.on_training_start(locals(), globals())

        # self.env returns a dict, where each key is (M * N, ...), M is number of envs, N is number of agents.
        # we determine number of envs based on the output shape (should find a better way to do this)
        reset_obs = self.env.reset()
        last_obs = defaultdict(dict)
        num_envs = int(list(reset_obs.values())[0].shape[0] / self.num_agents)
        for agent in range(self.num_agents):  #
            start, end = agent * num_envs, agent * (num_envs + 1)
            last_obs[agent] = {
                k: v[start:end, ...] for k, v in reset_obs.items()
            }

        for agent in self.agents:
            agent._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

        while num_timesteps < total_timesteps:
            # environment sampling. has to be done in this particular way because of
            # gridworld's perculiarities
            last_obs = self.collect_rollouts(last_obs, callbacks)
            num_timesteps += self.num_envs * self.n_steps

            # policy training.
            for agent_id, policy in enumerate(self.agents):
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps
                )
                if log_interval is not None and num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    policy.logger.record("policy_id", agent_id, exclude="tensorboard")
                    policy.logger.record(
                        "time/iterations", num_timesteps, exclude="tensorboard"
                    )
                    if (
                        len(policy.ep_info_buffer) > 0
                        and len(policy.ep_info_buffer[0]) > 0
                    ):
                        policy.logger.record(
                            "rollout/ep_rew_mean",
                            safe_mean(
                                [ep_info["r"] for ep_info in policy.ep_info_buffer]
                            ),
                        )
                        policy.logger.record(
                            "rollout/ep_len_mean",
                            safe_mean(
                                [ep_info["l"] for ep_info in policy.ep_info_buffer]
                            ),
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

                policy.train()

        for callback in callbacks:
            callback.on_training_end()

    def collect_rollouts(self, last_obs, callbacks):
        """
        Helper function to collect rollouts (sample the env)
        """

        # temporary lists to hold things before saved into history_buffer of agent.
        all_last_episode_starts = [None] * self.num_agents
        all_obs = [None] * self.num_agents
        all_last_obs = [None] * self.num_agents
        all_rewards = [None] * self.num_agents
        all_dones = [None] * self.num_agents
        all_infos = [None] * self.num_agents
        steps = 0

        for agent_id, agent in enumerate(self.agents):
            for envid in range(self.num_envs):
                assert (
                    last_obs[envid * self.num_agents + agent_id] is not None
                ), f"No previous observation was provided for env_{envid}_policy_{agent_id}"
            all_last_obs[agent_id] = np.array(
                [
                    last_obs[envid * self.num_agents + agent_id]
                    for envid in range(self.num_envs)
                ]
            )
            agent.policy.set_training_mode(False)
            callbacks[agent_id].on_rollout_start()
            all_last_episode_starts[agent_id] = agent._last_episode_starts

        while steps < self.n_steps:
            all_actions = [None] * self.num_agents
            all_values = [None] * self.num_agents
            all_log_probs = [None] * self.num_agents
            all_clipped_actions = [None] * self.num_agents
            with th.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    obs_tensor = obs_as_tensor(all_last_obs[agent_id], agent.device)
                    (
                        all_actions[agent_id],
                        all_values[agent_id],
                        all_log_probs[agent_id],
                    ) = agent.policy.forward(obs_tensor)
                    clipped_actions = all_actions[agent_id].cpu().numpy()
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
                    all_clipped_actions[agent_id] = clipped_actions

            all_clipped_actions = (
                np.vstack(all_clipped_actions).transpose().reshape(-1)
            )  # reshape as (env, action)

            # actually step
            obs, rewards, dones, infos = self.env.step(all_clipped_actions)

            for agent_id in range(self.num_agents):
                all_obs[agent_id] = np.array(
                    [
                        obs[envid * self.num_agents + agent_id]
                        for envid in range(self.num_envs)
                    ]
                )
                all_rewards[agent_id] = np.array(
                    [
                        rewards[envid * self.num_agents + agent_id]
                        for envid in range(self.num_envs)
                    ]
                )
                all_dones[agent_id] = np.array(
                    [
                        dones[envid * self.num_agents + agent_id]
                        for envid in range(self.num_envs)
                    ]
                )
                all_infos[agent_id] = np.array(
                    [
                        infos[envid * self.num_agents + agent_id]
                        for envid in range(self.num_envs)
                    ]
                )

            for policy in self.agents:
                policy.num_timesteps += self.num_envs

            for callback in callbacks:
                callback.update_locals(locals())
            if not [callback.on_step() for callback in callbacks]:
                break

            for agent_id, policy in enumerate(self.agents):
                policy._update_info_buffer(all_infos[agent_id])

            steps += 1

            # add data to the rollout buffers
            for agent_id, agent in enumerate(self.agents):
                if isinstance(self.action_space, Discrete):
                    all_actions[agent_id] = all_actions[agent_id].reshape(-1, 1)
                all_actions[agent_id] = all_actions[agent_id].cpu().numpy()
                agent.replay_buffer.add(
                    all_last_obs[agent_id],
                    all_actions[agent_id],
                    all_rewards[agent_id],
                    all_last_episode_starts[agent_id],
                    all_values[agent_id],
                    all_log_probs[agent_id],
                )
            all_last_obs = all_obs
            all_last_episode_starts = all_dones

        with th.no_grad():
            for agent_id, policy in enumerate(self.agents):
                obs_tensor = obs_as_tensor(all_last_obs[agent_id], policy.device)
                _, value, _ = policy.policy.forward(obs_tensor)
                print('')
                # policy.rollout_buffer.compute_returns_and_advantage(
                #     last_values=value, dones=all_dones[agent_id]
                # )

        for callback in callbacks:
            callback.on_rollout_end()

        for agent_id, policy in enumerate(self.agents):
            policy._last_episode_starts = all_last_episode_starts[agent_id]

        return obs

    @classmethod
    def load(
        cls,
        path: str,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: GymEnv,
        n_steps: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ) -> "IndependentPPO":
        model = cls(
            policy=policy,
            num_agents=num_agents,
            env=env,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs,
        )
        env_fn = lambda: DummyGymEnv(env.observation_space, env.action_space)
        dummy_env = DummyVecEnv([env_fn] * (env.num_envs // num_agents))
        for agent_id in range(num_agents):
            model.policies[agent_id] = PPO.load(
                path=path + f"/policy_{agent_id + 1}/model", env=dummy_env, **kwargs
            )
        return model

    def save(self, path: str) -> None:
        for agent_id in range(self.num_agents):
            self.agents[agent_id].save(path=path + f"/policy_{agent_id + 1}/model")