import time
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Type, Union

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
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        super()._convert_train_freq()

        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_env = DummyVecEnv([env_fn] * self.num_envs)
        # this is a wrapper class, so it will not hold any states like
        # buffer_size, or action_noise. pass those straight to DQN.
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
        callbacks: Optional[List[List[MaybeCallback]]] = None,
        log_interval: int = 1,
        tb_log_name: str = "IndependentDQN",
        reset_num_timesteps: bool = True,
    ):
        """
        Main learn function. Mainly goes as follows:

        target: collect rollbacks and train up till total_timesteps.
        """

        if callbacks is not None:
            assert len(callbacks) == self.num_agents, 'callbacks must a list of num_agents number of nested lists'
            assert all(isinstance(callback, list) for callback in callbacks), 'callbacks must a list of num_agents number of nested lists'

        num_timesteps = 0
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

        for agent in range(self.num_agents):  #
            for envid in range(self.num_envs):
                last_obs[envid * self.num_agents + agent] = {
                    k: v[envid * self.num_agents + agent, ...] for k, v in reset_obs.items()
                }
                print('--------NUMBER',     envid * self.num_agents + agent)
                print(last_obs[envid * self.num_agents + agent])

        for agent in self.agents:
            agent._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

        while num_timesteps < total_timesteps:
            # environment sampling. has to be done in this particular way because of
            # gridworld's perculiarities
            last_obs = self.collect_rollouts(
                last_obs=last_obs,
                callbacks=callbacks,
                train_freq=self.train_freq,
                learning_starts=self.learning_starts,
            )
            num_timesteps += self.num_envs * self.num_timesteps

            # agent training.
            for agent_id, agent in enumerate(self.agents):
                agent._update_current_progress_remaining(
                    agent.num_timesteps, total_timesteps
                )
                if log_interval is not None and num_timesteps % log_interval == 0:
                    fps = int(agent.num_timesteps / (time.time() - agent.start_time))
                    agent.logger.record("agent_id", agent_id, exclude="tensorboard")
                    agent.logger.record(
                        "time/iterations", num_timesteps, exclude="tensorboard"
                    )
                    if (
                        len(agent.ep_info_buffer) > 0
                        and len(agent.ep_info_buffer[0]) > 0
                    ):
                        agent.logger.record(
                            "rollout/ep_rew_mean",
                            safe_mean(
                                [ep_info["r"] for ep_info in agent.ep_info_buffer]
                            ),
                        )
                        agent.logger.record(
                            "rollout/ep_len_mean",
                            safe_mean(
                                [ep_info["l"] for ep_info in agent.ep_info_buffer]
                            ),
                        )
                    agent.logger.record("time/fps", fps)
                    agent.logger.record(
                        "time/time_elapsed",
                        int(time.time() - agent.start_time),
                        exclude="tensorboard",
                    )
                    agent.logger.record(
                        "time/total_timesteps",
                        agent.num_timesteps,
                        exclude="tensorboard",
                    )
                    agent.logger.dump(step=agent.num_timesteps)

                agent.train()

        for callback in callbacks:
            callback.on_training_end()

    def collect_rollouts(
            self,
            last_obs,
            callbacks,
            train_freq,
            learning_starts: int = 0,
        ):
        """
        Helper function to collect rollouts (sample the env and feed observations into the agents, vice versa)
        This function will be fed by an exiting last_observation; that is the one generated by self.env.reset
        
        """
        # temporary lists to hold things before saved into history_buffer of agent.
        all_last_episode_starts = [None] * self.num_agents
        all_clipped_actions = [None] * self.num_agents
        all_obs = [None] * self.num_agents
        all_next_obs = [None] * self.num_agents
        all_rewards = [None] * self.num_agents
        all_dones = [None] * self.num_agents
        all_infos = [None] * self.num_agents
        
        num_collected_steps, num_collected_episodes = 0, 0

        # iterate over agents, and do pre-rollout setups.
        for agent_id, agent in enumerate(self.agents):
            for envid in range(self.num_envs):
                assert (
                    last_obs[envid * self.num_agents + agent_id] is not None
                ), f"No previous observation was provided for env_{envid}_policy_{agent_id}"
            # tensorized
            all_obs[agent_id] = np.array(
                [
                    {k: obs_as_tensor(v, agent.device) 
                        for k, v in last_obs[envid * self.num_agents + agent_id].items()}
                    for envid in range(self.num_envs)
                ]
            )
            print('all_obs[agent_id]', all_obs[agent_id])
            agent.policy.set_training_mode(False)

            assert train_freq.frequency > 0, "Should at least collect one step or episode."

            if self.num_envs > 1:
                assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

            if agent.use_sde:
                agent.actor.reset_noise(self.num_envs)  # type: ignore[operator]

            print('callbacks', callbacks)
            [callback.on_rollout_start() for callback in callbacks]
            all_last_episode_starts[agent_id] = agent._last_episode_starts

        # loop over action -> step -> observation -> agents -> action ... loop
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            all_actions = [None] * self.num_agents
            with th.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    # for k, v in all_obs[agent_id]:
                    #     all_obs[agent_id][k] =
                    # obs_tensor = obs_as_tensor(all_obs[agent_id], agent.device)
                    # sample random action or according to policy.
                    actions, buffer_actions = self._sample_action(
                        agent=agent,
                        obs=all_obs[agent_id],
                        learning_starts=learning_starts,
                        n_envs=self.num_envs,
                    )
                    all_actions[agent_id] = actions
                    if hasattr(all_actions[agent_id], 'cpu'):
                        all_actions[agent_id] = all_actions[agent_id].cpu()
                    clipped_actions = all_actions[agent_id]
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

            # actually step in the environment
            obs, rewards, dones, infos = self.env.step(all_clipped_actions)

            # append observations, rewards and others into `all_` lists.
            for agent_id in range(self.num_agents):
                all_obs[agent_id] = np.array(
                    [
                        {k: vobs[envid * self.num_agents + agent_id].items()}
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

            self.num_timesteps += self.num_envs
            num_collected_steps += 1

            # add data to the replay buffers
            for agent_id, agent in enumerate(self.agents):
                if isinstance(self.action_space, Discrete):
                    all_actions[agent_id] = all_actions[agent_id].reshape(-1, 1)
                all_actions[agent_id] = all_actions[agent_id].cpu().numpy()
                agent.replay_buffer.add(
                    obs=all_obs[agent_id],
                    next_obs=all_next_obs[agent_id],
                    action=all_actions[agent_id],
                    reward=all_rewards[agent_id],
                    done=all_dones[agent_id],
                    infos=all_infos[agent_id],
                )
            all_obs = all_next_obs
            all_last_episode_starts = all_dones

        # special dqn thing.
        for agent_id, agent in enumerate(self.agents):
            if hasattr(agent, '_on_step'):
                agent._on_step()

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
    ) -> "IndependentDQN":
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
            model.policies[agent_id] = DQN.load(
                path=path + f"/policy_{agent_id + 1}/model", env=dummy_env, **kwargs
            )
        return model

    def save(self, path: str) -> None:
        for agent_id in range(self.num_agents):
            self.agents[agent_id].save(path=path + f"/policy_{agent_id + 1}/model")

    def _sample_action(
        self,
        agent,
        obs,
        learning_starts: int,
        n_envs: int = 1,
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
        if agent.num_timesteps < learning_starts and not (agent.use_sde and agent.use_sde_at_warmup):
            # Warmup phase
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