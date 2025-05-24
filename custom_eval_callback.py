import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from einops import rearrange
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from collections import defaultdict
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


class CustomEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        in_bits,
        agent_roles: list,
        policy_mapping: list,
        evaluate_policy: str,
        eval_env: Union[gym.Env, VecEnv],
        eval_env_config,
        training_config,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):  
        """
        init.
        Crucial args:
            - eval_env_config: Must-have the configuration state of eval,
            - eval_env: the actual environment to do evaluation
            - evaluate_policy: the evaluate_policy to proceed with (multi-policy or single-policy) (we will do checks on this)
            - agent_roles: the roles of each agent, e.g [1, 0, 0, 0]
            - policy_mapping: the policy each agent maps to: e.g [1, 0, 0, 0]
        """
        super().__init__(callback_after_eval, verbose=verbose)
        self.in_bits = in_bits
        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.training_config = training_config
        self.n_eval_episodes = self.training_config.n_eval_episodes
        self.eval_freq = self.training_config.eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.root_log_path = log_path
        # horrible triple nested list, but for the sake of keeping to original code flow
        # while injecting multiple policies.
        # first list elements are results for seperate policies.
        # next list elements are results across seperate runs
        # next list elements are results across each agent-env
        self.evaluations_results: dict[int, list[list[float]]] = defaultdict(list)
        self.evaluations_timesteps: dict[int, list[int]] = defaultdict(list)
        self.evaluations_length: dict[int, list[list[int]]] = defaultdict(list)
        # For computing success rate
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []

        # m-a-m-p: multi agent multi policy
        # m-a-s-p: multi agent single policy
        assert evaluate_policy in ['mp', 'sp'], 'Assertion failed. Unsupported evaluate_policy.'
        if evaluate_policy == 'mp':
            self.evaluate_policy = self.custom_marl_evaluate_policy
        else:
            self.evaluate_policy = self.seperate_agent_evaluate

        vec_role_indexes = np.array(agent_roles * eval_env_config.num_vec_envs)
        self.role_indexes = [
            np.where(vec_role_indexes == polid)[0] for polid in np.unique(vec_role_indexes)
        ]

        vec_policy_mapping = np.array(policy_mapping * eval_env_config.num_vec_envs)
        self.policy_agent_indexes = [
            np.where(vec_policy_mapping == polid)[0] for polid in np.unique(vec_policy_mapping)
        ]


    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.root_log_path is not None:
            self.num_policies = len(self.model.policies)
            for i in range(self.num_policies):
                os.makedirs(os.path.dirname(os.path.join(self.root_log_path, f'polid_{i}')), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            policy_episode_rewards, roles_episode_rewards, policy_episode_lengths = self.evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            print('roles_episode_rewards', roles_episode_rewards)
            print('policy_episode_rewards', policy_episode_rewards)
            print('policy_episode_lengths', policy_episode_lengths)

            if self.root_log_path is not None:
                assert isinstance(policy_episode_rewards, list)
                assert isinstance(policy_episode_lengths, list)
                for polid in range(len(policy_episode_lengths)):
                    self.evaluations_timesteps[polid].append(self.num_timesteps)
                    self.evaluations_results[polid].append(policy_episode_rewards[polid])
                    self.evaluations_length[polid].append(policy_episode_lengths[polid])

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)
                
                for polid in self.evaluations_results:
                    np.savez(
                        os.path.join(self.root_log_path, f'polid_{polid}'),
                        timesteps=self.evaluations_timesteps[polid],
                        results=self.evaluations_results[polid],
                        ep_lengths=self.evaluations_length[polid],
                        **kwargs,  # type: ignore[arg-type]
                    )
            # mean them all
            for polid, policy_episode_reward in enumerate(policy_episode_rewards):
                policy_episode_rewards[polid] = np.mean(policy_episode_reward)
                policy_episode_lengths[polid] = np.mean(policy_episode_lengths[polid])

            for roid, role_episode_reward in enumerate(roles_episode_rewards):
                roles_episode_rewards[roid] = np.mean(role_episode_reward)
                
            # Add to current Logger
            [self.logger.record(f"eval/polid_{polid}_mean_reward", float(mean_reward)) for polid, mean_reward in enumerate(policy_episode_rewards)]
            [self.logger.record(f"eval/polid_{polid}_mean_lengths", float(mean_lengths)) for polid, mean_lengths in enumerate(policy_episode_lengths)]
            [self.logger.record(f"eval/role_{roid}_mean_reward", float(mean_reward)) for roid, mean_reward in enumerate(roles_episode_rewards)]
            
            # TODO: save each policy based on its own best reward, not jointly.
            mean_policy_reward = np.mean(policy_episode_rewards)
            self.last_mean_reward = float(mean_policy_reward)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_policy_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean policy reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_policy_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

    def custom_marl_evaluate_policy(
        self,
        simulator,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
    ) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
        """
        Multi-agent, multi policy case.
        Requires discernment of which agents map to which policies. (accessible in policy_agent_indexes attribute).
        """

        is_monitor_wrapped = False
        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor

        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

        if not is_monitor_wrapped and warn:
            warnings.warn(
                "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
                "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
                "Consider wrapping environment first with ``Monitor`` wrapper.",
                UserWarning,
            )

        # initialize evaluation session trackers.
        num_policies = len(self.policy_agent_indexes)
        num_roles = len(self.role_indexes)
        n_envs = env.num_envs  # num agents times number of vector envs
        episode_policy_rewards = [[] for _ in range(num_policies)] # nested list, per policy.
        episode_roles_rewards = [[] for _ in range(num_roles)]
        episode_lengths = [[] for _ in range(num_policies)] 
        all_clipped_actions = [None] * num_policies
        all_actions = [None] * num_policies

        step_actions = np.empty(n_envs, dtype=np.int64)
            
        episode_counts = np.zeros(n_envs, dtype="int")
        # n_eval_episodes are episodes per num_agents times num_envs
        episode_count_targets = np.array([n_eval_episodes for _ in range(n_envs)], dtype="int")
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        # per policy per env per agent, rewards are aggregated.
        # btw n_envs is already the num_vec_envs * num_agents, cuz supersuit and parallel env are nice like that
        # for each env that reaches a ended state, we the reward of that agent-env to episode_rewards.
        # vice versa for episode_lengths
        current_policy_rewards = [np.zeros(len(policy_agent_index)) 
                for policy_agent_index in self.policy_agent_indexes]
        current_roles_rewards = [np.zeros(len(role_indexes)) 
                for role_indexes in self.role_indexes]
        current_lengths = [np.zeros(len(policy_agent_index), dtype='int') 
                for policy_agent_index in self.policy_agent_indexes]

        # reset and format last observation
        observations = env.reset()
        last_obs = simulator.format_env_returns(
            observations,
            self.policy_agent_indexes,
            to_tensor=False,
            device=simulator.policies[0].device
        )

        states = None
        # predict each policy

        while (episode_counts < episode_count_targets).any():
            for polid, policy in enumerate(simulator.policies):
                actions, states = policy.predict(
                    last_obs[polid],  # type: ignore[arg-type]
                    state=states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
                all_actions[polid] = actions

                if hasattr(all_actions[polid], 'cpu'):
                    all_actions[polid] = all_actions[polid].cpu().numpy()
                clipped_actions = all_actions[polid]
                if isinstance(policy.action_space, Box):
                    clipped_actions = np.clip(
                        clipped_actions,
                        policy.action_space.low,
                        policy.action_space.high,
                    )
                elif isinstance(policy.action_space, Discrete):
                    # get integer from numpy array
                    clipped_actions = np.array(
                        [action.item() for action in clipped_actions]
                    )
                all_clipped_actions[polid] = clipped_actions

            for polid, policy_agent_index in enumerate(self.policy_agent_indexes):
                step_actions[policy_agent_index] = all_clipped_actions[polid]

            # we split by policy and by roles.
            # policywise-split all returns
            # however role-wise, split only the reward and dones.
            # this is because we arent running n-roles, but rather n-policies on the observations.
            new_observations, rewards, dones, infos = env.step(step_actions)
            all_curr_obs = simulator.format_env_returns(new_observations, self.policy_agent_indexes, device=simulator.policies[0].device, to_tensor=False)
            all_rewards = simulator.format_env_returns(rewards, self.policy_agent_indexes, device=simulator.policies[0].device, to_tensor=False)
            all_dones = simulator.format_env_returns(dones, self.policy_agent_indexes, device=simulator.policies[0].device, to_tensor=False)
            all_infos = simulator.format_env_returns(infos, self.policy_agent_indexes, device=simulator.policies[0].device, to_tensor=False)
            
            all_role_rewards = simulator.format_env_returns(rewards, self.role_indexes, device=simulator.policies[0].device, to_tensor=False)
            all_roles_dones = simulator.format_env_returns(dones, self.role_indexes, device=simulator.policies[0].device, to_tensor=False)
            
            # the following code is hideous. please, avert your eyes.

            for polid, (
                policy_agent_index,
                current_reward, # all n_env * num_agents under policy long e.g 6
                current_length, # all n_env * num_agents under policy long e.g 6
                episode_reward, # empty list 
                episode_length, # empty list 
                all_reward, # all n_env * num_agents under policy long e.g 6
                all_done, # all n_env * num_agents under policy long e.g 6
                all_info, # all n_env * num_agents under policy long e.g 6
            ) in enumerate(
                zip(
                    self.policy_agent_indexes,
                    current_policy_rewards,  
                    current_lengths,
                    episode_policy_rewards, 
                    episode_lengths,
                    all_rewards,
                    all_dones,
                    all_infos,
            )):
                # policy-based enumeration: each thing in the big tuple up there are of length n_envs
                # policy_agent_index is the env_indexes of envs * num_agents, that correspond to each policy.
                for enum, env_index in enumerate(policy_agent_index):
                    current_reward[enum] += all_reward[enum]
                    # print('all_reward[enum]', all_reward[enum])
                    # print('current_rewards', current_rewards)
                    current_length[enum] += 1

                    if episode_counts[env_index] < episode_count_targets[env_index]:
                        # unpack values so that the callback can access the local variables
                        done = all_done[enum]
                        info = all_info[enum]
                        episode_starts[env_index] = done

                        if callback is not None:
                            callback(locals(), globals())

                        if done:
                            if is_monitor_wrapped:
                                # Atari wrapper can send a "done" signal when
                                # the agent loses a life, but it does not correspond
                                # to the true end of episode
                                if "episode" in info.keys():
                                    # Do not trust "done" with episode endings.
                                    # Monitor wrapper includes "episode" key in info if environment
                                    # has been wrapped with it. Use those rewards instead.
                                    episode_reward.append(info["episode"]["r"])
                                    episode_length.append(info["episode"]["l"])
                                    # Only increment at the real end of an episode
                                    episode_counts[env_index] += 1
                            else:
                                episode_reward.append(current_reward[enum])
                                episode_length.append(current_length[enum])
                                episode_counts[env_index] += 1
                            current_reward[enum] = 0
                            current_length[enum] = 0

            for roid, (role_index,
                       current_role_rewards,
                       episode_role_rewards,
                       ) in enumerate(
                           zip(
                               self.role_indexes,
                               current_roles_rewards,
                               episode_roles_rewards,
                            )):
                print('role_index')
                for enum, env_index in enumerate(role_index):
                    current_role_rewards[enum] += all_role_rewards[enum]
                    if episode_counts[env_index] < episode_count_targets[env_index]:
                        done = all_roles_dones[enum]
                        if done:
                            episode_role_rewards.append(current_role_rewards[enum])
                            current_role_rewards[enum] = 0

            last_obs = all_curr_obs

            if render:
                env.render()

        # print('each episode_policy_rewards:')
        # print('lengths', [len(episode_reward) for episode_reward in episode_policy_rewards])
        # print('means', [np.mean(episode_reward) for episode_reward in episode_policy_rewards])
        mean_reward = [np.mean(episode_reward) for episode_reward in episode_policy_rewards]  # num_policies long
        std_reward = [np.std(episode_reward) for episode_reward in episode_policy_rewards]
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
        if return_episode_rewards:
            return episode_policy_rewards, episode_role_rewards, episode_lengths

        return mean_reward, std_reward


    def seperate_agent_evaluate(
        self,
        simulator,
        _env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
    ) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
        """
        Evaluates each policy seperately. This means that we make a distinction not between the policies the agents
        are operating under (because they are all the same), but rather the nature of each agent.

        We will still use policy_agent_indexes
        """
        is_monitor_wrapped = False
        observations = _env.reset()

        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor

        if not isinstance(_env, VecEnv):
            _env = DummyVecEnv([lambda: _env])  # type: ignore[list-item, return-value]

        is_monitor_wrapped = is_vecenv_wrapped(_env, VecMonitor) or _env.env_is_wrapped(Monitor)[0]

        if not is_monitor_wrapped and warn:
            warnings.warn(
                "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
                "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
                "Consider wrapping environment first with ``Monitor`` wrapper.",
                UserWarning,
            )

        n_envs = _env.num_envs
        episode_rewards = [[] for _ in range(self.num_policies)] # nested list, per policy.
        # each nested list is appended on a per-env basis
        episode_lengths = [[] for _ in range(self.num_policies)] 

        episode_counts = np.zeros(n_envs, dtype="int")
        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array([n_eval_episodes for _ in range(n_envs)], dtype="int")
        episode_starts = np.ones((_env.num_envs,), dtype=bool)

        scout_envs = int(_env.num_envs * len(policy_agent_indexes[1]))
        guard_envs = int(_env.num_envs * len(policy_agent_indexes[0]))

        current_rewards = [np.zeros(guard_envs), np.zeros(scout_envs)]
        current_lengths = [np.zeros(guard_envs, dtype="int"), np.zeros(scout_envs, dtype="int")]

        states = None
        episode_starts = np.ones((_env.num_envs,), dtype=bool)
        while (episode_counts < episode_count_targets).any():
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            new_observations, rewards, dones, infos = _env.step(actions)
            policy_agent_indexes = model.get_2_policy_agent_indexes_from_obs(last_obs=observations)

            for env in range(n_envs):
                polid = next(polid for polid, envs in enumerate(policy_agent_indexes) if env in envs)
                current_rewards[polid][env] += rewards[env]
                current_lengths[polid][env] += 1
                if episode_counts[env] < episode_count_targets[env]:
                    # unpack values so that the callback can access the local variables
                    reward = rewards[env]
                    done = dones[env]
                    info = infos[env]
                    episode_starts[env] = done

                    if callback is not None:
                        callback(locals(), globals())

                    if dones[env]:
                        if is_monitor_wrapped:
                            # Atari wrapper can send a "done" signal when
                            # the agent loses a life, but it does not correspond
                            # to the true end of episode
                            if "episode" in info.keys():
                                # Do not trust "done" with episode endings.
                                # Monitor wrapper includes "episode" key in info if environment
                                # has been wrapped with it. Use those rewards instead.
                                episode_rewards[polid].append(info["episode"]["r"])
                                episode_lengths[polid].append(info["episode"]["l"])
                                # Only increment at the real end of an episode
                                episode_counts[env] += 1
                        else:
                            episode_rewards[polid].append(current_rewards[polid][env])
                            episode_lengths[polid].append(current_lengths[polid][env])
                            episode_counts[env] += 1
                        current_rewards[polid][env] = 0
                        current_lengths[polid][env] = 0

            observations = new_observations

            if render:
                env.render()

        mean_reward = [np.mean(episode_reward) for episode_reward in episode_rewards]  # num_policies long
        std_reward = [np.std(episode_reward) for episode_reward in episode_rewards]
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
        if return_episode_rewards:
            return episode_rewards, episode_lengths
        return mean_reward, std_reward

