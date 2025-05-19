import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import EventCallback, BaseCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def custom_evaluate_policy(
    model: "type_aliases.PolicyPredictor",
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
    yoink evaluate_policy from stable_baselines3/common/evaluation, but tweak to support model
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
    n_envs = env.num_envs
    episode_rewards = [[] for _ in range(model.num_policies)] # nested list, per policy.
    # each nested list is appended on a per-env basis
    episode_lengths = [[] for _ in range(model.num_policies)] 
    all_clipped_actions = [None] * model.num_policies
    all_observations = [None] * model.num_policies
    placeholder_actions = np.empty(model.num_agents * model.num_envs, dtype=np.int64)
    all_actions = [None] * model.num_policies
        
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    episode_starts = [np.ones((env.num_envs,), dtype=bool)] * model.num_policies

    # per policy per env per agent, rewards are aggregated.
    # btw n_envs is already the num_vec_envs * num_agents, cuz supersuit and parallel env are nice like that
    # for each env that reaches a ended state, we the reward of that agent-env to episode_rewards.
    # vice versa for episode_lengths
    current_rewards = [np.zeros(n_envs)] * model.num_policies  
    current_lengths = [np.zeros(n_envs, dtype="int")] * model.num_policies

    # reset and format last observation
    observations = env.reset()
    policy_agent_indexes = model.get_policy_agent_indexes_from_viewcone(last_obs=observations)
    last_obs = model.format_env_returns(observations, policy_agent_indexes, device=model.policies[0].device)

    states = None
    # predict each policy
    while (episode_counts < episode_count_targets).any():
        for polid, policy in enumerate(model.policies):
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
            if isinstance(model.action_space, Box):
                clipped_actions = np.clip(
                    clipped_actions,
                    model.action_space.low,
                    model.action_space.high,
                )
            elif isinstance(model.action_space, Discrete):
                # get integer from numpy array
                clipped_actions = np.array(
                    [action.item() for action in clipped_actions]
                )
            all_clipped_actions[polid] = clipped_actions

        for polid, policy_agent_index in enumerate(policy_agent_indexes):
            placeholder_actions[policy_agent_index] = all_clipped_actions[polid]

        # check placeholder actions?
        print('placeholder_actions', placeholder_actions)
        
        new_observations, rewards, dones, infos = env.step(placeholder_actions)
        policy_agent_indexes = model.get_policy_agent_indexes_from_viewcone(last_obs=new_observations)
        all_curr_obs = model.format_env_returns(new_observations, policy_agent_indexes, device=model.policies[0].device)
        all_rewards = model.format_env_returns(rewards, policy_agent_indexes, device=model.policies[0].device, to_tensor=False)
        all_dones = model.format_env_returns(dones, policy_agent_indexes, device=model.policies[0].device, to_tensor=False)
        all_infos = model.format_env_returns(infos, policy_agent_indexes, device=model.policies[0].device, to_tensor=False)
        # print('all_dones', all_dones)
        # print('placeholder_actions', placeholder_actions)


        for polid, (
            policy_agent_index,
            current_reward,
            current_length,
            episode_count,
            episode_count_target,
            episode_start,
            episode_reward,
            episode_length,
            all_reward,
            all_done,
            all_info,
        ) in enumerate(zip(
            policy_agent_indexes,
            current_rewards,
            current_lengths,
            episode_counts,
            episode_count_targets,
            episode_starts,
            episode_rewards,
            episode_lengths,
            all_rewards,
            all_dones,
            all_infos,
        )):
            # policy-based enumeration: each thing in the big tuple up there are of length n_envs
            # policy_agent_index is the indexes of envs * num_agents, that correspond to each policy.
            for i in range(policy_agent_index):
                current_reward[i] += all_reward[i]
                current_length[i] += 1
                if episode_count[i] < episode_count_target[i]:
                    # unpack values so that the callback can access the local variables
                    done = all_done[i]
                    info = all_info[i]
                    episode_start[i] = done

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
                                episode_count[i] += 1
                        else:
                            episode_reward.append(current_reward[i])
                            episode_length.append(current_length[i])
                            episode_count[i] += 1
                        current_reward[i] = 0
                        current_length[i] = 0

        last_obs = all_curr_obs

        if render:
            env.render()
    print('episode_rewards, episode_lengths', episode_rewards, episode_lengths)
    mean_reward = np.mean(episode_rewards)  # expects list of lists
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths

    return mean_reward, std_reward


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
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
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
        self.log_path = log_path
        self.evaluations_results: list[list[float]] = []
        self.evaluations_timesteps: list[int] = []
        self.evaluations_length: list[list[int]] = []
        # For computing success rate
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

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

            episode_rewards, episode_lengths = custom_evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
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

