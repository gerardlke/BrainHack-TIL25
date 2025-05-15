import time
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Type, Union
from copy import deepcopy
import gymnasium
from gymnasium import spaces
import numpy as np
import torch as th
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
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

from copy import deepcopy
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy

SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")
from sb3_contrib.ppo_recurrent import RecurrentPPO

class DummyGymEnv(gymnasium.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class IndependentRecurrentPPO(OnPolicyAlgorithm):
    """
    Acts as a wrapper of Multi-agent implementation of RecurrentPPO, specific to gridworld's case.
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

    We only train one scout and one guard model. The guard model will be trained on the other three
    guards' experience, while the scout only learns from scout experience. Hence technically only two models exist.
    Instead of num_agents, we will map each output from the environments to an agemt. Then stack along the batch dimension and
    feed it to the corresponding agent.  
    """

    def __init__(
        self,
        policy: Union[str, type[RecurrentActorCriticPolicy]],
        num_agents: int,
        env: ParallelEnv | MarkovVectorEnv,
        num_policies: int = 2,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """
        New non RecurrentPPO arguments here:
        
        -> num_agents: The number of agents in the environment, should be 4.
        -> num_policies: The distinct policies we are training over, should be 2.
        """
        self.env = env
        assert num_agents == 4
        assert num_policies == 2
        self.num_agents = num_agents
        self.num_policies = num_policies

        # keep seperate variables for number of envs 
        self.num_envs = env.num_envs // num_agents

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self._logger = None
        env_fn = lambda: DummyGymEnv(self.observation_space, self.action_space)
        dummy_envs = [DummyVecEnv([env_fn] * 6), DummyVecEnv([env_fn] * 2)]
        # this is a wrapper class, so it will not hold any states like
        # buffer_size, or action_noise. pass those straight to DQN.

        # we keep self.agents as a list, where the first is always the scout and the second is always the guard model.
        # for now do not 
        self.policies = [
            RecurrentPPO(
                policy=policy,
                env=dummy_envs[i],
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                normalize_advantage=normalize_advantage,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                target_kl=target_kl,
                stats_window_size=stats_window_size,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                seed=seed,
                device=device,
                _init_setup_model=_init_setup_model,
            )
            for i in range(self.num_policies)
        ]

        assert self.num_policies == 2, 'Right now only supports 2 num_policies; 1 for scout and 1 unified guard'

    def learn(
        self,
        total_timesteps: int,
        n_rollout_steps: int,
        callbacks: Optional[List[List[MaybeCallback]]] = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Main learn function. Mainly goes as follows:

        target: collect rollbacks and train up till total_timesteps.
        """

        if callbacks is not None:
            assert len(callbacks) == self.num_policies, 'callbacks must a list of num_policies number of nested lists'
            assert all(isinstance(callback, list) for callback in callbacks), 'callbacks must a list of num_policies number of nested lists'

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
                "policy",
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

        while num_timesteps < total_timesteps:
            # environment sampling. has to be done in this particular way because of
            # gridworld's perculiarities
            last_obs, rollout_timesteps = self.collect_rollouts(
                last_obs=reset_obs,
                n_rollout_steps=n_rollout_steps,
                callbacks=callbacks,
            )
            num_timesteps += rollout_timesteps

            # agent training.
            for polid, policy in enumerate(self.policy):
                policy._update_current_progress_remaining(
                    policy.num_timesteps, total_timesteps  # 
                )
                if log_interval is not None and num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    policy.logger.record("polid", polid, exclude="tensorboard")
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

    def collect_rollouts(
            self,
            last_obs,
            n_rollout_steps: int,
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
        all_last_episode_starts = [None] * self.num_policies
        all_clipped_actions = [None] * self.num_policies
        all_next_obs = [None] * self.num_policies
        all_rewards = [None] * self.num_policies
        all_dones = [None] * self.num_policies
        all_infos = [None] * self.num_policies
        all_actions = [None] * self.num_policies
        all_lstm_states = [None] * self.num_policies
        
        n_steps = 0
        # before formatted, last_obs is the direct return of self.env.reset()
        policy_mapping = last_obs['scout']
        # the boolean mask to apply on each observation, mapping each to each policy.
        # as a sanity check, this should be [1 0 0 0] since self.env is freshly initialized.
        last_obs = self.format_env_returns(last_obs, policy_mapping, device=self.policies[0].device)

        # iterate over agents, and do pre-rollout setups.
        for polid, policy in enumerate(self.policies):
            policy.policy.set_training_mode(False)
            policy.n_steps = 0
            policy.rollout_buffer.reset()

            if policy.use_sde:
                policy.policy.reset_noise(self.num_envs)  # type: ignore[operator]

            [callback.on_rollout_start() for callback in callbacks]
            all_last_episode_starts[polid] = policy._last_episode_starts

            lstm_states = deepcopy(policy._last_lstm_states)

        while n_steps < n_rollout_steps:
            for polid, policy in enumerate(self.policies):
                if policy.use_sde and policy.sde_sample_freq > 0 and n_steps % policy.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    policy.policy.reset_noise(self.num_envs)

            with th.no_grad():
                for polid, policy in enumerate(self.policies):
                    print('last_obs[polid]', last_obs[polid])
                    episode_starts = th.tensor(policy._last_episode_starts, dtype=th.float32, device=policy.device)
                    actions, values, log_probs, lstm_states = policy.policy.forward(
                        last_obs[polid],
                        lstm_states,
                        episode_starts
                    )
                    print('actions??', actions)
                    all_actions[polid] = actions
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

            vstack_clipped_actions = (
                np.vstack(all_clipped_actions).transpose().reshape(-1)
            )  # reshape as (env, action)

            # actually step in the environment
            obs, rewards, dones, infos = self.env.step(vstack_clipped_actions)
            # obs may be a list like the others, or may be a dict, each string key indexing sometensor of shape 
            all_curr_obs = self.format_env_returns(obs, device=self.agents[0].device)
            all_rewards = self.format_env_returns(rewards, device=self.agents[0].device)
            all_dones = self.format_env_returns(dones, device=self.agents[0].device)
            all_infos = self.format_env_returns(infos, device=self.agents[0].device)

            for policy in self.agents:
                policy.num_timesteps += self.num_envs

            for callback in callbacks:
                callback.update_locals(locals())
            if not [callback.on_step() for callback in callbacks]:
                break

            n_steps += self.num_envs

            # add data to the replay buffers
            for polid, policy in enumerate(self.policies):
                policy._update_info_buffer(all_infos[polid])
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                if hasattr(all_actions[polid], 'cpu'):
                    all_actions[polid] = all_actions[polid].cpu().numpy()
                # all_obs[polid] is a list[dict[str, np.ndarray]], but add only wants per dict. hence this loop
                policy.rollout_buffer.add(
                    obs=deepcopy(last_obs[polid]),
                    next_obs=deepcopy(all_curr_obs[polid]),
                    action=deepcopy(all_buffer_actions[polid]),
                    reward=deepcopy(all_rewards[polid]),
                    done=deepcopy(all_dones[polid]),
                    infos=deepcopy(all_infos[polid]),
                )
                # rollout_buffer.add(
                #     self._last_obs,
                #     actions,
                #     rewards,
                #     self._last_episode_starts,
                #     values,
                #     log_probs,
                #     lstm_states=self._last_lstm_states,
                # )

            last_obs = all_curr_obs
            all_last_episode_starts = all_dones
            # if any(np.any(done) for done in dones):
            #     last_obs = self.env.reset()
            #     last_obs = self.format_env_returns(last_obs, device=self.agents[0].device)

        # special dqn thing.
        if run_on_step:
            for polid, agent in enumerate(self.agents):
                if hasattr(agent, '_on_step'):
                    agent._on_step()

        for callback in callbacks:
            callback.on_rollout_end()

        for polid, policy in enumerate(self.agents):
            policy._last_episode_starts = all_last_episode_starts[polid]

        return obs, num_collected_steps

    @classmethod
    def load(
        cls,
        path: str,
        policy: Union[str, Type[ActorCriticPolicy]],
        num_agents: int,
        env: GymEnv,
        learning_starts: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        **kwargs,
    ) -> "IndependentDQN":
        model = cls(
            policy=policy,
            num_agents=num_agents,
            env=env,
            learning_starts=learning_starts,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **kwargs,
        )
        env_fn = lambda: DummyGymEnv(env.observation_space, env.action_space)
        dummy_env = DummyVecEnv([env_fn] * (env.num_envs // num_agents))
        for polid in range(num_agents):
            model.agents[polid] = DQN.load(
                path=path + f"/agents_{polid}/model", env=dummy_env, **kwargs
            )
        return model

    def save(self, path: str) -> None:
        for polid in range(self.num_agents):
            self.agents[polid].save(path=path + f"/agents_{polid}/model")


    def format_env_returns(
        self,
        env_returns: dict[str, np.ndarray] | np.ndarray | list[dict],
        policy_mapping: list,
        to_tensor=True,
        device=None,
    ):
        """
        Helper function to format returns based on if they are a dict of arrays or just arrays.
        We expect the first dimension of these arrays to be (num_envs * num_agents).

        The flow is as follows:
        1. Generate indexes for each policy
        2. Use indexes to extract the appropriate observations per policy.

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

        policy_mapping = np.array(policy_mapping)
        policy_indexes = [None] * self.num_policies
        for polid in range(self.num_policies):
            policy_indexes[polid] = np.where(policy_mapping == polid)[0]
            print('policy_indexes[polid]', policy_indexes[polid])

        # 1. appropriate indexing
        if isinstance(env_returns, dict):
            to_policies = [
                    {k: mutate_func(np.take(v, policy_indexes[polid], axis=0), **kwargs) 
                        for k, v in env_returns.items()}
                for polid in range(self.num_policies)
            ]
        elif isinstance(env_returns, np.ndarray) or isinstance(env_returns, list):
            to_policies = [
                mutate_func(np.take(env_returns, policy_indexes[polid], axis=0), **kwargs) for polid in range(self.num_policies)
            ]
        else:
            raise AssertionError(f'Assertion failed. format_env_returns recieved unexpected type {type(env_returns)}. \
                Expected dict[str, np.ndarray] or np.ndarray or list.')

        return to_policies
