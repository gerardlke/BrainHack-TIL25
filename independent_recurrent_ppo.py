import time
from collections import deque
from typing import Any, Dict, List, Optional, Type, Union, TypeVar
from copy import deepcopy
import gymnasium

import numpy as np
import torch
from operator import itemgetter
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

from stable_baselines3.common.utils import safe_mean

from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv


from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

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
        n_steps: int,
        env: ParallelEnv | MarkovVectorEnv,
        num_policies: int = 2,
        learning_rate: Union[float, Schedule] = 3e-4,
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
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """
        New non RecurrentPPO arguments here:
        
        -> num_agents: The number of agents in the environment, should be 4.
        -> num_policies: The distinct policies we are training over, should be 2.
        """
        self.env = env
        self.n_steps = n_steps
        assert num_agents == 4
        assert num_policies == 2
        self.num_agents = num_agents
        self.num_policies = num_policies

        # keep seperate variables for number of envs 
        self.num_envs = env.num_envs // num_agents  # what this really represents is the number of AECEnvs.
        # what the below represents is the equivalent environment space that observations will be stacked along.
        # e.g for a base AECEnv with 4 agents, vectorized in 2 envs, there are indeed 2 envs, but the way the policies
        # communicate is through treating it as if it is a 'number of agents under that policy * number of actual environments these
        # agents are acting under', hence why in this case, self.num_scout_envs is 2, self.num_guard_envs is 6.
        self.num_scout_envs = self.num_envs * 1  # i love hardcoding!
        self.num_guard_envs = self.num_envs * 3

        self.observation_space = env.observation_space
        self.action_space = env.action_space

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
            RecurrentPPO(
                policy=policy,
                env=self.dummy_envs[i],
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
        print(f'NOTE: TOTAL TIMESTEPS {total_timesteps} INCLUDES NUMBER OF ENVIRONMENTS (currently {self.num_envs}), AND IS NOT A PER-ENV BASIS')

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

        while num_timesteps < total_timesteps:
            # environment sampling. has to be done in this particular way because of
            # gridworld's perculiarities
            rollout_timesteps = self.collect_rollouts(
                last_obs=reset_obs,
                n_rollout_steps=self.n_steps * self.num_envs,  # rollout increments timesteps by number of envs
                callbacks=callbacks,
            )
            num_timesteps += rollout_timesteps

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
        all_values = [None] * self.num_policies
        all_log_probs = [None] * self.num_policies
        
        n_steps = 0
        # before formatted, last_obs is the direct return of self.env.reset()
        placeholder_actions = np.empty(self.num_agents * self.num_envs, dtype=np.int64)
        policy_agent_indexes = self.get_policy_agent_indexes_from_scout(last_obs=last_obs)
        # the boolean mask to apply on each observation, mapping each to each policy.
        # as a sanity check, this should be [1 0 0 0] * self.num_envs since self.env is freshly initialized.
        last_obs = self.format_env_returns(last_obs, policy_agent_indexes, device=self.policies[0].device)

        # iterate over policies, and do pre-rollout setups.
        for polid, (policy, num_envs) in enumerate(zip(self.policies, [self.num_guard_envs, self.num_scout_envs])):
            policy.policy.set_training_mode(False)
            policy.n_steps = 0
            policy.rollout_buffer.reset()
            policy.ep_info_buffer = deque(maxlen=policy._stats_window_size)
            policy.ep_success_buffer = deque(maxlen=policy._stats_window_size)

            if policy.use_sde:
                policy.policy.reset_noise(self.num_envs)  # type: ignore[operator]

            [callback.on_rollout_start() for callback in callbacks]
            policy._last_episode_starts = np.ones((num_envs,), dtype=bool)
            all_last_episode_starts[polid] = policy._last_episode_starts


        # do rollout
        while n_steps < n_rollout_steps:
            # print('n_steps', n_steps)
            with torch.no_grad():
                for polid, policy in enumerate(self.policies):

                    lstm_states = deepcopy(policy._last_lstm_states)
                    episode_starts = torch.tensor(policy._last_episode_starts, dtype=torch.float32, device=policy.device)
                    actions, values, log_probs, lstm_states = policy.policy.forward(
                        last_obs[polid],
                        lstm_states,
                        episode_starts
                    )
                    all_actions[polid] = actions
                    all_values[polid] = values
                    all_log_probs[polid] = log_probs
                    all_lstm_states[polid] = lstm_states
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

            for polid, policy_agent_index in enumerate(policy_agent_indexes):
                placeholder_actions[policy_agent_index] = all_clipped_actions[polid]

            # actually step in the environment
            obs, rewards, dones, infos = self.env.step(placeholder_actions)
            policy_agent_indexes = self.get_policy_agent_indexes_from_scout(last_obs=obs)

            all_curr_obs = self.format_env_returns(obs, policy_agent_indexes=policy_agent_indexes, device=self.policies[0].device)
            all_rewards = self.format_env_returns(rewards, policy_agent_indexes=policy_agent_indexes, device=self.policies[0].device)
            all_dones = self.format_env_returns(dones, policy_agent_indexes=policy_agent_indexes, device=self.policies[0].device)
            all_infos = self.format_env_returns(infos, policy_agent_indexes=policy_agent_indexes, device=self.policies[0].device)
            # print('all_rewards', all_rewards)
            # apparently handles some timeout issues
            # see GitHub issue #633 of sb3_contrib
            for polid, (dones, policy) in enumerate(zip(all_dones, self.policies)):
                for idx, done_ in enumerate(dones):
                    if (
                        done_
                        and all_infos[polid][idx].get("terminal_observation") is not None
                        and all_infos[polid][idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = policy.policy.obs_to_tensor(all_infos[polid][idx]["terminal_observation"])[0]
                        with torch.no_grad():
                            terminal_lstm_state = (
                                all_lstm_states[polid].vf[0][:, idx : idx + 1, :].contiguous(),
                                all_lstm_states[polid].vf[1][:, idx : idx + 1, :].contiguous(),
                            )
                            # terminal_lstm_state = None
                            episode_starts = torch.tensor([False], dtype=torch.float32, device=self.device)
                            terminal_value = policy.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                        rewards[idx] += policy.gamma * terminal_value

            for policy in self.policies:
                policy.num_timesteps += self.num_envs

            for callback in callbacks:
                callback.update_locals(locals())
            
            # TODO: find out why this line exists.
            # if not [callback.on_step() for callback in callbacks]:
            #     break
            n_steps += self.num_envs

            # add data to the rollout buffers
            for polid, policy in enumerate(self.policies):
                policy._update_info_buffer(all_infos[polid])
                if isinstance(self.action_space, Discrete):
                    all_actions[polid] = all_actions[polid].reshape(-1, 1)
                if hasattr(all_actions[polid], 'cpu'):
                    all_actions[polid] = all_actions[polid].cpu().numpy()
                # all_obs[polid] is a list[dict[str, np.ndarray]], but add only wants per dict. hence this loop
                policy.rollout_buffer.add(
                    obs=deepcopy(last_obs[polid]),
                    action=deepcopy(all_actions[polid]),
                    reward=deepcopy(all_rewards[polid]),
                    episode_start=deepcopy(policy._last_episode_starts),
                    value=deepcopy(all_values[polid]),
                    log_prob=deepcopy(all_log_probs[polid]),
                    lstm_states=deepcopy(policy._last_lstm_states)
                )
                policy._last_obs = all_curr_obs[polid]
                policy._last_episode_starts = all_dones[polid]
                policy._last_lstm_states = all_lstm_states[polid]

            last_obs = all_curr_obs
            all_last_episode_starts = all_dones


        for callback in callbacks:
            callback.on_rollout_end()

        return n_steps

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
        Helper func to decode which indexes of observations of shape (self.num_envs * self.num_agents, ...)
        map to which policies.
        """
        policy_mapping = last_obs['scout']
        policy_agent_indexes = [None] * self.num_policies
        for polid in range(self.num_policies):
            policy_agent_indexes[polid] = np.where(policy_mapping == polid)[0]

        return policy_agent_indexes