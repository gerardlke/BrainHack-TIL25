import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from einops import rearrange
from operator import itemgetter
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.ppo import PPO

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class ModifiedPPO(PPO):
    """
    Modified with some added functionality lol thats about it
    """

    @staticmethod
    def get_scout_from_obs(observation, already_bits=False, is_flattened=True, frame_stack_dim=0):
        """
        Overall handler function to do everything for us.
        Observation may be a dictionary or simple numpy array.
        It may be frame-stacked. It may also have been vectorized.
        Whatever way it is, we will rearrange it into an N, (C R B) array, where C R B values are hardcoded.
        """

        if isinstance(observation, dict):
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
        
        Right now, very hardcoded to two agents
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
            to_policies = [
                mutate_func(np.take(env_returns, policy_agent_indexes[polid], axis=0), **kwargs) for polid in range(num_policies)
            ]
        elif isinstance(env_returns, list):
            # for now only 'info' fits in here. dont need to mutate?
            to_policies = [
                list(itemgetter( *(policy_agent_indexes[polid].tolist()) )(env_returns))
                if len(policy_agent_indexes[polid]) > 1 else [itemgetter( *(policy_agent_indexes[polid].tolist()) )(env_returns)]
                for polid in range(num_policies)
            ]
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
    
    
    def run_tests(self):
        # for box viewcones
        test_case_1 = np.random.randint(0, 255, (7, 5))  # during true eval, observation we get will be just a 7 by 5 tensor
        test_case_2 = np.random.randint(0, 2, (16, 7, 5))  # only vectorized environment or frame stacking
        test_case_3 = np.random.randint(0, 2, (16, 16, 7, 5))  # both vectorized environment and frame stacking
        
        self.get_scout_from_obs(test_case_1, already_bits=False, is_flattened=False)
        self.get_scout_from_obs(test_case_2, already_bits=False, is_flattened=False)
        self.get_scout_from_obs(test_case_3, already_bits=False, is_flattened=False, frame_stack_dim=0)

        # for flattened, binarized viewcones
        test_case_4 = np.random.randint(0, 255, (8 * 7 * 5))
        test_case_5 = np.random.randint(0, 2, (4, 8 * 7 * 5 * 2)) # here is 4 is env * agents, 2 is frame stacking along 1d.  




        