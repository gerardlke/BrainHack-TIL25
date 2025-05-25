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
from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO

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
    
    
# class ModifiedMaskedPPO(MaskablePPO):


        