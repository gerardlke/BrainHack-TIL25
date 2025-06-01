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
    
    
class ModifiedMaskedPPO(MaskablePPO):

    def get_action_masks(self, observation):
        """
        action indexes:
        0 - move forward
        1 - move backward
        2 - turn left
        3 - turn right
        4 - stationary

        if agent is facing right:
        ACTION SPACE (absolute viewpoint as north 0) -> AGENT POV
        0 -> 2  'to move cardinal forward, agent must turn left'
        1 -> 3  'to move cardinal backward, agent must turn right'
        2 -> 1  'to move (turn) cardinal left, agent must move backwards'
        3 -> 0  'to move (turn) cardinal right, agent must move forward'

        if agent is facing left:
        0 -> 3
        1 -> 2
        2 -> 0
        3 -> 1

        if agent is facing up:
        # everything is normal

        if agent is facing down:
        0 -> 1
        1 -> 0
        2 -> 3
        3 -> 2

        observation bit indexes:
        Value of last 2 bits (tile & 0b11)	Meaning
        0	No vision
        1	Empty tile
        2	Recon (1 point)
        3	Mission (5 points)
        Bit index (least significant bit lowest)	Tile contains a...
        2	Scout
        3	Guard
        # the following are WITH RESPECT TO AGENT POV, NOT THE ABSOLUTE VIEW
        4	Right wall -> index 3 of bit split
        5	Bottom wall -> index 2 of bit split
        6	Left wall -> index 1 of bit split
        7	Top wall -> index 0 of bit split

        direction of facing:
        0 - right
        1 - bottom
        2 - left
        3 - top

        """
        viewcone = observation['viewcone']

        # will fail for the case of normal obs for now
        action_masks = np.zeros((viewcone.shape[0], self.action_space.n))
        test = rearrange(viewcone, 'A (S R C B) -> A S B R C', B=8, R=7, C=5)
        test = test[:, -1, :, :, :]

        # 1. Disable the corresponding action if there is a wall there.
        # step a: Rearrange the order of directions to fit action space order
        curr_tile_walls = test[:, :4, 2, 2]
        ### IN HERE, LETS SAY YOU GET [0, 0, 1, 1]. This translates to 
        # [0, 0, 1, 1, 0, 1, 0, 0]
        # left of the agent, behind the agent, right of the agent, forward of agent. AGENT POV
        # i,e top and right of agent have a wall. why did ryan make it this way???
        # anyways lbrf -> fblr
        action_wall_indexes = np.array([3, 1, 0, 2])  # i.e index 3 of wall obs (top wall) maps to index 0 of which 
        # action is influenced by its presence
        enabled_actions = np.where(curr_tile_walls == 1, 0, 1)
        enabled_actions = enabled_actions[:, action_wall_indexes]
        # now that we've permuted from obs to action arrangement, permute according to
        # to agent's facing direction.

       
        # do other stuff like masking for scout if need to. later compile all information
        # to build this action mask. for now its just wall-enabled actions
        action_masks[:, :4] = enabled_actions

        # 0: only enable stationary action if all of them are disabled.
        null_action_rows = np.all(action_masks == 0, axis=1)
        action_masks[null_action_rows, -1] = 1

        return action_masks

        