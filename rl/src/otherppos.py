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

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback


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

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        print('callback before init', callback)
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

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
        if viewcone.ndim == 2:
            action_masks = np.zeros((viewcone.shape[0], self.action_space.n))
            viewcone = rearrange(viewcone, 'A (S R C B) -> A S B R C', B=8, R=7, C=5)
            viewcone = viewcone[:, -1, :, :, :]
            center_tile_walls = viewcone[:, :4, 2, 2]
            adjacent_tile_guards = viewcone[:, 4, 1:4, 1:4]
            scout_in_view = viewcone[:, 5, :, :]
        elif viewcone.ndim == 1:
            action_masks = np.zeros((1, self.action_space.n))
            viewcone = rearrange(viewcone, '(S R C B) -> S B R C', B=8, R=7, C=5)
            viewcone = viewcone[-1, :, :, :]
            center_tile_walls = viewcone[:4, 2, 2][np.newaxis, :]
            # left and right 2, front and back one. so it has greater foresight
            # and can turn one move in advance
            adjacent_tile_guards = viewcone[4, :5, 1:4][np.newaxis, :]
            scout_in_view = viewcone[5, :, :][np.newaxis, :]
        else:
            raise AssertionError('Assertion failed. viewcone has more than 2 dimensions, which is not currently supported by the get_action_masks ' \
                'method in ModifiedMaskedPPO.')

        # 1. Disable the corresponding action if there is a wall there.
        # step a: Rearrange the order of directions to fit action space order
        
        ### IN HERE, LETS SAY YOU GET [0, 0, 1, 1]. This translates to 
        # [0, 0, 1, 1, 0, 1, 0, 0]
        # left of the agent, behind the agent, right of the agent, forward of agent. AGENT POV
        # i,e top and right of agent have a wall. why did ryan make it this way???
        # anyways lbrf -> fblr
        action_wall_indexes = np.array([3, 1, 0, 2])  # i.e index 3 of wall obs (top wall) maps to index 0 of which 
        # action is influenced by its presence
        # print('center_tile_walls', center_tile_walls)
        enabled_actions = np.where(center_tile_walls == 1, 0, 1)
        # print('enabled_actions', enabled_actions)
        enabled_actions = enabled_actions[:, action_wall_indexes]
        # now that we've permuted from obs to action arrangement, permute according to
        # to agent's facing direction.
        # print('enabled-actions', enabled_actions)
        # masking to help scout/guard targetting.
        if isinstance(adjacent_tile_guards, th.Tensor):
            adjacent_tile_guards = adjacent_tile_guards.numpy()
        center_is_0 = adjacent_tile_guards[:, 1, 1] == 0
        overall_masks = enabled_actions
        # overall_masks = check_four_divisions(
        #     adjacent_tile_guards,
        #     existing_masks=enabled_actions,
        #     condition=center_is_0,
        #     attract=False,
        #     combine='and',
        # )        
        # print('after scout masks', overall_masks)
        center_is_1 = adjacent_tile_guards[:, 1, 1] == 1 
        if isinstance(scout_in_view, th.Tensor):
            scout_in_view = scout_in_view.numpy()
        overall_masks = check_four_divisions(
            scout_in_view,
            existing_masks=overall_masks,
            condition=center_is_1,
            attract=True,
            combine='guards',
        )
        # print('after guards masks', overall_masks)
        
        # combine together boolean masks here:::
       
        # do other stuff like masking for scout if need to. later compile all information
        # to build this action mask. for now its just wall-enabled actions
        action_masks[:, :4] = overall_masks

        null_action_rows = np.all(action_masks == 0, axis=1)
        action_masks[null_action_rows, :4] = 1  # hail mary. something is better than nothing
        # print('action_masks', action_masks)
        return action_masks


def check_four_divisions(arr, existing_masks, condition, attract=True, combine='guards'):
    is_torch = isinstance(arr, th.Tensor)

    # Helper functions
    def any_fn(x, axis):
        if attract:
            return th.any(x, dim=axis) if is_torch else np.any(x, axis=axis)
        else:
            return ~th.any(x, dim=axis) if is_torch else np.logical_not(np.any(x, axis=axis))

    def cast_to_int(x):
        return x.int() if is_torch else x.astype(int)

    def stack_fn(tensors, axis):
        return th.stack(tensors, dim=axis) if is_torch else np.stack(tensors, axis=axis)

    # Divisions
    bottom    = arr[..., :2, :]   # rows 0–1
    top = arr[..., 3:, :]   # rows 3–6
    right   = arr[..., :, :2]   # cols 0–1
    left  = arr[..., :, 3:]   # cols 3–4

    # Check for any 1s in each division
    top_has_1    = any_fn(top == 1, axis=(-2, -1))
    bottom_has_1 = any_fn(bottom == 1, axis=(-2, -1))
    left_has_1   = any_fn(left == 1, axis=(-2, -1))
    right_has_1  = any_fn(right == 1, axis=(-2, -1))

    # Combine into binary indicator: shape (..., 4)
    result = stack_fn([
        top_has_1,
        bottom_has_1,
        left_has_1,
        right_has_1
    ], axis=-1)

    result = cast_to_int(result)

    new_result = update_result_where_condition(
        result=existing_masks,
        new_result=result,
        condition=condition,
        mode=combine
    )
    # for idx, con in enumerate(condition):
    #     if con and np.any(result[idx]) and not np.array_equal(result[idx], new_result[idx]):
    #         print('result[idx]', result[idx])
    #         print('new_result[idx]', new_result[idx])
    #         print('existing_masks', existing_masks)
    #         print('result', result)
    #         print('new_result', new_result)

    return new_result  # shape (..., 4)


def compute_outcome(base, mask):
    base = np.asarray(base)
    mask = np.asarray(mask)

    assert base.shape == mask.shape, "Shapes must match"

    outcome = np.zeros_like(base, dtype=float)

    # Base rule
    both_one = (base == 1) & (mask == 1)
    base_only = (base == 1) & (mask == 0)

    # Determine per-row if any (base == 1 and mask == 1)
    row_has_both_one = np.any(both_one, axis=-1)  # shape: (batch,)

    for i in range(base.shape[0]):
        if row_has_both_one[i]:
            # Normal case: base==1 & mask==1 → 1, base==1 & mask==0 → 0.1
            outcome[i][both_one[i]] = 1.0
        else:
            # Override case: if no (1 & 1), then base==1 → 1.0
            outcome[i][base_only[i]] = 1.0

    return outcome


def update_result_where_condition(result, new_result, condition, mode="guards"):
    if isinstance(result, th.Tensor):
        mask = condition.unsqueeze(-1).expand_as(result)

        if mode == "guards":
            result = result.clone()
            result[mask] = compute_outcome(result[mask], new_result[mask])
        elif mode == "and":
            result = result.clone()
            result[mask] = result[mask] & new_result[mask]
        else:
            raise ValueError("mode must be 'or' or 'and'")
    else:
        # NumPy
        mask = np.expand_dims(condition, axis=-1)
        mask = np.broadcast_to(mask, result.shape)

        if mode == "guards":
            result_conditioned = np.where(mask, result, 0)
            maybe_chase = compute_outcome(result_conditioned, new_result)
            result_maybe_chase = np.where(mask, maybe_chase, result)
            
            # for idx, chase in enumerate(result_maybe_chase):
            #     if not np.array_equal(result[idx], chase):
            #         print('result', result)
            #         print('result after masking again', result_maybe_chase)
                    

            result = result_maybe_chase
            
        elif mode == "and":
            result = result.copy()
            result[mask] = result[mask] & new_result[mask]
        else:
            raise ValueError("mode must be 'or' or 'and'")

    return result
