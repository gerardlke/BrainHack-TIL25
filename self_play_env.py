"""
Yoink from gridworld.py, but mutates with a whole host of additional functionalities
"""
import supersuit as ss
import functools
import logging
import warnings
import os
import hashlib
import cv2
import time
import gymnasium
import numpy as np
import pygame
import inspect

from rl.db.db import RL_DB
from copy import deepcopy
from functools import partial
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper
from pettingzoo.utils.env import ParallelEnv
from collections import defaultdict
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.seeding import np_random
from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper
from supersuit import frame_stack_v2
from til_environment.flatten_dict import FlattenDictWrapper
from til_environment.helpers import (
    convert_tile_to_edge,
    get_bit,
    idx_to_view,
    is_idx_valid,
    is_world_coord_valid,
    manhattan,
    rotate_right,
    supercover_line,
    view_to_idx,
    view_to_world,
    world_to_view,
)
from til_environment.types import Action, Direction, Player, RewardNames, Tile, Wall
from einops import rearrange

from supersuit.utils.frame_stack import stack_init, stack_obs, stack_obs_space
from til_environment.gridworld import raw_env

from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from pettingzoo.utils.wrappers import BaseWrapper

from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper


def build_env(
    reward_names,
    rewards_dict,
    policy_mapping,
    agent_roles,
    self_play,
    npcs,
    db_path,
    env_config,
    num_iters=100,
    env_wrappers: list[BaseWrapper] | None = None,
):
    """
    Define configurations of our environment.
    TODO: update this.
    Args:
        - reward_names: customized RewardNames class to pass to the class init
        - rewards_dict: rewards dictionary to specify exactly what values
        - env_wrappers: list of other env wrappers before we parallelize, framestack and vectorize
            the environment
        - num_vec_envs: number of vector environments to stack
        - render_mode: How to render the environment (should be none during training unless you are trying to
            debug)
        - env_type: right now only supports 'binary_viewcone' or 'normal'.
        - eval_mode: Boolean to set environment to eval mode; this will remove things like 
            automatic guard distance penalties from being applied.
        - novice: True or False, are we making novice env
        - debug: Enable debug mode for environments, which can offer insight into
            its inner workings
        - frame_stack_size: how many past frames to stack, mutates the observation space
        **kwargs: other kwargs to pass to the environment class
    """
    _env_config = deepcopy(env_config)
    num_vec_envs = _env_config.pop('num_vec_envs', True)
    binary = _env_config.pop('binary', True)
    render_mode = _env_config.pop('render_mode', None)
    eval_mode = _env_config.pop('eval_mode')
    novice = _env_config.pop('novice', True)
    debug = _env_config.pop('debug', False),
    frame_stack_size = _env_config.pop('frame_stack_size', 4)
    opponent_sampling  = _env_config.pop('opponent_sampling', 'random')
    collisions = _env_config.pop('collisions', True)
    viewcone_only = _env_config.pop('viewcone_only', False)

    orig_env = modified_env(
        render_mode=render_mode,
        reward_names=reward_names,
        rewards_dict=rewards_dict,
        eval=eval_mode,
        novice=novice,
        debug=debug,
        binary=binary,
        viewcone_only=viewcone_only,
        collisions=collisions,
        num_iters=num_iters
        # **_env_config
    )

    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(orig_env)
    # Provides a wide variety of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    
    if self_play:
        assert npcs is not None, 'Assertion failed. Are you certain you are running selfplay.py if you are trying to conduct self play? We use an orchestrator there instead, which' \
            ' contains this missing argument.'
        assert db_path is not None, 'Assertion failed. We require a path to database of network pools.'
        env = SelfPlayWrapper(env,
            policy_mapping=policy_mapping,
            agent_roles=agent_roles,
            npcs=npcs,
            opponent_sampling=opponent_sampling,
            db_path=db_path,
        )
    else:
        env = aec_to_parallel(env)

    env = frame_stack_v3(env, stack_size=frame_stack_size, stack_dim=0)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=4, base_class='stable_baselines3')

    return orig_env, env


class SelfPlayWrapper(aec_to_parallel_wrapper):

    """
    A wrapper around pettingzoo env for selfplay. Ok not really self-play especially if you have different policies, but lets roll
    with it. A selfplay env differs from the normal env according to the following:
    - observation_space: Only as big as one agent's observation space
    - action_space: Only as big as one agent's action space
    - step: Each step only takes in one agent's action, and automatically runs prediction for other NPC models.
        (lets call them these for now on)
    - reset: Depending on the reset strategy, we may randomly choose other weights to be loaded in, so the learning
        agent has more variety of opponents to play against. We could swap opponents after each train method call
        (i.e after learning policy's rollout buffer is completely gathered) or just every reset call of the environment. see how
    
    This environment will wrap around a parallelenv, so that when we call step with one agent's set of actions, we run
    a custom step function that first runs predict to get all policies actions, then runs the parallelenv's step function
    on all actions concatenated together.

    Args:
        - env: Instantiated env
        - policy_mapping: Mapping of policies to agents. This is because our network pool
            technically encompasses policies, not agents, and the mapping from policy to agents
            is defined in config (which is passed here.)
        - agent_roles: Indexes of each agent. This is used for db setup and retrieval, and how we concatenate
            the final actions before calling step.
        - npcs: Defined at initialization, what agents will be controlled by the environment. Requires this so
            it knows what policies to load, which it can then map to which agent roles.
        - opponent_sampling: Strategy on how we load opponents each time environment resets. TODO integrate with gerard
        DB.
        - db_path: Path to gerard db
    """
    def __init__(self, env, policy_mapping, agent_roles, npcs, opponent_sampling, db_path):
        super().__init__(env)
        self.db_path = db_path
        # db interface (gerard)
        self.db = RL_DB(db_file=self.db_path)

        self.policy_mapping = policy_mapping
        self.agent_roles = agent_roles
        self.opponent_sampling = opponent_sampling
        # somethign about loading best models here
        assert max(npcs) == 1 and min(npcs) == 0, 'Assertion failed, npcs list recieved by SelfPlayWrapper reset is not a boolean mask.'
        assert npcs.count(0) == 1, 'Assertion failed, npcs list recieved by SelfPlayWrapper contains more than one element 1; we only support ' \
            'one external non-npc agent (mask value 0) currently.'
        
        self.npcs = npcs  # this tells us what agents are to be considered as part of environment.

        self.environment_policies = [policy for policy, keep in zip(self.policy_mapping, self.npcs) if keep]
        self.environment_agents = [agent for agent, keep in zip(self.possible_agents, self.npcs) if keep]
        self.agent_policy_mapping = {
            agent: policy for agent, policy in zip(self.environment_agents, self.environment_policies)
        }

        self.loaded_policies = None
        self.opponent_sampling = opponent_sampling

        # temp hash
        import hashlib
        import secrets

        # Generate a random 32-byte (256-bit) value
        random_bytes = secrets.token_bytes(32)

        # Hash it using SHA-256
        random_hash = hashlib.sha256(random_bytes).hexdigest()

        self.hash = random_hash

    def load_policies(self):
        # arbitrary load checkpoint for each agent function
        self.db.set_up_db(timeout=100)
        print('env', self.hash, 'connected to db')
        checkpoints = [
            self.db.get_checkpoint_by_policy(policy) for policy in self.environment_policies
        ]
        self.db.export_database_to_sql_dump(self.db_path, self.db_path)
        print('self.db_path', self.db_path)

        self.db.shut_down_db()
        
        self.loaded_policies = {
            policy: self.load_policy(c) for c, policy in zip(checkpoints, self.environment_policies)
        }

    def load_policy(self, path=None):
        if path is None:
            # if no checkpoint, default to random behaviour.
            return 'random'
        else:
            return 'example_policy_here'

    def reset(self, seed: int | None=None, options: dict | None=None):  # type: ignore
        """
        This is the core reason for incompatibility with supersuit vector environments n whatnot. Reset now requires us to recieve
        some kinda of boolean mask on which agents are currently NPCs and which are learning (i.e their actions are incoming and not internal to the
        environment.) As this is not supported generally, this wrapper must be the last to wrap around the environment.
        
        Args:
            - npcs: Boolean list of what agents are considered part of environment. If [0, 1, 1, 1], this means that this environment
                will load in the policies tied to agents at indexes 1, 2 and 3 (irregardless of if it is the same policy)
        """
        self.load_policies()
        
        return super().reset(seed, options)

    def step(self, actions):
        """
        Although we support many to one mapping of agents to policies, during stepping we will manually loop through all NPC agents
        and run their relevant policy on the relevant agent observation. 

        actions are a dict of agent names to their actions. Since this is nested within the vectorized wrapper,
        expect exactly what the base environment's action space tells us, without worrying about batches etc.
        """
        for agent in actions:
            if agent in self.environment_agents:
                policy_to_run = self.agent_policy_mapping[agent]
                loaded_policy = self.loaded_policies[policy_to_run]
                if loaded_policy == 'random':
                    action = np.random.randint(0, 5)
                else:
                    obs = self.aec_env.observe(agent)
                    if 'action_masks' not in inspect.signature(loaded_policy.forward).parameters:
                        action, _, _ = loaded_policy.forward(obs)
                    else:
                        action_masks = loaded_policy.get_action_masks(obs)
                        action, _, _ = loaded_policy.forward(obs, action_masks=action_masks)

                actions[agent] = action

        return super().step(actions)


class modified_env(raw_env):
    """
    yoink provided env and add some stuff to it

    Base, standard training env. We disable different agent selection as we would like to fix the indexes of
    the scout and guards, so that we can more easily extract it and parse it into seperate policies.

    Also, action masking is not conducted here, due to multiple wrappings of vector envs
    paralzying us and rendering us unable to call this function here. It will lie within the simulator instead.
    
    Also, default to normalizing environment step and location things, to stabilize training.
    This is done within this modified env, without a wrapper (i couldn't find one that does normalization within each key's values)
    """
    def __init__(self,
            reward_names,
            rewards_dict,
            binary,
            num_iters=100,
            eval=False, 
            collisions=True,
            viewcone_only=False,
            see_scout_reward=True,
        **kwargs):
        # self.rewards_dict is defined in super init call
        super().__init__(**kwargs)
        self.binary = binary
        self.num_iters = num_iters
        self.reward_names = reward_names
        self.rewards_dict = rewards_dict
        self.eval = eval
        self.collisions = collisions
        self.prev_distances = {agent: None for agent in self.possible_agents[:]}
        self.viewcone_only = viewcone_only
        self.see_scout_reward = see_scout_reward
        # Generate 32 bytes of random data
        random_bytes = os.urandom(32)
        # Hash it with SHA-256, so each env can have its own unique identifier
        self.hash = hashlib.sha256(random_bytes).hexdigest()
        
        # because cyclical player shuffling is disabled for this environment, we call this here and never again.
        self._scout = self._scout_selector.next()
        # we can use _scout to check against the latest scout attribute, which allows us to pick up any changes in
        # the scout id. This functionality has not been implemented yet.
        self.scout = self._scout

        # it is the environment's job to hold the state of which policies map to which agents to which policy.
        # normally it should be defined here.
        # however, one might notice that the below is hardcoded. this is intentional, as we lock the scout to always be
        # the first player (0th index).
        self.policy_mapping = [0] * 4
        self.policy_mapping[0] = 1

        # include here a role_mapping item. This is to make clear which players map to which roles.
        # once again, hardcoded to be [1 0 0 0] due to our lock (which is a significant convenience)
        # self.role_mapping = [1, 0, 0, 0]
        # scratch that now its 0, 1, 2, 3
        self.role_mapping = [0, 1, 2, 3]


    def _render_frame(self):
        if self.window is None and self.render_mode in self.metadata["render_modes"]:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode(
                    (self.window_width, self.window_size)
                )
                pygame.display.set_caption("TIL-AI 2025 Environment")
            else:
                self.window = pygame.Surface((self.window_width, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if self.font is None:
            try:
                self.font = pygame.font.Font("freesansbold.ttf", 12)
            except:
                warnings.warn("unable to import font")

        self.window.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # add gridlines
        self._draw_gridlines(self.size, self.size, pix_square_size)

        # draw environment tiles
        for x, y in np.ndindex((self.size, self.size)):
            tile = self._state[x, y]
            # draw whether the tile contains points
            Tile(tile % 4).draw(self.window, x, y, pix_square_size)

            # draw walls
            for wall in Wall:
                if not get_bit(tile, wall.value):
                    continue
                wall.draw(self.window, x, y, pix_square_size)

        # draw all the players
        for agent, location in self.agent_locations.items():
            p = Player.SCOUT if agent == self.scout else Player.GUARD
            p.draw(self.window, location[0], location[1], pix_square_size)

            center = (location + 0.5) * pix_square_size
            # draw direction indicator
            pygame.draw.line(
                self.window,
                (0, 255, 0),
                center,
                (
                    location
                    + 0.5
                    + Direction(self.agent_directions[agent]).movement * 0.33
                )
                * pix_square_size,
                3,
            )
            self._draw_text(agent[-1], center=center)

        # draw debug view
        if self.debug:
            # dividing allowable vertical space (0.2 of the vertical window)
            subpix_square_size = int(0.2 * self.window_size / self.viewcone_width)
            x_corner = int(self.window_size * 1.04)
            x_lim = int(self.window_size * 1.47)
            for agent in self.agents:
                agent_id = int(agent[-1])
                observation = self.observe(agent, default_return=True)

                y_corner = int(self.window_size * (0.24 * agent_id + 0.04))

                # draw gridlines
                self._draw_gridlines(
                    self.viewcone_length,
                    self.viewcone_width,
                    subpix_square_size,
                    x_corner,
                    y_corner,
                )

                # draw debug text information
                try:
                    for i, text in enumerate(
                        [
                            f"id: {agent[-1]}",
                            f"direction: {observation['direction']}",
                            f"scout: {observation['scout']}",
                            f"reward: {self.rewards[agent]:.1f}",
                            f"location: {self.agent_locations[agent]}",
                            f"action {self.num_moves}: {self.actions.get(agent)}",
                        ]
                    ):
                        self._draw_text(
                            text,
                            topright=(x_lim, y_corner + i * 15),
                        )
                except IndexError as e:
                    raise e('IndexError: Expected observation was a dictionary with direction and scout keys. If this is raised, you likely have debug mode on whilst only returning a Box observation, not a dictionary.' \
                            f'Debug: Is debug mode on? {self.debug}'
                            )

                # plot observation
                for x, y in np.ndindex((self.viewcone_length, self.viewcone_width)):
                    tile = observation["viewcone"][x, y]
                    # draw whether the tile contains points
                    Tile(tile % 4).draw(
                        self.window, x, y, subpix_square_size, x_corner, y_corner, True
                    )
                    for player in Player:
                        if not get_bit(tile, player.value):
                            continue
                        player.draw(
                            self.window, x, y, subpix_square_size, x_corner, y_corner
                        )

                for x, y in np.ndindex((self.viewcone_length, self.viewcone_width)):
                    tile = observation["viewcone"][x, y]
                    # draw walls
                    for wall in Wall:
                        if not get_bit(tile, wall.value):
                            continue
                        wall.draw(
                            self.window, x, y, subpix_square_size, x_corner, y_corner
                        )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            array = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
            # cv2.imwrite(f'/home/jovyan/interns/ben/BrainHack-TIL25/env_{self.hash}.jpg', array)
            # time.sleep()
            cv2.imshow(f'env_{self.hash}', array)
            cv2.waitKey(1)
            # return array
    
    def _capture_scout(self, capturers):
        """
        Given a list of agents who captured the scout, processes those agents' capture of the scout.
        Terminates the game and gives guards and the scout corresponding rewards.
        """
        self.logger.debug(f"{capturers} have captured the scout")
        # scout gets captured, terminate and reward guard
        self.terminations = {agent: True for agent in self.agents}
        for agent in self.agents:
            if agent == self.scout:
                self.rewards[self.scout] += self.rewards_dict.get(
                    self.reward_names.SCOUT_CAPTURED, 0
                )
                continue
            self.rewards[agent] += self.rewards_dict.get(self.reward_names.GUARD_WINS, 0)
            if agent in capturers:
                self.rewards[agent] += self.rewards_dict.get(
                    self.reward_names.GUARD_CAPTURES, 0
                )

    def _handle_agent_collision(self, agent1: AgentID, agent2: AgentID):
        """
        Given two agents, handle agent1 colliding into agent2
        """
        self.logger.debug(f"{agent1} collided with {agent2}")
        self.rewards[agent1] += self.rewards_dict.get(self.reward_names.AGENT_COLLIDER, 0)
        self.rewards[agent2] += self.rewards_dict.get(self.reward_names.AGENT_COLLIDEE, 0)

    def _handle_wall_collision(self, agent: AgentID):
        self.logger.debug(f"{agent} collided with a wall")
        self.rewards[agent] += self.rewards_dict.get(self.reward_names.WALL_COLLISION, 0)

    def _move_agent(self, agent: AgentID, action: int):
        """
        Updates agent location, accruing rewards along the way
        return the name of the agent collided into, or None
        """
        _action = Action(action)
        if _action in (Action.FORWARD, Action.BACKWARD):
            _direction = Direction(
                self.agent_directions[agent]
                if _action is Action.FORWARD
                else (self.agent_directions[agent] + 2) % 4
            )
            # enforce collisions with walls and other agents
            direction, collision = self._enforce_collisions(agent, _direction)
            # use np.clip to not leave grid
            self.agent_locations[agent] = np.clip(
                self.agent_locations[agent] + direction, 0, self.size - 1
            )
            # update scout rewards
            if agent == self.scout:
                x, y = self.agent_locations[agent]
                tile = self._state[x, y]
                match Tile(tile % 4):
                    case Tile.RECON:
                        self.rewards[self.scout] += self.rewards_dict.get(
                            self.reward_names.SCOUT_RECON, 0
                        )
                        self._state[x, y] -= Tile.RECON.value - Tile.EMPTY.value
                    case Tile.MISSION:
                        self.rewards[self.scout] += self.rewards_dict.get(
                            self.reward_names.SCOUT_MISSION, 0
                        )
                        self._state[x, y] -= Tile.MISSION.value - Tile.EMPTY.value
                    case Tile.EMPTY:
                        self.rewards[self.scout] += self.rewards_dict.get(
                            self.reward_names.SCOUT_STEP_EMPTY_TILE, 0
                        )
            if self.collisions:
                return collision
        if _action in (Action.LEFT, Action.RIGHT):
            # update direction of agent, right = +1 and left = -1 (which is equivalent to +3), mod 4.
            self.agent_directions[agent] = (
                self.agent_directions[agent] + (3 if _action is Action.LEFT else 1)
            ) % 4
            self.rewards[agent] += self.rewards_dict.get(
                self.reward_names.LOOKING, 0
            )
        if _action is (Action.STAY):
            # apply stationary penalty
            self.rewards[agent] += self.rewards_dict.get(
                self.reward_names.STATIONARY_PENALTY, 0
            )
        if agent != self.scout and not self.eval:
            # we now give guards negative rewards, based on their distance to the scout
            distance = self.get_info(agent)['euclidean']
            # if self.prev_distances[agent] is not None:
            #     diff = abs(self.prev_distances[agent] - distance)
            # else:
            #     diff = 0
            # self.prev_distances[agent] = distance
            # negative of distance differences as reward increment? 
            # self.rewards[agent] += -diff 
            self.rewards[agent] += -distance / 5

            # see if we want to give rewards based on if scout is visible?
            # print('agent_obs', self.observations[agent])
        
        return None
    
    def compute_mask(self, agent):
        # Define custom logic here
        # For example, disallow some actions based on internal state
        mask = np.ones(self.env.action_space(agent).n, dtype=np.int8)
        # e.g., disable action 2:
        mask[2] = 0
        return mask

    
    def get_info(self, agent: AgentID):
        """
        Returns accessory info for training/reward shaping.
        For now, only calculates distances between agent and scout locations,
        but does not handle for if we want to find agent and guard locations
        """
        # if agent == self.scout:

        return {
            "euclidean": np.linalg.norm(
                self.agent_locations[agent] - self.agent_locations[self.scout],
            ),
            # "manhattan": manhattan(
            #     self.agent_locations[agent],
            #     self.agent_locations[self.scout],
            # ),
        }
    
    def step(self, action: ActionType):
        """
        Takes in an action for the current agent (specified by agent_selection),
        only updating internal environment state when all actions have been received.
        """
        render_outputs = None
        if self.agent_selector.is_first():
            # clear actions from previous round
            self.actions = {}

        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent, or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.actions[self.agent_selection] = action

        # handle actions and rewards if it is the last agent to act
        if self.agent_selector.is_last():
            # execute actions
            self._handle_actions()

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            if self.num_moves >= self.num_iters:
                self.truncations = {agent: True for agent in self.agents}
                for agent in self.agents:
                    self.rewards[agent] += (
                        self.rewards_dict.get(self.reward_names.SCOUT_TRUNCATION, 0)
                        if agent == self.scout
                        else self.rewards_dict.get(self.reward_names.GUARD_TRUNCATION, 0)
                    )
            else:
                for agent in self.agents:
                    self.rewards[agent] += (
                        self.rewards_dict.get(self.reward_names.SCOUT_STEP, 0)
                        if agent == self.scout
                        else self.rewards_dict.get(self.reward_names.GUARD_STEP, 0)
                    )

            # observe the current state and get new infos
            for agent in self.agents:
                self.observations[agent] = self.observe(agent)
                # update infos
                self.infos[agent] = self.get_info(agent)

            # render
            if self.render_mode in self.metadata["render_modes"]:
                render_outputs = self.render()
        else:
            # no rewards are allocated until all players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self.agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        return render_outputs

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        # TODO: tidy ts up nga
        all_obs_spaces = {}
        if self.binary:
            # flatten, because stacking does not like 3D box.
            all_obs_spaces["viewcone"] = Box(
                            0,
                            1,
                            shape=(
                                8 * self.viewcone_length * self.viewcone_width,
                            ),
                            dtype=np.int64,
                        )
        else:
            all_obs_spaces["viewcone"] = Box(
                        0,
                        2**8 - 1,
                        shape=(
                            self.viewcone_length,
                            self.viewcone_width,
                        ),
                        dtype=np.int64,
                    )
            
        if not self.viewcone_only:
            # all values are normed / standardized to be between 0 and 1. 
            # this should help significantly stabilize training.
            all_obs_spaces["direction"] = Box(0, 1, shape=(len(Direction), ))
            all_obs_spaces["scout"] = Box(0, 1, shape=(1, ))
            all_obs_spaces["location"] = Box(0, 1, shape=(2,))
            all_obs_spaces["step"] = Box(0, 1, shape=(1,))

        thing = Dict(
            **all_obs_spaces
        )

        return thing

    def reset(self, seed=None, options=None):
        """
        Resets the environment. MUST be called before training to set up the environment.
        Automatically selects the next agent to be the Scout, and generates a new arena for each match.

        Call with `seed` to seed internal numpy RNG.
        `options` dictionary is ignored.
        """
        if self._np_random is None or seed is not None:
            self._init_random(seed)

        # As you can see, cyclical agent selection of scout has been disabled, to maintain the indexes of scout
        # i.e player_0 is always scout
        self.agents = self.possible_agents[:]
        self.agent_selector = AgentSelector(self.agents)
        self.agent_selection = self.agent_selector.next()

        # here is the original code swapping scouts around
        # select the next player to be the scout
        # self.scout: AgentID = self._scout_selector.next()
        
        # generate arena for each match for advanced track
        if self._arena is None or (self._scout_selector.is_first() and not self.novice):
            self._generate_arena()

        self._reset_state()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.actions: dict[AgentID, Action] = {}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.num_moves = 0

        if self.render_mode in self.metadata["render_modes"]:
            self.render()

    def _capture_scout(self, capturers):
        """
        Given a list of agents who captured the scout, processes those agents' capture of the scout.
        Terminates the game and gives guards and the scout corresponding rewards.
        """
        self.logger.debug(f"{capturers} have captured the scout")
        # scout gets captured, terminate and reward guard
        self.terminations = {agent: True for agent in self.agents}
        for agent in self.agents:
            if agent == self.scout:
                self.rewards[self.scout] += self.rewards_dict.get(
                    self.reward_names.SCOUT_CAPTURED, 0
                )
                continue
            self.rewards[agent] += self.rewards_dict.get(self.reward_names.GUARD_WINS, 0)
            if agent in capturers:
                self.rewards[agent] += self.rewards_dict.get(
                    self.reward_names.GUARD_CAPTURES, 0
                )
                print('CAPTURED!!!!', self.rewards[agent],
                      self.rewards_dict.get(
                    self.reward_names.GUARD_CAPTURES, 0
                ))
                # gsdfSGdFGF

    def observe(self, agent, default_return=False):
        """
        Returns the observation of the specified agent.
        """
        view = np.zeros((self.viewcone_length, self.viewcone_width), dtype=np.uint8)
        direction = Direction(self.agent_directions[agent])
        location = self.agent_locations[agent]
        for idx in np.ndindex((self.viewcone_length, self.viewcone_width)):
            view_coord = idx_to_view(np.array(idx), self.viewcone)
            world_coord = view_to_world(location, direction, view_coord)
            if not is_world_coord_valid(world_coord, self.size):
                continue
            # check if tile is visible
            if self._is_visible(agent, world_coord):
                # in theory we should filter the state to only include the visible walls, but whatever
                val = self._state[tuple(world_coord)]
                points = val % 4
                # shift orientation of the tile to local position
                view[idx] = (
                    rotate_right(val >> 4, direction.value, bit_width=4) << 4
                ) + points

        # add players
        for _agent, loc in self.agent_locations.items():
            view_coord = world_to_view(location, direction, loc)
            idx = view_to_idx(view_coord, self.viewcone)
            # check only if player is within viewcone, not whether tile is actually visible
            # this lets you "hear" nearby players without seeing them
            if is_idx_valid(idx, self.viewcone_length, self.viewcone_width):
                view[idx] += (
                    np.uint8(Player.SCOUT.power)
                    if _agent == self.scout
                    else np.uint8(Player.GUARD.power)
                )

        if self.binary and not default_return:
            view = np.unpackbits(view.astype(np.uint8))  # made R C B into (R C B).
            # because stacking does not accept 3-D box, we flatten it by default.

        if self.viewcone_only and not default_return:
            to_return =  {
                "viewcone": view,
                }
        else:
            to_return = {
                "viewcone": view,
                "direction": self.agent_directions[agent],
                "location": self.agent_locations[agent],
                "scout": 1 if agent == self.scout else 0,
                "step": self.num_moves,
            }

        if not default_return:
            to_return = self.alter_obs(to_return)

        return to_return
    
    def alter_obs(self, obs):
        """
        Utilite function to alter the observations to fit our observation space.
        This is crucial, since I refuse to alter the base behaviour of observe but want to
        format it into a model friendly stackable output.
        """
        # return integer direction as a one-hot array
        direction = obs['direction'].item()
        one_hot_dir = np.zeros(len(Direction))
        one_hot_dir[direction] = 1

        # return scout integer as a one size array (so stacker doesnt complain)
        scout = obs['scout'] # 0 or 1.
        scout = np.array([scout])

        # do the same for step
        step = obs['step']
        step = np.array([step / (self.num_iters + 1)])  # normalize.

        # normalize for location.
        location = obs['location']
        location = location / self.size

        obs['direction'] = one_hot_dir
        obs['scout'] = scout
        obs['step'] = step
        obs['location'] = location

        return obs


def frame_stack_v3(env, stack_size=4, stack_dim=-1):
    """
    Lmao why dont they support dict (stacking along each key)
    Meh f-it we ball ourselves lol

    All we need to do is, for each key in dictionary, stack the observation space along that key
    """
    assert isinstance(stack_size, int), "stack size of frame_stack must be an int"

    class FrameStackModifier(BaseModifier):
        def modify_obs_space(self, obs_space):
            self.old_obs_space = obs_space
            tmp_obs_space = {}
            if isinstance(obs_space, Dict):
                for k, o_space in obs_space.items():
                    if isinstance(o_space, Box):
                        assert (
                            1 <= len(o_space.shape) <= 3
                        ), "frame_stack only works for 1, 2 or 3 dimensional observations"
                    elif isinstance(o_space, Discrete):
                        pass
                    else:
                        assert (
                            False
                        ), "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(
                            obs_space
                        )
                    tmp_obs_space[k] = stack_obs_space(o_space, stack_size, stack_dim)

            self.observation_space = Dict(
                **tmp_obs_space
            )
            return self.observation_space

        def reset(self, seed=None, options=None):
            tmp_stack = {}
            for k, o_space in self.old_obs_space.items():
                tmp_stack[k] = stack_init(o_space, stack_size, stack_dim)


            self.stack = tmp_stack

        def modify_obs(self, obs):
            for k, stack in self.stack.items():
                o = obs[k]

                if not hasattr(obs[k], 'shape'):
                    # fixed an issue where stack obs explodes if its an integer its trying to stack
                    # technically this can be fixed in the observe function of the env, but assuming what they pass
                    # to us are just integers, do this
                    o = np.array(o)

                self.stack[k] = stack_obs(
                    stack, o, self.old_obs_space[k], stack_size, stack_dim
                )

            return self.stack

        def get_last_obs(self):
            return self.stack

    return shared_wrapper(env, FrameStackModifier)

# build_env(
#     1,
#     RewardNames,
#     {},
#     binary=True,
#     self_play=True,
#     policy_mapping=[0, 1, 2, 3],
#     agent_roles=[0, 1, 2, 3]
# )