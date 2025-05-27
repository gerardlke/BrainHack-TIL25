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

from functools import partial
from pettingzoo.utils.conversions import aec_to_parallel
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
    num_vec_envs,
    reward_names,
    rewards_dict,
    binary,
    use_action_masking=True,
    env_wrappers: list[BaseWrapper] | None = None,
    render_mode: str | None = None,
    eval_mode: bool = False,
    novice: bool = True,
    debug: bool = False,
    frame_stack_size: int = 4,
    **kwargs,
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
    env = modified_env(
        render_mode=render_mode,
        reward_names=reward_names,
        rewards_dict=rewards_dict,
        eval=eval_mode,
        novice=novice,
        debug=debug,
        use_action_masking=use_action_masking,
        binary=binary,
        **kwargs
    )
    # if use_action_masking:
    #     env = ActionMaskWrapper(env)
    if env_wrappers is None:
        env_wrappers = [
            FlattenDictWrapper,
        ]
        print('YES USING FLATTENDICTWRAPPER')
    else:
        raise AssertionError('not using flatten dict, this behaviour is unexpected')
    for wrapper in env_wrappers:
        env = wrapper(env)  # type: ignore
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    
    parallel_env = aec_to_parallel(env)
    parallel_env = ss.frame_stack_v2(parallel_env, stack_size=frame_stack_size, stack_dim=0)
    if use_action_masking:
        assert num_vec_envs == 1, 'Assertion failed. Due to the particular way of calling environment custom '\
            'functions when masking actions, we disable vectorized environments for action masking. '\
                'You may still compensate for this by parallelizing across multiple trials with different '\
                    'hyperparameters.'
        return env, parallel_env

    vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=num_vec_envs, num_cpus=4, base_class='stable_baselines3')
    # return env, vec_env
    return env, vec_env


"""
TODO: things to try for our environments:
1. Negative reward if agents run into walls or stays (incentivises non trivial exploration)
2. Guide guards using distance from scout as a reward
3. See if you can exploit distance from scout into some latent state of the model (recurrentPPO?)
"""


class modified_env(raw_env):
    """
    yoink provided env and add some stuff to it

    Base, standard training env. We disable different agent selection as we would like to fix the indexes of
    the scout and guards, so that we can more easily extract it and parse it into seperate policies.

    Also supports action masking, just does not return it in observation space.
    """
    def __init__(self,
            reward_names,
            rewards_dict,
            binary,
            use_action_masking=True,
            num_iters=1000,
            eval=False, 
            collisions=True,
            viewcone_only=False,
            see_scout_reward=True,
        **kwargs):
        # self.rewards_dict is defined in super init call
        super().__init__(**kwargs)
        self.binary = binary
        self.use_action_masking = use_action_masking
        if self.use_action_masking:
            print('-----------------------------------------')
            print('---------ACTION MASKING ENABLED----------')
            print('-----------------------------------------')
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
                observation = self.observe(agent)

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
        """
        We take the concept of looking left or right as an effective 'step left or right', since that would be the next most efficient
        move for the agent looking left or right. (if it were going backwards, may as well look the other direction and step forward)
        0. For all agents
            - If there is a wall in front or behind you, disable the corresponding action.
            - If there is a wall beside you, disable looking left/right correspondingly.
            - If there are no other valid actions, enable 'do-nothing'. Else, do-nothing is default disabled.


        Possible action masks, if we decide that they need more support, else we would rather refuse to hardcode these behaviours
        and let them and other strategies emerge naturally.
        1. Scout
            - If guard is 1-2 blocks in front or behind you, only can move back / forward to imminently escape.
            - If guard is 1-2 blocks beside you, disable looking left/right correspondingly, so as to imminently escape.
            - If guard is directly diagonal, disable the corresponding directions (e.g top left, disable forward and right)
        2. 
        
        """
        # Define custom logic here
        print('agent', agent)
        # if self.scout == agent:

        # mask = np.ones(self.action_space(agent).n, dtype=np.int8)


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
            all_obs_spaces["viewcone"] = Box(
                            0,
                            1,
                            shape=(
                                8,  # hardcode lol
                                self.viewcone_length,
                                self.viewcone_width,
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
            all_obs_spaces["direction"] = Discrete(len(Direction))
            all_obs_spaces["scout"] = Discrete(2)
            all_obs_spaces["location"] = Box(0, self.size, shape=(2,), dtype=np.int64)
            all_obs_spaces["step"] = Box(0, self.num_iters + 1, shape=(1,))

        # if self.use_action_masking:
        #     all_obs_spaces["action_mask"] = Box(0, 1, shape=(len(Action), ), dtype=np.int64)

        return Dict(
            **all_obs_spaces
        )

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

    def observe(self, agent):
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

        if self.binary:
            view = np.unpackbits(view.astype(np.uint8))  # made R C B into (R C B)
        
        if self.viewcone_only:
            return {
                "viewcone": view,
                }

        return {
            "viewcone": view,
            "direction": self.agent_directions[agent],
            "location": self.agent_locations[agent],
            "scout": 1 if agent == self.scout else 0,
            "step": self.num_moves,
        }

# depreciated code for now.
# def frame_stack_v3(env, stack_size=4, stack_dim=-1):
#     """
#     Lmao why dont they support dict (stacking along each key)
#     Meh f-it we ball ourselves lol

#     All we need to do is, for each key in dictionary, stack the observation space along that key
#     """
#     assert isinstance(stack_size, int), "stack size of frame_stack must be an int"

#     class FrameStackModifier(BaseModifier):
#         def modify_obs_space(self, obs_space):
#             print('OBS_SPACE IN FRAME STACK MODIFIER', obs_space)
#             self.old_obs_space = obs_space
#             tmp_obs_space = {}
#             if isinstance(obs_space, Dict):
#                 for k, o_space in obs_space.items():
#                     if isinstance(o_space, Box):
#                         assert (
#                             1 <= len(o_space.shape) <= 3
#                         ), "frame_stack only works for 1, 2 or 3 dimensional observations"
#                     elif isinstance(o_space, Discrete):
#                         pass
#                     else:
#                         assert (
#                             False
#                         ), "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(
#                             obs_space
#                         )
#                     tmp_obs_space[k] = stack_obs_space(o_space, stack_size, stack_dim)

#             self.observation_space = Dict(
#                 **tmp_obs_space
#             )
#             print('self.new obs space', self.observation_space)
#             return self.observation_space

#         def reset(self, seed=None, options=None):
#             tmp_stack = {}
#             for k, o_space in self.old_obs_space.items():
#                 tmp_stack[k] = stack_init(o_space, stack_size, stack_dim)

#             self.stack = tmp_stack
#             print('self.stack', self.stack)

#         def modify_obs(self, obs):
#             for k, stack in self.stack.items():
#                 self.stack[k] = stack_obs(
#                     stack, obs, self.old_obs_space[k], stack_size, stack_dim
#                 )

#             return self.stack

#         def get_last_obs(self):
#             return self.stack

#     return shared_wrapper(env, FrameStackModifier)

