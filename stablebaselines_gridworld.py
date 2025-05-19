"""
Yoink from gridworld.py, but inherit gynasium.Env to integrate with ray rllib
"""
import functools
import logging
import warnings
from functools import partial
import os
import hashlib
import cv2
import time

import gymnasium
import numpy as np
import pygame
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

from til_environment.gridworld import raw_env


def build_env(
    reward_names,
    env_wrappers: list[BaseWrapper] | None = None,
    render_mode: str | None = None,
    env_type: str = 'normal',
    **kwargs,
):
    """
    Main entrypoint to the environment, allowing configuration of render mode
    and what wrappers to wrap around the environment. If you write a custom
    wrapper(s), pass them in a list to `env_wrappers`.
    See `flatten_dict.FlattenDictWrapper` for a very simple wrapper example.
    """
    if env_type == 'normal':
        to_build = normal_env
    elif env_type == 'binary_viewcone':
        to_build = binary_viewcone_env
    else:
        assert AssertionError('not accepted env_type.')
    print('env_type', env_type)
    env = to_build(render_mode=render_mode, reward_names=reward_names, **kwargs)
    print('ENV NOVICE??????????????', env.novice)
    if env_wrappers is None:
        env_wrappers = [
            FlattenDictWrapper,
            partial(frame_stack_v2, stack_size=4, stack_dim=-1),
        ]
    for wrapper in env_wrappers:
        env = wrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env

"""
TODO: things to try for our environments:
1. Negative reward if agents run into walls or stays (incentivises non trivial exploration)
2. Guide guards using distance from scout as a reward
3. See if you can exploit distance from scout into some latent state of the model (recurrentPPO?)
"""


class normal_env(raw_env):
    """
    yoink provided env and add some stuff to it

    Base, standard training env.
    """
    def __init__(self, reward_names, num_iters=1000, **kwargs):
        # self.rewards_dict is defined in super init call
        super().__init__(**kwargs)
        self.num_iters = num_iters
        self.reward_names = reward_names
        # Generate 32 bytes of random data
        random_bytes = os.urandom(32)
        # Hash it with SHA-256
        self.hash = hashlib.sha256(random_bytes).hexdigest()

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
            # cv2.imshow(f'env_{self.hash}', array)
            # cv2.waitKey(1)
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
                        print('THIS SHOULD BE ZERO DURING EVAL!!!!!!!!')
                        print(self.rewards_dict.get(
                            self.reward_names.SCOUT_STEP_EMPTY_TILE, 0
                        ))
                        self.rewards[self.scout] += self.rewards_dict.get(
                            self.reward_names.SCOUT_STEP_EMPTY_TILE, 0
                        )
        if _action in (Action.LEFT, Action.RIGHT):
            # update direction of agent, right = +1 and left = -1 (which is equivalent to +3), mod 4.
            self.agent_directions[agent] = (
                self.agent_directions[agent] + (3 if _action is Action.LEFT else 1)
            ) % 4
        if _action is (Action.STAY):
            # apply stationary penalty
            self.rewards[agent] += self.rewards_dict.get(
                self.reward_names.STATIONARY_PENALTY, 0
            )
        # if agent != self.scout:
        #     # we now give guards negative rewards, based on their distance to the scout
        #     distance = self.get_info(agent)['euclidean']
        #     # negative of distance as reward increment? 
        #     self.rewards[agent] += -distance / 5  # probably divide it so the poor guard doesnt explode

        return None
    
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
                self.render()
        else:
            # no rewards are allocated until all players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self.agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        # return Dict(
        #     {
        #         "viewcone": Box(
        #             0,
        #             2**8 - 1,
        #             shape=(
        #                 self.viewcone_length,
        #                 self.viewcone_width,
        #             ),
        #             dtype=np.int64,
        #         ),
        #         "direction": Discrete(len(Direction)),
        #         "scout": Discrete(2),
        #         "location": Box(0, self.size, shape=(2,), dtype=np.int64),
        #         "step": Discrete(self.num_iters),
        #     }
        # )
        return Box(
                    0,
                    2**8 - 1,
                    shape=(
                        self.viewcone_length,
                        self.viewcone_width,
                    ),
                    dtype=np.int64,
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

        # agent_selector utility cyclically steps through agents list
        self.agents = self.possible_agents[:]
        self.agent_selector = AgentSelector(self.agents)
        self.agent_selection = self.agent_selector.next()
        # select the next player to be the scout
        self.scout: AgentID = self._scout_selector.next()
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
        # run super method, but prune to only return the viewcone observation.
        # this will break rendering for now.
        observations = super().observe(agent)
        # print('obs', observations)
        # print('returning viewcone', observations['viewcone'])
        return observations['viewcone']
        
class binary_viewcone_env(normal_env):
    # yoink brainhack code but change some tings
    """
    ok so now make it into a 3d box where the viewcone is binarized
    """
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        return Dict(
            {
                "viewcone": Box(
                    0,
                    1,
                    shape=(
                        8,  # hardcode lol
                        self.viewcone_length,
                        self.viewcone_width,
                    ),
                    dtype=np.int64,
                ),
                "direction": Discrete(len(Direction)),
                "scout": Discrete(2),
                "location": Box(0, self.size, shape=(2,), dtype=np.int64),
                "step": Discrete(self.num_iters),
            }
        )

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

        bit_planes = np.unpackbits(view.astype(np.uint8))
        bit_planes = rearrange(bit_planes, 
            '(R C B) -> B C R', 
            R=self.viewcone_width, C=self.viewcone_length, B=8)

        return {
            "viewcone": bit_planes,
            "direction": self.agent_directions[agent],
            "location": self.agent_locations[agent],
            "scout": 1 if agent == self.scout else 0,
            "step": self.num_moves,
        }
    