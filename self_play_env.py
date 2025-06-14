"""
Yoink from gridworld.py, but mutates with a whole host of additional functionalities
"""
import random
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
import hashlib
import secrets
import json
import torch

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
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper
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

from otherppos import ModifiedMaskedPPO, ModifiedPPO

from stable_baselines3.common.vec_env import DummyVecEnv

class DummyGymEnv(gymnasium.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


def build_env(
    reward_names,
    rewards_dict,
    policy_mapping,
    agent_roles,
    self_play,
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
    top_opponents  = _env_config.pop('top_opponents', 1)
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

    env = aec_to_parallel(env)
    env = frame_stack_v3(env, stack_size=frame_stack_size, stack_dim=0)
    if self_play:
        assert db_path is not None, 'Assertion failed. We require a path to database of network pools.'
        env = SelfPlayWrapper(env,
            policy_mapping=policy_mapping,
            agent_roles=agent_roles,
            top_opponents=top_opponents,
            db_path=db_path,
        )
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    print('before concat vec envs')
    print('num_vec_envs', num_vec_envs)
    # more cpus = more parallelism, but higher mem consumption. Honestly the thing is quite fast already
    # so screw parallelism this got me held up for a whole weekend figuring where the memory issue was
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=1, base_class='stable_baselines3')
    print('after concat vec envs')
    return orig_env, env

class SelfPlayWrapper(BaseParallelWrapper):

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

    Now onto choosing policies:
    Choosing policies has two scenarios, one where eval_mode is False and another when it is True.
    When not evaluating, we will randomly pick an opponent.
    When in evaluation mode, there are several changes we must enforce.
    -> Despite the (quite bad seperation of) between responsibilities of env and simulator, we must at this scope
        consider an evaluation episode incomplete, if we have not played all permutations of best opponents.
        Hence, forcibly mutate done=1 to done=0 when it is set to 1, and only return done=1 when we have ran the complete
        gauntlet.
    -> Opponent selection cannot be random, but rather sequentially permute each, resulting in complexity O(c^m), where c is
        constant and m is number of unique opponent policies.

    Args:
        - env: Instantiated env
        - policy_mapping: Mapping of policies to agents. This is because our network pool
            technically encompasses policies, not agents, and the mapping from policy to agents
            is defined in config (which is passed here.)
        - agent_roles: Indexes of each agent. This is used for db setup and retrieval, and how we concatenate
            the final actions before calling step.
        - npcs: Defined at initialization, what agents will be controlled by the environment. Requires this so
            it knows what policies to load, which it can then map to which agent roles.
        DB.
        - db_path: Path to gerard db
    """
    def __init__(self,
        env,
        policy_mapping,
        agent_roles,
        db_path,
        top_opponents
    ):
        super().__init__(env)
        self.db_path = db_path
        # db interface (gerard)
        self.db = RL_DB(db_file=self.db_path, num_roles=len(agent_roles))

        self.policy_mapping = policy_mapping
        self.agent_roles = agent_roles
        # self.opponent_sampling = opponent_sampling
        # somethign about loading best models here

        # self.environment_policies = [policy for policy, keep in zip(self.policy_mapping, self.npcs) if keep]
        # self.environment_agents = [agent for agent, keep in zip(self.possible_agents, self.npcs) if keep]
        self.agent_policy_mapping = {
            agent: policy for agent, policy in zip(self.possible_agents, self.policy_mapping)
        }  # lookup table, to see what agent maps to what policy id.

        self.loaded_policies = None
        self.db.set_up_db(timeout=100)

        # get all the currently best policies.
        self.top_opponents = top_opponents  # temporarily hardcode to 1 rn for debugging
        self.loaded = {
            agent: self.load_policies(
                self.db.get_checkpoint_by_role(
                    policy=self.agent_policy_mapping[agent], role=idx, shuffle=False
                )) for idx, agent
            in enumerate(self.possible_agents)
        }  # load all policies at the start and then just select them later
        # print('done building self.loaded')
        self.loaded_policies = {
            agent: loads[0] for agent, loads
            in self.loaded.items()
        }
        self.loaded_desc = {
            agent: loads[1] for agent, loads
            in self.loaded.items()
        }
        # print('self.loaded_desc', self.loaded_desc)
        self.db.shut_down_db()
        del self.db  # cant subprocess sqlite connection in vectorized env
        self.do_reset_policy_pointers = True
        self.eval = self.aec_env.eval
        self.deterministic = False
        self.do_eval_reset = False

        self.episode_policies = {}  # temporary dict to index what policies are available for use each episode.
        # reset when reset is called.
        self.episode_rewards = defaultdict(float)  # dictionary to store the current episode rewards of each agent.

        # ----------------------- DEPRECEATED (BUT NOT DELETED) --------------------------
        # Generate a random 32-byte (256-bit) value
        # random_bytes = secrets.token_bytes(32)

        # # Hash it using SHA-256
        # random_hash = hashlib.sha256(random_bytes).hexdigest()

        # self.hash = random_hash
    
    def load_policies(self, checkpoints: list):
        all_loaded = []
        all_desc = []  # purely for display now
        # print('checkpoints', checkpoints, len(checkpoints))
        # print('self.top_opponents', self.top_opponents)
        if len(checkpoints) == 0:
            # if no checkpoint, default to random behaviour.
            [all_loaded.append('random') for _ in range(self.top_opponents)]
            [all_desc.append('random') for _ in range(self.top_opponents)]
        else:
            for checkpoint in checkpoints[:self.top_opponents]:
                hyperparams = json.loads(checkpoint['hyperparameters'])
                filepath = checkpoint['filepath']

                _checkpoint = deepcopy(dict(checkpoint))
                all_desc.append(_checkpoint)

                algo_type = eval(hyperparams['algorithm'])
                hyperparams.pop('algorithm')
                policy_type = hyperparams['policy']
                print('loading policy...............')
                policy = algo_type.load(
                    path=filepath,
                    policy=policy_type,
                )
                all_loaded.append(policy)
                
        return all_loaded, all_desc
    
    def reset_and_enable_choose(self, env_agents):
        """
        Calls reset, then enables choose to be called.
        """
        self.reset()
        self.choose_policies(env_agents)

    def choose_policies(self, env_agents):
        """
        Connects to database and recieves information on what opponent policies exist.
        Loading policies has two scenarios, one where eval_mode is False and another when it is True.

        When not evaluating, we will randomly pick an opponent.
        When in evaluation mode, we sequentially iterate over all.

        Return:
            - list of selected opponents, length equal to the number of policies we control
        """
        if not self.eval:
            for agent, policies in self.loaded_policies.items():
                random_policy = random.choice(policies)
                self.episode_policies[agent] = random_policy

        else:
            self.reset_policy_pointers(env_agents)
            # select episode policies based on policy pointers' indexes.
            for agent, policy_intraidx in self.policy_pointers.items():
                policies = self.loaded_policies[agent]
                self.episode_policies[agent] = policies[policy_intraidx]

            # now merry go round pointer time
            for agent, policy_intraidx in self.policy_pointers.items():
                policies = self.loaded_policies[agent]
                if self.policy_pointers[agent] != len(policies) - 1:
                    self.policy_pointers[agent] += 1
                    break
                else:   # e.g top opponents 3, we only have until index 2 to pick, so reset for that polid
                    self.policy_pointers[agent] = 0
                    # if we reset for final pointer, its joever, reset all
                    # print('curr agent', agent)
                    # print('index of curr', env_agents.index(agent))
                    # print('len(env_agents) - 1', len(env_agents) - 1)
                    if env_agents.index(agent) == len(env_agents) - 1:
                        self.do_eval_reset = True
                        self.do_reset_policy_pointers = True

    def reset_policy_pointers(self, env_agents):
        # Helper func to reset policy pointers 
        if self.do_reset_policy_pointers:
            self.policy_pointers = {agent: 0 for agent in self.possible_agents if agent in env_agents}
            self.do_reset_policy_pointers = False

    def reset(self, seed: int | None=None, options: dict | None=None):  # type: ignore
        """
        Wraps around super class reset.
        Will connect to DB and update its score values, for each policy that is being evaluated.
        Uses a system of 'score difference times 10%' to change the score of the policy.
        """
        self.has_reset = True

        if not hasattr(self, 'db'):  # setup again because we dont need to deal with multiproc
            self.db = RL_DB(db_file=self.db_path, num_roles=len(self.agent_roles))

        # print(any(episode_reward > 0 for episode_reward in list(self.episode_rewards.values())))
        # if we have nonzero rewards, and if we are in evaluation mode (where we are truly grading the policies)
        if any(episode_reward > 0 for episode_reward in list(self.episode_rewards.values())) and self.eval:
            self.db.set_up_db(timeout=100)
            # print('self.episode_rewards', self.episode_rewards)
            for agent, episode_reward in self.episode_rewards.items():
                if agent in self.policy_pointers:

                    policies_desc = self.loaded_desc[agent]
                    selected_policy_idx = self.policy_pointers[agent]
                    policy_desc = policies_desc[selected_policy_idx]

                    if not isinstance(policy_desc, str):  # make sure it is a dict and not 'random'.
                        # find its unique id, and the score that was relevant in its choice
                        checkpoint = self.db.get_checkpoint_by_id(id=policy_desc['id'])[0]
                        role_idx = self.possible_agents.index(agent)
                        prev_score = checkpoint[f'score_{role_idx}'] 
                        # TODO: update how we update scores to be score-specific id
                        diff = episode_reward - prev_score
                        prev_score += diff * 0.05  # completely agar agar only. lmao
                        self.db.update_score(prev_score, role=role_idx, id=policy_desc['id'])
                        # checkpoints = self.db.get_checkpoint_by_id(id=policy_desc['id'])
                        # checkpoint = checkpoints[0]
                        # print('thing after update', checkpoint)
                        # print('new score', checkpoint['score'] )
            
            self.db.shut_down_db()

        self.episode_rewards = defaultdict(float)

        return super().reset(seed, options)


    def step(self, actions):
        """
        Although we support many to one mapping of agents to policies, during stepping we will manually loop through all NPC agents
        and run their relevant policy on the relevant agent observation. 

        actions are a dict of agent names to their actions. Since this is nested within the vectorized wrapper,
        expect exactly what the base environment's action space tells us, without worrying about batches etc.

        Serious, significant NOTE: During evaluation, we want to evaluate on ALL best opponent checkpoints. The problem is, the callback / simulator
        doesn't handle policy loading, we do. However, they handle the looping and stopping of evaluation. Because of constrained supersuit vector env 
        api things, communciation of when to start and stop isn't possible without a lot of customization which we want to avoid.
        Thus, mutate stepping.

        We opt to not post done=1 in our returns, until ALL best opponents have been evaluated against. This means that when stepping, if
        the underlying env tells us done=1, we check for remaining opponents to evaluate against. Once we have no more, then set done=1.
        
        We also have to update ELO of each policy within this function, since the knowledge of what policies
        are being evaluated is contained within the environment. This feels wrong but oh well, what todo.
        """
        # first, look at actions. For agent ids with -1, those indexes are meant to be controlled by us.
        env_agents = [k for k, v in actions.items() if v.item() is -1]
        
        # check if we have chosen our policies yet. self.choose_policies should be called only after every reset.
        if self.has_reset:
            self.has_reset = False  
            self.choose_policies(env_agents)
            # print('do we truly reset', self.do_eval_reset)

        for agent, action in actions.items():
            if agent in env_agents:
                loaded_policy = self.episode_policies[agent]

                if loaded_policy == 'random':
                    action = np.random.randint(0, 5)
                else:
                    obs = self.modifiers[agent].get_last_obs()

                    with torch.no_grad():
                        if 'action_masks' not in inspect.signature(loaded_policy.policy.forward).parameters:
                            action, _ = loaded_policy.policy.predict(obs, deterministic=self.deterministic)
                        else:
                            action_masks = loaded_policy.get_action_masks(obs)

                            action, _ = loaded_policy.policy.predict(obs, action_masks=action_masks, deterministic=self.deterministic)

                if isinstance(action, torch.Tensor):
                    action = action.item()
        
            actions[agent] = action

        # i have no clue why there are two dones, but wokie
        observations, rewards, terms, truncs, infos = super().step(actions)  # type: ignore

        for agent, reward in rewards.items():
            self.episode_rewards[agent] += reward

        # intercept terms/truncs. if any True, it means we are doing a reset. choose policies first, to see if we should do a true reset.
        # i.e TRULY return True to the vector env wrapper around this, thereby returning True to the simulator or callback.
        if any(list(terms.values())) or any(list(truncs.values())):
            self.reset()  # within this is where self.do_eval_reset is called.
            if self.eval:  # first if eval mode, set terminations n truncations to False.
                terms = {k: False for k in terms}
                truncs = {k: False for k in truncs}
                if self.do_eval_reset:  # then check for and do eval reset.
                    terms = {k: True for k in terms}  # commit sewer side
                    truncs = {k: True for k in truncs}

                    self.do_eval_reset = False
        # time.sleep(0.2)
        return observations, rewards, terms, truncs, infos


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
            self.rewards[agent] += -distance * 0.05

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
            self.generate_arena()

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