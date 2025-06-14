"""Manages the RL model."""
from stable_baselines3 import PPO
import numpy as np
import json
import torch
import inspect
import os
from copy import deepcopy
from einops import rearrange
import random
from db import RL_DB
from otherppos import ModifiedMaskedPPO, ModifiedPPO
from pathlib import Path

from supersuit.utils.frame_stack import stack_obs


class RLManager:

    def __init__(self, db_path, policy_mapping, top_opponents=2):
        # self.policy_paths = [
        #     "/mnt/e/BrainHack-TIL25/results/checkpoints/CustomTrainer_8e68e/CustomTrainer_8e68e_00666/polid_0/4_agents_450000_steps",
        #     "/mnt/e/BrainHack-TIL25/results/checkpoints/CustomTrainer_8e68e/CustomTrainer_8e68e_00666/polid_1/4_agents_500000_steps",
        #     "/mnt/e/BrainHack-TIL25/results/checkpoints/CustomTrainer_8e68e/CustomTrainer_8e68e_00666/polid_2/4_agents_150000_steps",
        #     "/mnt/e/BrainHack-TIL25/results/checkpoints/CustomTrainer_8e68e/CustomTrainer_8e68e_00666/polid_3/4_agents_350000_steps",
        # ]
        self.top_opponents = top_opponents
        self.policy_mapping = policy_mapping
        self.db_path = Path(db_path)
        print('self.db_path', self.db_path)
        self.db = RL_DB(db_file=self.db_path, num_roles=4)
        self.starting_locations = {
            'player_0': [0, 0],
            'player_1': [1, 8], # x left, y down
            'player_2': [9, 0],
            'player_3': [12, 9],
        }
        self.frame_stacks = {
            polid: None for polid in self.starting_locations
        }
        self.possible_agents = ['player_0', 'player_1', 'player_2', 'player_3']
        self.agent_policy_mapping = {
            agent: policy for agent, policy in zip(self.possible_agents, self.policy_mapping)
        }  # lookup table, to see what agent maps to what policy id.

        self.db.set_up_db(timeout=100)
        self.loaded = {
            agent: self.load_policies(
                self.db.get_checkpoint_by_role(
                    policy=self.agent_policy_mapping[agent], role=idx, shuffle=False
                )) for idx, agent
            in enumerate(self.possible_agents)
        }  # load all policies at the start and then just select them later
        self.db.shut_down_db()

        self.loaded_policies = {
            agent: loads[0] for agent, loads
            in self.loaded.items()
        }
        self.loaded_desc = {
            agent: loads[1] for agent, loads
            in self.loaded.items()
        }
        self.episode_policies = {}  # temporary dict to index what policies are available for use each episode.
        self.episode_policies_desc = {}
        self.choose_policies()
        self.has_first_step = False
        self.past_obs = None
        self.curr_policy = None

    def choose_policies(self):
        """
        Connects to database and recieves information on what opponent policies exist.
        Loading policies has two scenarios, one where eval_mode is False and another when it is True.

        When not evaluating, we will randomly pick an opponent.
        When in evaluation mode, we sequentially iterate over all.

        Return:
            - list of selected opponents, length equal to the number of policies we control
        """
        for agent, policies in self.loaded_policies.items():
            random_policy = random.choice(policies)
            idx = policies.index(random_policy)
            self.episode_policies[agent] = random_policy
            self.episode_policies_desc[agent] = self.loaded_desc[agent][idx]

    def load_policies(self, checkpoints: list):
        all_loaded = []
        all_desc = []  # purely for display now
        if len(checkpoints) == 0:
            # if no checkpoint, default to random behaviour.
            [all_loaded.append('random') for _ in range(self.top_opponents)]
            [all_desc.append('random') for _ in range(self.top_opponents)]
        else:
            for checkpoint in checkpoints[:self.top_opponents]:
                hyperparams = json.loads(checkpoint['hyperparameters'])
                filepath = checkpoint['filepath']
                # hacky fix for different systems. should have saved with a root directory
                # column. damn.
                print('dict checkpoints', dict(checkpoint))
                filepath = filepath.split('checkpoints/')
                filepath[0] = self.db_path.parent
                filepath = filepath[0] / 'checkpoints' / filepath[1]

                _checkpoint = deepcopy(dict(checkpoint))
                all_desc.append(_checkpoint)

                algo_type = eval(hyperparams['algorithm'])
                hyperparams.pop('algorithm')
                policy_type = hyperparams['policy']
                
                policy = algo_type.load(
                    path=filepath,
                    policy=policy_type,
                )
                all_loaded.append(policy)
                
        return all_loaded, all_desc

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.

        ## CURRENTLY ONLY SUPPORTS ONE AGENT AT A TIME, BEFORE ENVIRONMENT IS RESET!!!
        """
        if not self.has_first_step:
            # determine which model is to run based on location. very hardcoded. very based.
            ini_location = observation['location']
            self.agent = next((k for k, v in self.starting_locations.items() if v == ini_location))
            self.curr_policy = self.episode_policies[self.agent]
            self.curr_policy_stacksize = 16
            self.has_first_step = True
    
        observation = self.alter_obs(observation)

        past_obs = self.frame_stacks[self.agent]
        past_obs = self.stack_frames(past_obs, observation, self.curr_policy.observation_space)
        self.frame_stacks[self.agent] = past_obs
        
        with torch.no_grad():
            if 'action_masks' not in inspect.signature(self.curr_policy.policy.predict).parameters:
                action, _ = self.curr_policy.policy.predict(past_obs, deterministic=False)
            else:
                action_masks = self.curr_policy.get_action_masks(past_obs)

                action, _ = self.curr_policy.policy.predict(past_obs,
                        action_masks=action_masks, deterministic=False)

        return action

    def stack_frames(self, past_obs, observation, observation_space):
        size = self.curr_policy_stacksize

        if past_obs is None:
            past_obs = {}
            for k, o_space in observation_space.items():
                past_obs[k] = np.zeros(o_space.shape, dtype=o_space.dtype)
        
        for k, stack in past_obs.items():
            o = observation[k]
            past_obs[k] = stack_obs(
                    stack, o, observation_space[k], size, stack_dim=0
                )

        return past_obs
    
    def reset(self):
        # clear all frame stacks.
        self.frame_stacks = {
            polid: None for polid in self.starting_locations
        }
        self.curr_policy = None
        self.has_first_step = False

        self.choose_policies()
        

    def alter_obs(self, obs):
        """
        Utilite function to alter the observations to fit our observation space.
        This is crucial, since I refuse to alter the base behaviour of observe but want to
        format it into a model friendly stackable output.
        """
        # return integer direction as a one-hot array
        viewcone = np.array(obs['viewcone'])
        if len(viewcone.shape) == 2:
            viewcone = np.unpackbits(viewcone.astype(np.uint8))
    
        direction = obs['direction']
        one_hot_dir = np.zeros(4)
        one_hot_dir[direction] = 1

        # return scout integer as a one size array (so stacker doesnt complain)
        scout = obs['scout'] # 0 or 1.
        scout = np.array([scout])

        # do the same for step
        step = obs['step']
        step = np.array([step / (100 + 1)])  # normalize.

        # normalize for location.
        location = obs['location']
        location = np.array([loc / 16 for loc in location])

        obs['direction'] = one_hot_dir
        obs['scout'] = scout
        obs['step'] = step
        obs['location'] = location
        obs['viewcone'] = viewcone

        return obs
