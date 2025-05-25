"""Manages the RL model."""
from stable_baselines3 import PPO
import numpy as np
from einops import rearrange

class RLManager:

    def __init__(self, vis=True):
        # self.policy_paths = [
        #     "/mnt/e/BrainHack-TIL25/results/checkpoints/CustomTrainer_8e68e/CustomTrainer_8e68e_00666/polid_0/4_agents_450000_steps",
        #     "/mnt/e/BrainHack-TIL25/results/checkpoints/CustomTrainer_8e68e/CustomTrainer_8e68e_00666/polid_1/4_agents_500000_steps",
        #     "/mnt/e/BrainHack-TIL25/results/checkpoints/CustomTrainer_8e68e/CustomTrainer_8e68e_00666/polid_2/4_agents_150000_steps",
        #     "/mnt/e/BrainHack-TIL25/results/checkpoints/CustomTrainer_8e68e/CustomTrainer_8e68e_00666/polid_3/4_agents_350000_steps",
        # ]
        self.policy_paths = [
            "/workspace/model_folder/polid_0/4_agents_450000_steps",
            "/workspace/model_folder/polid_1/4_agents_500000_steps",
            "/workspace/model_folder/polid_2/4_agents_150000_steps",
            "/workspace/model_folder/polid_3/4_agents_350000_steps",
        ]
        self.starting_locations = [
            [0, 0],
            [1, 8],  # x left, y down
            [9, 0],
            [12, 9],
        ]
        self.frame_stacks = [
            None for _ in self.starting_locations
        ]
        self.policies = [
            PPO.load(path=path) for path in self.policy_paths
        ]
        self.has_first_step = False
        self.past_obs = None
        self.curr_policy = None


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
            if ini_location not in self.starting_locations:
                ## explode for now
                fsdghds
            else:
                self.polid = self.starting_locations.index(ini_location)
                self.curr_policy = self.policies[self.polid]
            self.has_first_step = True
    
        viewcone = np.array(observation['viewcone'])
        # print('viewcone shape???', viewcone.shape)
        
        if len(viewcone.shape) == 2:
            viewcone = np.unpackbits(viewcone.astype(np.uint8))
    
        # we expect viewcone only, flattened
        past_obs = self.frame_stacks[self.polid]
        past_obs = self.stack_frames(past_obs, viewcone, self.curr_policy.observation_space)
        action, _ = self.curr_policy.predict(past_obs, deterministic=False)

        return action

    @staticmethod
    def stack_frames(past_obs, observation, observation_space):
        size = observation.shape[0]
        tile = int(observation_space.shape[0] / observation.shape[0])

        if past_obs is None:
            past_obs = np.tile(observation, tile)
        else:
            past_obs[:-size] = past_obs[size:]  # shifts all to the back
            past_obs[-size:] = observation  # adds most recent observation first

        return past_obs
    
    def reset(self):
        # clear all frame stacks.
        self.frame_stacks = [
            None for _ in self.starting_locations
        ]
        self.curr_policy = None
        self.has_first_step = False
