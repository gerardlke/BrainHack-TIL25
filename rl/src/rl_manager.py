"""Manages the RL model."""
from stable_baselines3 import PPO
import numpy as np
from einops import rearrange

class RLManager:

    def __init__(self, vis=True):
        self.path = "/workspace/model_folder/novice_ppo_long_binaryenv_worldmodel_1331200_steps.zip"
        self.model = PPO.load(
            path=self.path
        )
        self.past_obs = None


    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        viewcone = np.array(observation['viewcone'])
        # print('viewcone shape???', viewcone.shape)
        
        if len(viewcone.shape) == 2:
            viewcone = np.unpackbits(viewcone.astype(np.uint8))

        # binary_obs = rearrange(viewcone, 
        #     '(C R B) -> B C R', 
        #     R=5, C=7, B=8)
        # is_scout = binary_obs[5, 2, 2]
        self.past_obs = self.stack_frames(self.past_obs, viewcone, self.model.observation_space)
        action, _ = self.model.predict(self.past_obs, deterministic=True)

        return action

    @staticmethod
    def stack_frames(past_obs, observation, observation_space):
        size = observation.shape[0]
        tile = int(observation_space.shape[0] / observation.shape[0])
        # print('tile?', tile)
        # print('size', size)
        if past_obs is None:
            past_obs = np.tile(observation, tile)
        else:
            past_obs[:-size] = past_obs[size:]  # shifts all to the back
            past_obs[-size:] = observation  # adds most recent observation first

        return past_obs