import supersuit as ss
import ray
import argparse
import numpy as np
from independent_ppo import IndependentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch
import os
import copy
from einops import rearrange
from custom_eval_callback import CustomEvalCallback
# from til_environment.training_gridworld import env
from pettingzoo.utils.conversions import aec_to_parallel
from til_environment.types import Action, Direction, Player, RewardNames, Tile, Wall
from stable_baselines3 import DQN
from ray import tune
from ray.tune import Tuner
from ray.air import session
from ray.tune.schedulers import PopulationBasedTraining
from stable_baselines3.common.utils import configure_logger, obs_as_tensor
from modifiedppo import ModifiedPPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
    CallbackList,
    # ProgressBarCallback,
    StopTrainingOnNoModelImprovement,
)
from stablebaselines_gridworld import build_env

from enum import IntEnum, StrEnum, auto
class CustomRewardNames(StrEnum):
    GUARD_WINS = auto()
    GUARD_CAPTURES = auto()
    SCOUT_CAPTURED = auto()
    SCOUT_RECON = auto()
    SCOUT_MISSION = auto()
    WALL_COLLISION = auto()
    AGENT_COLLIDER = auto()
    AGENT_COLLIDEE = auto()
    STATIONARY_PENALTY = auto()
    GUARD_TRUNCATION = auto()
    SCOUT_TRUNCATION = auto()
    GUARD_STEP = auto()
    SCOUT_STEP = auto()
    SCOUT_STEP_EMPTY_TILE = auto()
    LOOKING = auto()
    FORWARD = auto()
    BACKWARD = auto()

STD_REWARDS_DICT = {
    CustomRewardNames.GUARD_CAPTURES: 50,
    CustomRewardNames.SCOUT_CAPTURED: -50,
    CustomRewardNames.SCOUT_RECON: 1,
    CustomRewardNames.SCOUT_MISSION: 5,
    CustomRewardNames.WALL_COLLISION: 0,
    CustomRewardNames.STATIONARY_PENALTY: 0,
    CustomRewardNames.SCOUT_STEP_EMPTY_TILE: 0,
}

GLOB_NUM_ITERS = 100
GLOB_NOVICE = True
# GLOB_ENV = 'binary_viewcone'
GLOB_ENV = 'normal'
GLOB_DEBUG = False
EXPERIMENT_NAME = 'novice_ppo_long_binaryenv_worldmodel'
root_dir = "/home/jovyan/interns/ben/BrainHack-TIL25"
# root_dir = "/mnt/e/BrainHack-TIL25"



def train(config):
    """
    Main training function. Instantiates evaluation and training environments, model, and calls learn.
    Configurations from ray tune's hyperparam search space are thrown in here via config (a dictionary.)
    """
    copy_config = copy.deepcopy(config)
    num_vec_envs = 2
    frame_stack_size = copy_config.pop('frame_stack_size')
    n_steps = copy_config.pop('n_steps')
    total_timesteps = 5_000_000
    training_iters = int(total_timesteps / n_steps)
    
    GUARD_CAPTURES = copy_config.pop('guard_captures', 50)
    SCOUT_CAPTURED = copy_config.pop('scout_captured', -50)
    SCOUT_RECON = copy_config.pop('scout_recon', 1)
    SCOUT_MISSION = copy_config.pop('scout_mission', 5)
    WALL_COLLISION = copy_config.pop('wall_collision', 0)
    STATIONARY_PENALTY = copy_config.pop('stationary_penalty', 0)
    SCOUT_STEP_EMPTY_TILE = copy_config.pop('scout_step_empty_tile', 0)
    LOOKING = copy_config.pop('looking', 0)

    num_iters = copy_config.pop('num_iters', GLOB_NUM_ITERS)
    novice = copy_config.pop('novice', GLOB_ENV)
    eval_mode = copy_config.pop('distance_penalty', False)
    REWARDS_DICT = {
        CustomRewardNames.GUARD_CAPTURES: GUARD_CAPTURES,
        CustomRewardNames.SCOUT_CAPTURED: SCOUT_CAPTURED,
        CustomRewardNames.SCOUT_RECON: SCOUT_RECON,
        CustomRewardNames.SCOUT_MISSION: SCOUT_MISSION,
        CustomRewardNames.WALL_COLLISION: WALL_COLLISION,
        CustomRewardNames.STATIONARY_PENALTY: STATIONARY_PENALTY,
        CustomRewardNames.SCOUT_STEP_EMPTY_TILE: SCOUT_STEP_EMPTY_TILE,
        CustomRewardNames.LOOKING: LOOKING,
    }
    _, train_vec_env = build_env(
        reward_names=CustomRewardNames,
        rewards_dict=REWARDS_DICT,
        render_mode=None,
        # render_mode='rgb_array',
        num_vec_envs=num_vec_envs,
        env_type=GLOB_ENV,
        novice=novice,
        num_iters=num_iters,
        eval_mode=eval_mode,
        frame_stack_size=4,)

    trial_name = session.get_trial_name()
    trial_code = trial_name[:-6]
    eval_log_path = f"{root_dir}/ppo_logs/{trial_code}/{trial_name}"
    model = ModifiedPPO(
        policy="MlpPolicy",
        n_steps=n_steps,
        env=train_vec_env,
        verbose=1,
        tensorboard_log=eval_log_path,
        **copy_config
    )
    eval_env_type = 'binary'
    # constraints: eval mode on, standard rewards dict, 1 vector env, 100 iters, novice
    _, eval_env = build_env(
        reward_names=CustomRewardNames,
        rewards_dict=STD_REWARDS_DICT,
        render_mode=None,
        # render_mode='rgb_array',
        num_vec_envs=1,
        env_type=eval_env_type,
        novice=True,
        eval_mode=True,
        num_iters=100,
        frame_stack_size=4,
    )

    # initialize callbacks 
    # this is all under the assumption that we do about 100 evaluations and checkpoints
    # per run.
    # divide by number of evals you want to run.
    num_evals = 150
    # eval_freq = int(max(training_iters / num_evals / 4, 1)) * n_steps  # cuz parallel env
    eval_freq = 1

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=f"{root_dir}/checkpoints/{trial_code}/{trial_name}",
        name_prefix=f"{EXPERIMENT_NAME}"
        )
    no_improvement = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=5,
        verbose=1
    )
    above_reward = StopTrainingOnRewardThreshold(
        reward_threshold=100.0,
        verbose=1
    )
    eval_callback = CustomEvalCallback(
        in_bits=True if eval_env_type == 'binary_viewcone' else False,  # TODO this is really bad code
        eval_freq=eval_freq,
        eval_env=eval_env,                    
        n_eval_episodes=30,
        log_path=eval_log_path,
        callback_after_eval=no_improvement,
        callback_on_new_best=above_reward,
        deterministic=False,
    )
    # progress_bar = ProgressBarCallback()

    # Combine callbacks
    callback = CallbackList([
        eval_callback,
        checkpoint_callback,
        # progress_bar,
    ])

    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback)
    # print('guards rewards:', np.unique(model.policies[0].replay_buffer.rewards, return_counts=True))
    # print('scout rewards:', np.unique(model.policies[1].replay_buffer.rewards, return_counts=True))

    save_path = f"{root_dir}/checkpoints/ppo/{trial_name}/final_ppo_model_for_run_{trial_name}"
    model.save(save_path)

    # load from the save path of the eval callback.
    # for now, just take mean of guard and scout scores instead of seperating them.
    mean_policy_scores = []
    for polid in range(2):
        path = os.path.join(eval_log_path, "evaluations", f"polid_{polid}.npz")
        thing = np.load(path)
        mean_scores = np.mean(thing['results'], axis=-1)
        max_mean_eval = np.max(mean_scores)
        mean_policy_scores.append(max_mean_eval)

    mean_guard_scores = mean_policy_scores[0] * 3
    mean_scout_scores = mean_policy_scores[1] 
    mean_all_scores = (mean_scout_scores + mean_guard_scores) / 2

    tune.report(
        dict(
            mean_guard_scores=mean_guard_scores,
            mean_scout_scores=mean_scout_scores,
            mean_all_score=mean_all_scores,
        )
    )

    # path = os.path.join(eval_log_path, "evaluations.npz")
    # thing = np.load(path)
    # mean_scores = np.mean(thing['results'], axis=-1)
    # max_mean_eval = np.max(mean_scores)

    # tune.report(
    #     dict(
    #         mean_all_score=max_mean_eval,
    #     )
    # )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", default=False, help="Simple option to train or test"
    )
    parser.add_argument(
        "--ckpt", default='checkpoints/ppo_model_960000_steps', help="Where to load from"
    )
    args, _ = parser.parse_known_args()

    if args.train:

        pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="mean_all_score",
        mode="max",
        perturbation_interval=5,  # every n trials
        hyperparam_mutations={
                "learning_rate": tune.loguniform(5e-4, 1e-3),
                "gamma": tune.uniform(0.80, 0.99),
                "n_steps": tune.choice([1024, 2048, 4096]),
                "batch_size": tune.choice([32, 64]),
                "n_epochs": tune.choice([5, 7, 10]),
                "vf_coef": tune.uniform(0.35, 0.50),
                "ent_coef": tune.loguniform(1e-6, 1e-4),
                "gae_lambda": tune.uniform(0.80, 0.99),
                "frame_stack_size": tune.choice([4, 8, 16]),
                "novice": tune.choice([True]),
                "distance_penalty": tune.choice([False]),
                "num_iters": tune.choice([100, 300, 1000]),
                "guard_captures": tune.choice([50, 200, 500]),
                "scout_captured": tune.choice([-20, -50, -200]),
                "scout_recon": tune.choice([1, 2]),
                "scout_mission": tune.choice([5, 10, 20]),
                "scout_step_empty_tile": tune.choice([-2, -1, 0]),
                "wall_collision": tune.choice([-2, -1, 0]),
                "stationary_penalty": tune.choice([-2, -1, 0]),
                "looking": tune.choice([-0.5, -0.2, 0]),
            })
        tuner = Tuner(
            tune.with_resources(train, resources={"cpu": 4}),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=1,
                reuse_actors=True,
                max_concurrent_trials=1,
            ),
            run_config=tune.RunConfig(
                name=EXPERIMENT_NAME,
                storage_path=f"{root_dir}/ray_results",
                verbose=1
            )
        )

        results = tuner.fit()
        print('best results', results)

    else:
        gridworld, vec_env = make_new_vec_gridworld(
            reward_names=CustomRewardNames,
            rewards_dict=STD_REWARDS_DICT,
            render_mode='human',
            num_vec_envs=1,
            env_type='normal',
            eval=True,
            frame_stack_size=16,
        )
        path = "/mnt/e/BrainHack-TIL25/checkpoints/" \
            "train_a3a30_00029/" \
                "novice_ppo_long_binaryenv_varyall_851968_steps"
        model = PPO.load(
            path = path
        )

        def stack_frames(past_obs, observation, observation_space):
            size = observation.shape[0]
            tile = int(observation_space.shape[0] / observation.shape[0])
            if past_obs is None:
                past_obs = np.tile(observation, tile)
            else:
                past_obs[:-size] = past_obs[size:]  # shifts all to the back
                past_obs[-size:] = observation  # adds most recent observation first

            return past_obs


        NUM_ROUNDS = 8
        rewards = {agent: 0 for agent in gridworld.possible_agents}
        for _ in range(NUM_ROUNDS):
            gridworld.reset()

            # reset frame stacks.
            past_obs = {agent: None for agent in gridworld.agents}
            for agent in gridworld.agent_iter():
                observation, reward, termination, truncation, info = gridworld.last()

                for a in gridworld.agents:
                    rewards[a] += gridworld.rewards[a]

                if termination or truncation:
                    action = None
                else:
                    if True:
                        agent_past_obs = past_obs[agent]
                        agent_past_obs = stack_frames(agent_past_obs, observation, model.observation_space)
                        past_obs[agent] = agent_past_obs

                        action, _ = model.predict(agent_past_obs, deterministic=False)

                    else:
                        action = np.random.randint(0, 5)

                gridworld.step(action)
            # past_obs = 

        gridworld.close()
        print(f"total rewards: {rewards}")
        print(f"avg rewards:", {k: reward / NUM_ROUNDS for k, reward in rewards.items()})



# from default api
# observation {'viewcone': array([[  0,   0,   0,   0,  67],
#        [  0,   0, 226, 162,   0],
#        [  0,   0, 155,  50,   0],
#        [  0,   0,   0,   0,   0],
#        [  0,   0,   0,   0,   0],
#        [  0,   0,   0,   0,   0],
#        [  0,   0,   0,   0,   0]], dtype=uint8), 'direction': np.int64(3), 'location': array([1, 8]), 'scout': 0, 'step': 21