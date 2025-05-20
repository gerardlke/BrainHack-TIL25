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

# from til_environment.training_gridworld import env
from pettingzoo.utils.conversions import aec_to_parallel
from til_environment.types import Action, Direction, Player, RewardNames, Tile, Wall
from stable_baselines3 import DQN
from ray import tune
from ray.tune import Tuner
from ray.air import session
from ray.tune.schedulers import PopulationBasedTraining
from stable_baselines3.common.utils import configure_logger, obs_as_tensor

from custom_eval_callback import CustomEvalCallback
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
GLOB_ENV = 'binary_viewcone'
# GLOB_ENV = 'normal'
GLOB_DEBUG = False
EXPERIMENT_NAME = 'novice_ppo_long_binaryenv_varyall'
root_dir = "/home/jovyan/interns/ben/BrainHack-TIL25"


def make_new_vec_gridworld(
    rewards_dict,
    reward_names,
    render_mode=None,
    env_type='normal',
    num_vec_envs=1,
    frame_stack_size=4,
    num_iters=GLOB_NUM_ITERS,
    eval=False,
    novice=True,
    ):
    """
    Helper func to build gridworld into a vectorized form. Will also return original AEC env for evaluation too, so dont worry
    """
    gridworld = build_env(
        env_type=env_type,
        env_wrappers=[],
        render_mode=render_mode,
        novice=GLOB_NOVICE,
        rewards_dict=rewards_dict,
        reward_names=reward_names,
        num_iters=num_iters,
        debug=GLOB_DEBUG,
        eval=eval,
    )

    parallel_gridworld = aec_to_parallel(gridworld)
    frame_stack = ss.frame_stack_v2(parallel_gridworld, stack_size=16, stack_dim=0)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(frame_stack)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=num_vec_envs, num_cpus=2, base_class='stable_baselines3')

    return gridworld, vec_env


def train(config):
    """
    Main training function. Instantiates evaluation and training environments, model, and calls learn.
    Configurations from ray tune's hyperparam search space are thrown in here via config (a dictionary.)
    """
    frame_stack_size = config.pop('frame_stack_size')
    n_steps = config.pop('n_steps')
    num_vec_envs = 2

    WALL_COLLISION = config.pop('wall_collision', 0)
    STATIONARY_PENALTY = config.pop('stationary_penalty', 0)
    SCOUT_STEP_EMPTY_TILE = config.pop('scout_step_empty_tile', 0)
    LOOKING = config.pop('looking', 0)

    num_iters = config.pop('num_iters', GLOB_NUM_ITERS)
    novice = config.pop('novice', GLOB_ENV)
    REWARDS_DICT = {
        CustomRewardNames.GUARD_CAPTURES: 50,
        CustomRewardNames.SCOUT_CAPTURED: -50,
        CustomRewardNames.SCOUT_RECON: 1,
        CustomRewardNames.SCOUT_MISSION: 5,
        CustomRewardNames.WALL_COLLISION: WALL_COLLISION,
        CustomRewardNames.STATIONARY_PENALTY: STATIONARY_PENALTY,
        CustomRewardNames.SCOUT_STEP_EMPTY_TILE: SCOUT_STEP_EMPTY_TILE,
        CustomRewardNames.LOOKING: LOOKING,
    }
    _, train_vec_env = make_new_vec_gridworld(
        reward_names=CustomRewardNames,
        rewards_dict=REWARDS_DICT,
        render_mode=None,
        num_vec_envs=num_vec_envs,
        env_type=GLOB_ENV,
        novice=novice,
        num_iters=num_iters,
        frame_stack_size=frame_stack_size,)

    trial_name = session.get_trial_name()
    trial_code = trial_name[:-6]
    num_agents = 4
    num_policies = 2
    print('initing model')
    model = IndependentPPO(
        policy="MlpPolicy",
        num_agents=num_agents,
        num_policies=num_policies,
        n_steps=n_steps,
        env=train_vec_env,
        verbose=1,
        tensorboard_log=f"{root_dir}/ppo_logs/{trial_code}/{trial_name}",
        **config
    )
    print('eval env made')
    _, eval_env = make_new_vec_gridworld(
        reward_names=CustomRewardNames,
        rewards_dict=STD_REWARDS_DICT,
        render_mode=None,
        num_vec_envs=1,
        env_type=GLOB_ENV,
        novice=True,
        eval=True,
        frame_stack_size=frame_stack_size,
    )
    total_timesteps=20_000_000
    # total_timesteps=1000
    eval_freq = max(total_timesteps / 100 // num_vec_envs, 1)
    print('initing callbacks')
    callbacks = [
        [
            CheckpointCallback(
                save_freq=eval_freq,
                save_path=f"{root_dir}/checkpoints/ppo_agent_{i}/{trial_name}",
                name_prefix=f"{EXPERIMENT_NAME}_novice_{GLOB_NOVICE}_run_{trial_name}"
                ),
        ] for i in range(num_policies)
    ]

    eval_log_path = f"{root_dir}/ppo_logs/{trial_name}"
    eval_callback = CustomEvalCallback(
        eval_freq=eval_freq,
        eval_env=eval_env,                    
        n_eval_episodes=20,
        log_path=eval_log_path,
        deterministic=False,
        )
    model.learn(
        total_timesteps=total_timesteps, 
        callbacks=callbacks,
        eval_callback=eval_callback)
    # print('guards rewards:', np.unique(model.policies[0].replay_buffer.rewards, return_counts=True))
    # print('scout rewards:', np.unique(model.policies[1].replay_buffer.rewards, return_counts=True))

    save_path = f"{root_dir}/checkpoints/ppo/{trial_name}/final_ppo_model_for_run_{trial_name}"
    model.save(save_path)

    # load from the save path of the eval callback.
    all_policy_max_scores = []
    for polid in range(num_policies):
        path = os.path.join(eval_log_path, 'evaluations', f"polid_{polid}.npz")
        thing = np.load(path)
        mean_scores = np.mean(thing['results'], axis=-1)
        max_score = np.max(mean_scores)
        all_policy_max_scores.append(max_score)

    mean_all_score = sum(all_policy_max_scores) / len(all_policy_max_scores)

    tune.report(
        dict(
            mean_scout_score=all_policy_max_scores[1],
            mean_guard_score=all_policy_max_scores[0],
            mean_all_score=mean_all_score,
        )
    )

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
        perturbation_interval=10,  # every n trials
        hyperparam_mutations={
                "learning_rate": tune.loguniform(5e-4, 1e-3),
                # "gamma": tune.choice([0.90, 0.99]),
                "n_steps": tune.choice([128, 2048]),
                "batch_size": tune.choice([32, 64]),
                "n_epochs": tune.choice([5, 7, 10]),
                # "vf_coef": tune.uniform(0.30, 0.70),
                # "ent_coef": tune.loguniform(1e-5, 1e-2),
                # "gae_lambda": tune.uniform(0.90, 0.99),
                "frame_stack_size": tune.choice([8, 16, 32]),
                "novice": tune.choice([False, True]),
                "num_iters": tune.choice([100, 300, 1000]),
                # "scout_step_empty_tile": tune.choice([-2, 0]),
                # "wall_collision": tune.choice([-2, 0]),
                # "stationary_penalty": tune.choice([-2, 0]),
                # "looking": tune.choice([0]),
            })
        tuner = Tuner(
            tune.with_resources(train, resources={"cpu": 10}),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=1,
                reuse_actors=True,
                max_concurrent_trials=3,
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
            rewards_dict=REWARDS_DICT,
            render_mode='human',
            num_vec_envs=1,
            env_type=GLOB_ENV,
            eval=True,
            frame_stack_size=16,
        )
        # save_path = "{root_dir}/checkpoints/dqn/train_e24cc_00000/final_dqn_model_for_run_train_e24cc_00000"
        # eval_model = IndependentPPO.load(
        #     policy="MultiInputPolicy",
        #     path=save_path,
        #     learning_starts=0,
        #     num_policies=2,
        #     buffer_size=int(1e5),
        #     num_agents=4,
        #     env=vec_env,
        #     train_freq=100 * 1,
        #     verbose=1,
        #     device='cpu',
        # )

        # # hijack collect_rollouts function to evaluate for us.
        # all_scout_score, all_guard_score = [], []
        # num_eval_rounds = 10
        # for _ in range(num_eval_rounds):
        #     total_rewards = eval_model.eval(
        #         total_timesteps=100, 
        #         progress_bar=True,
        #     )
        #     all_guard_score.append(
        #         (np.sum(np.concatenate(total_rewards[0])) / len(total_rewards[0])).item()
        #     )
        #     all_scout_score.append(
        #         (np.sum(np.concatenate(total_rewards[1])) / len(total_rewards[1])).item()
        #     )
        #     print(all_guard_score, all_scout_score)

        # mean_scout_score = sum(all_scout_score) / len(all_scout_score)
        # mean_guard_score = sum(all_guard_score) / len(all_guard_score)
        # mean_all_score = (mean_scout_score +  mean_guard_score) / 2
        # esfgzrfdthyvjb
        save_path = f"{root_dir}/checkpoints/dqn_agent_0/train_e24cc_00000/novice_dqn_long_normalenv_novice_True_run_train_e24cc_00000_100000_steps"
        eval_scout = DQN.load(
                policy="MultiInputPolicy",
                path= f"{root_dir}/checkpoints/dqn_agent_0/train_4b550_00002/novice_dqn_long_normalenv_novice_True_run_train_4b550_00002_90000_steps",
                env=vec_env,
                learning_starts=0,
                train_freq=100,
            )
        eval_guard = DQN.load(
                policy="MultiInputPolicy",
                path= f"{root_dir}/checkpoints/dqn_agent_1/train_4b550_00002/novice_dqn_long_normalenv_novice_True_run_train_4b550_00002_90000_steps",
                env=vec_env,
                learning_starts=0,
                train_freq=100,
            )
        # test out pre-rollout setup?
        for polid, policy in enumerate([eval_guard, eval_scout]):
            policy.policy.set_training_mode(False)

            if policy.use_sde:
                policy.actor.reset_noise(1)  # type: ignore[operator]

        NUM_ROUNDS = 8
        for _ in range(NUM_ROUNDS):
            gridworld.reset()
            rewards = {agent: 0 for agent in gridworld.possible_agents}

            for agent in gridworld.agent_iter():
                observation, reward, termination, truncation, info = gridworld.last()
                # print('observation before', observation)
                # observation['direction'] = np.array(observation['direction'])
                # observation['scout'] = np.array(observation['scout'])
                # observation['step'] = np.array(observation['step'])
                
                # observation = {
                #     k: obs_as_tensor(v, device='cpu') for k, v in observation.items()
                # }
                # print('observation after', observation)

                for a in gridworld.agents:
                    rewards[a] += gridworld.rewards[a]

                if termination or truncation:
                    action = None
                elif observation['scout'] == 1:
                    # run scout model
                    action, _ = eval_scout.predict(observation, deterministic=False)
                    print('scout action', action)
                elif observation['scout'] == 0:
                    action, _ = eval_guard.predict(observation, deterministic=False)
                    print('guard action', action)

                print(action)
                gridworld.step(action)

        gridworld.close()
        print(f"total rewards: {rewards}")



# from default api
# observation {'viewcone': array([[  0,   0,   0,   0,  67],
#        [  0,   0, 226, 162,   0],
#        [  0,   0, 155,  50,   0],
#        [  0,   0,   0,   0,   0],
#        [  0,   0,   0,   0,   0],
#        [  0,   0,   0,   0,   0],
#        [  0,   0,   0,   0,   0]], dtype=uint8), 'direction': np.int64(3), 'location': array([1, 8]), 'scout': 0, 'step': 21