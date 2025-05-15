import supersuit as ss
import ray
import argparse
import numpy as np
from independent_recurrent_ppo import IndependentRecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from til_environment.stablebaselines_gridworld import build_env
# from til_environment.training_gridworld import env
from pettingzoo.utils.conversions import aec_to_parallel

from ray import tune
from ray.tune import Tuner
from ray.air import session
from ray.tune.schedulers import PopulationBasedTraining

GLOB_NOVICE = True
GLOB_ENV = 'normal'
EXPERIMENT_NAME = 'novice_long_test_dqn'

def make_new_vec_gridworld(render_mode=None, env_type='normal', num_vec_envs=1):
    """
    Helper func to build gridworld into a vectorized form. Will also return original AEC env for evaluation too, so dont worry
    """
    gridworld = build_env(
        env_type=env_type,
        env_wrappers=[],
        render_mode=render_mode,
        novice=GLOB_NOVICE,
    )
    # gridworld = env(
    #     env_wrappers=[],
    #     render_mode=render_mode,
    #     novice=GLOB_NOVICE,
    # )
    gridworld = aec_to_parallel(gridworld)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(gridworld)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=num_vec_envs, num_cpus=2, base_class='stable_baselines3')

    return gridworld.aec_env, vec_env


def train(config):

    num_policies = 2
    num_agents = 4
    gridworld, vec_env = make_new_vec_gridworld(render_mode='human', num_vec_envs=num_policies, env_type=GLOB_ENV)
    trial_name = session.get_trial_name()

    model = IndependentRecurrentPPO(
        "MultiInputLstmPolicy",
        num_policies=num_policies,
        num_agents=num_agents,
        env=vec_env,
        verbose=1,
        tensorboard_log=f"/mnt/e/BrainHack-TIL25/ppo_logs/{trial_name}",
        **config
    )

    # TODO: make it multi agent Set where to save the model
    checkpoint_callbacks = [[CheckpointCallback(
        save_freq=10000,                    # Save every n steps
        save_path=f"/mnt/e/BrainHack-TIL25/checkpoints/dqn_agent_{i}/{trial_name}",         # Target directory
        name_prefix=f"{EXPERIMENT_NAME}_novice_{GLOB_NOVICE}_run_{trial_name}"
    )] for i in range(num_policies)]

    model.learn(
        total_timesteps=2000, 
        n_rollout_steps=1000,
        callbacks=checkpoint_callbacks)
    print('rewards:', np.unique(model.agents[0].replay_buffer.rewards, return_counts=True))
    print('replay buff pos', model.agents[0].replay_buffer.pos)

    model.save(f"/mnt/e/BrainHack-TIL25/checkpoints/ppo/{trial_name}/final_ppo_model_for_run_{trial_name}")

    # .load instantiates a new instance of the model. This is to simulate how you would run inference for this model.
    # instantiate the model, and do rollouts without action noise or randomness or sampling from replay buffer.
    gridworld, vec_env = make_new_vec_gridworld(render_mode='rgb_array', num_vec_envs=1, env_type=GLOB_ENV)
    eval_model = IndependentRecurrentPPO.load(
        policy="MultiInputLstmPolicy",
        path=f"/mnt/e/BrainHack-TIL25/checkpoints/ppo/{trial_name}/final_ppo_model_for_run_{trial_name}",
        num_policies=num_policies,
        train_freq=(100, 'episode'),  # we arent training, we're just running for 100 rounds before stopping rollout.
        learning_starts=0,  # no random sampling of action space
        env=vec_env,
    )

    reset_obs = vec_env.reset()
    # hijack collect_rollouts function to evaluate for us.
    eval_model.collect_rollouts(
        last_obs=reset_obs,
        train_freq=eval_model.train_freq,
        learning_starts=eval_model.learning_starts,
        run_on_step=False,
    )
    
    print('rewards:', np.unique(eval_model.agents[0].replay_buffer.rewards))
    
    # results_dict = evaluate(model, num_rounds=100, render_mode=None)
    mean_all_score = sum(results_dict.values()) / 4
    mean_scout_score = results_dict.pop('player_0')
    mean_guard_score = sum(results_dict.values()) / 3

    tune.report(
        dict(
            mean_scout_score=mean_scout_score,
            mean_guard_score=mean_guard_score,
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
        perturbation_interval=5,  # every n trials
        hyperparam_mutations={
                "learning_rate": tune.loguniform(1e-5, 1e-2),
                # "gamma": tune.uniform(0.80, 0.999),
                # "n_steps": tune.choice([256, 512, 1024]),
                # "batch_size": tune.choice([64, 128]),
                # "n_epochs": tune.choice([5, 7, 10]),
                # "vf_coef": tune.uniform(0.10, 0.80),
                # "ent_coef": tune.loguniform(1e-6, 1e-4),
                # "gae_lambda": tune.uniform(0.70, 0.99),
            })
        tuner = Tuner(
            tune.with_resources(train, resources={"cpu": 2.0, "memory": 4 * 1024 ** 3}),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=1,
                reuse_actors=True,
            ),
        )

        results = tuner.fit()
        print('best results', results)

        # analysis = tune.run(
        #     train,
        #     config={
        #         "learning_rate": tune.loguniform(5e-5, 5e-3),
        #         "gamma": tune.uniform(0.8, 0.999),
        #         "n_steps": tune.choice([256, 512, 1024]),
        #         "batch_size": tune.choice([64, 128]),
        #         "n_epochs": tune.choice([5, 7, 10]),
        #         "vf_coef": tune.uniform(0.2, 0.8),
        #         "ent_coef": tune.loguniform(1e-5, 1e-3),
        #         "gae_lambda": tune.uniform(0.8, 0.99),
        #     },
        #     num_samples=200,
        #     metric="mean_scout_score",
        #     mode="max",
        #     max_concurrent_trials=2,
        #     resources_per_trial={"cpu": 5, "gpu": 0.5},
        # )

    else:
        model = PPO.load(args.ckpt)
        # gridworld = build_env(
        #     env_type=GLOB_ENV,
        #     env_wrappers=[],
        #     render_mode='human',
        #     novice=False,
        # )
        gridworld = env(
            env_wrappers=[],
            render_mode=render_mode,
            novice=GLOB_NOVICE,
        )

        evaluate(model, gridworld.possible_agents, num_rounds=10, render_mode='human')
        

