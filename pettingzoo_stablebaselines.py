import supersuit as ss
import ray
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from til_environment.stablebaselines_gridworld import build_env
from pettingzoo.utils.conversions import aec_to_parallel

from ray import tune
from ray.tune import Tuner
from ray.air import session
from ray.tune.schedulers import PopulationBasedTraining

GLOB_NOVICE = False
GLOB_ENV = 'binary_viewcone'
EXPERIMENT_NAME = 'adv_binary_long_test'

def make_new_vec_gridworld(render_mode=None, env_type='normal', num_vec_envs=1):
    gridworld = build_env(
        env_type=env_type,
        env_wrappers=[],
        render_mode=render_mode,
        novice=GLOB_NOVICE,
    )
    gridworld = aec_to_parallel(gridworld)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(gridworld)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=num_vec_envs, num_cpus=2, base_class='stable_baselines3')

    return gridworld, vec_env

def evaluate(model, agents, num_rounds=10, render_mode='human'):
    rewards = {agent: 0 for agent in agents}

    for _ in range(num_rounds):
        # environment switches player when reset is called, but we want fixed evaluation
        # where the first player is always the scout and the others are guards
        # so just build world from scratch again
        gridworld = build_env(
            env_type=GLOB_ENV,
            env_wrappers=[],
            render_mode=render_mode,
            novice=GLOB_NOVICE,
        )
        gridworld.reset()
        interm_rewards = {agent: 0 for agent in agents}
        for agent in gridworld.agent_iter():
            observation, reward, termination, truncation, info = gridworld.last()
            observation = {
                k: v if type(v) is int else v.tolist() for k, v in observation.items()
            }
            for a in gridworld.agents:
                rewards[a] += gridworld.rewards[a]
                interm_rewards[a] += gridworld.rewards[a]
            if termination or truncation:
                action = None
            else:
                action, _states = model.predict('what', deterministic=False)
                print('actions???', action)
            gridworld.step(action)
        
        print('intermediate rewards', interm_rewards)

    gridworld.close()

    mean_rewards = {k: v / num_rounds for k, v in rewards.items()}

    return mean_rewards

def train(config):

    gridworld, vec_env = make_new_vec_gridworld(num_vec_envs=4, env_type=GLOB_ENV)
    trial_name = session.get_trial_name()

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f"/mnt/e/BrainHack-TIL25/ppo_logs/{trial_name}",
        **config
    )

    # Set where to save the model
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,                    # Save every n steps
        save_path=f"/mnt/e/BrainHack-TIL25/checkpoints/ppo/{trial_name}",         # Target directory
        name_prefix=f"{EXPERIMENT_NAME}_novice_{GLOB_NOVICE}_run_{trial_name}"
    )

    model.learn(
        total_timesteps=10, 
        callback=checkpoint_callback)

    model.save(f"/mnt/e/BrainHack-TIL25/checkpoints/ppo/{trial_name}/final_ppo_model_for_run_{trial_name}")

    results_dict = evaluate(model, gridworld.possible_agents, num_rounds=100, render_mode=None)

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
                "gamma": tune.uniform(0.80, 0.999),
                "n_steps": tune.choice([256, 512, 1024]),
                "batch_size": tune.choice([64, 128]),
                "n_epochs": tune.choice([5, 7, 10]),
                "vf_coef": tune.uniform(0.10, 0.80),
                "ent_coef": tune.loguniform(1e-6, 1e-4),
                "gae_lambda": tune.uniform(0.70, 0.99),
            })
        tuner = Tuner(
            tune.with_resources(train, resources={"cpu": 2.0, "gpu":0.5, "memory": 4 * 1024 ** 3}),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=200,
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
        gridworld = build_env(
            env_type=GLOB_ENV,
            env_wrappers=[],
            render_mode='human',
            novice=False,
        )

        evaluate(model, gridworld.possible_agents, num_rounds=10, render_mode='human')
        

