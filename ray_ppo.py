import os
import random
import pprint

import ray
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.schedulers import PopulationBasedTraining

ray.init(runtime_env={"env_vars": {"TF_DISABLE_MKL": "1", "CUDA_VISIBLE_DEIVCES": "-1"}})

from til_environment import gridworld
from til_environment.training_gridworld import TrainRawEnv, train_env


print('train_env', train_env)

env = train_env(
        env_wrappers=[],
        render_mode=None,
        novice=True,
        num_iters=1000,
    )

def env_creator(env_config):
    """
    This function runs when registering til environment into gymnasium
    """
    # return gridworld.env(env_wrappers=[], render_mode=None, novice=True)
    return env

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args, _ = parser.parse_known_args()

    abs_path = os.path.abspath("./my_training_runs")

    register_env("custom_env", env_creator)

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    hyperparam_mutations = {
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_epochs": lambda: random.randint(1, 30),
        "minibatch_size": lambda: random.randint(128, 16384),
        "train_batch_size_per_learner": lambda: random.randint(2000, 160000),
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    # Stop when we've either reached 100 training iterations or reward=300
    stopping_criteria = {"training_iteration": 100, "episode_reward_mean": 300}

    config = (
        PPOConfig()
        .environment("Humanoid-v2")
        .env_runners(num_env_runners=4)
        .training(
            # These params are tuned from a fixed starting value.
            kl_coeff=1.0,
            lambda_=0.95,
            clip_param=0.2,
            lr=1e-4,
            # These params start off randomly drawn from a set.
            num_epochs=tune.choice([10, 20, 30]),
            minibatch_size=tune.choice([128, 512, 2048]),
            train_batch_size_per_learner=tune.choice([10000, 20000, 40000]),
        )
        .rl_module(
            model_config=DefaultModelConfig(free_log_std=True),
        )
    )

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            scheduler=pbt,
            num_samples=1 if args.smoke_test else 2,
        ),
        param_space=config,
        run_config=tune.RunConfig(stop=stopping_criteria),
    )
    results = tuner.fit()