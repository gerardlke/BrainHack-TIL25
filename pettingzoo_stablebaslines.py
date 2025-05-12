import supersuit as ss
import ray

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from til_environment.stablebaselines_gridworld import new_raw_env, env
from pettingzoo.utils.conversions import aec_to_parallel

from ray import tune
from ray.air import session

ray.init(include_dashboard=True)

def make_new_vec_gridworld(render_mode=None):
    gridworld = env(
        env_wrappers=[],
        render_mode=render_mode,
        novice=False,
    )
    gridworld = aec_to_parallel(gridworld)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(gridworld)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=4, num_cpus=2, base_class='stable_baselines3')

    return gridworld, vec_env

def evaluate(model, agents, num_rounds=10, render_mode='human'):
    rewards = {agent: 0 for agent in agents}

    for _ in range(num_rounds):
        # environment switches player when reset is called, but we want fixed evaluation
        # where the first player is always the scout and the others are guards
        # so just build world from scratch again
        gridworld = env(
            env_wrappers=[],
            render_mode=render_mode,
            novice=False,
        )
        gridworld.reset()

        for agent in gridworld.agent_iter():
            observation, reward, termination, truncation, info = gridworld.last()
            observation = {
                k: v if type(v) is int else v.tolist() for k, v in observation.items()
            }
            for a in gridworld.agents:
                rewards[a] += gridworld.rewards[a]
            if termination or truncation:
                action = None
            else:
                action, _states = model.predict(observation, deterministic=False)
            gridworld.step(action)

    gridworld.close()

    mean_rewards = {k: v / num_rounds for k, v in rewards.items()}

    return mean_rewards

def train(config):

    gridworld, vec_env = make_new_vec_gridworld()
    trial_name = session.get_trial_name()
    print('trial_name', trial_name)
    model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=f"./ppo_logs/{trial_name}", **config)

    # Set where to save the model
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,                    # Save every n steps
        save_path="./mnt/e/BrainHack-TIL25/checkpoints/ppo",         # Target directory
        name_prefix=f"advanced_run_{trial_name}"
    )

    model.learn(
        total_timesteps=1, 
        callback=checkpoint_callback)

    model.save(f"/mnt/e/BrainHack-TIL25/checkpoints/ppo/final_ppo_model_for_run_{trial_name}")

    results_dict = evaluate(model, gridworld.possible_agents, num_rounds=2, render_mode='human')

    mean_scout_score = results_dict.pop('player_1')
    mean_guard_score = sum(results_dict.values()) / 3
    print(mean_scout_score, mean_guard_score)

    tune.report(
        dict(
            mean_scout_score=mean_scout_score,
            mean_guard_score=mean_guard_score,
        )
    )

import argparse

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
        analysis = tune.run(
            train,
            config={
                "learning_rate": tune.loguniform(5e-5, 5e-3),
                "gamma": tune.uniform(0.8, 0.999),
                "n_steps": tune.choice([2048, 4098, 8192]),
                "batch_size": tune.choice([256, 512, 1024]),
                "n_epochs": tune.choice([5, 7, 10]),
            },
            num_samples=10,
            metric="mean_scout_score",
            mode="max",
            max_concurrent_trials=2,
            resources_per_trial={"cpu": 5, "gpu": 0.5},
        )

    # else:
        

