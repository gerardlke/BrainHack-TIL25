import supersuit as ss
import ray
import argparse
import numpy as np
from independent_recurrent_ppo import IndependentRecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# from til_environment.stablebaselines_gridworld import build_env
from til_environment.training_gridworld import env
from pettingzoo.utils.conversions import aec_to_parallel

from ray import tune
from ray.tune import Tuner
from ray.air import session
from ray.tune.schedulers import PopulationBasedTraining

GLOB_NOVICE = False
GLOB_ENV = 'normal'
EXPERIMENT_NAME = 'adv_rppo_test_normalenv'

def make_new_vec_gridworld(render_mode=None, env_type='normal', num_vec_envs=1):
    """
    Helper func to build gridworld into a vectorized form. Will also return original AEC env for evaluation too, so dont worry
    """
    # gridworld = build_env(
    #     env_type=env_type,
    #     env_wrappers=[],
    #     render_mode=render_mode,
    #     novice=GLOB_NOVICE,
    # )
    gridworld = env(
        env_wrappers=[],
        render_mode=render_mode,
        novice=GLOB_NOVICE,
    )
    gridworld = aec_to_parallel(gridworld)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(gridworld)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=num_vec_envs, num_cpus=6, base_class='stable_baselines3')

    return gridworld.aec_env, vec_env


def train(config):
    n_steps = config.pop('n_steps')
    num_policies = 2
    num_agents = 4
    num_vec_envs = 4
    gridworld, vec_env = make_new_vec_gridworld(render_mode='human', num_vec_envs=num_vec_envs, env_type=GLOB_ENV)
    trial_name = session.get_trial_name()

    model = IndependentRecurrentPPO(
        "MultiInputLstmPolicy",
        n_steps=n_steps,  # fixed at 100 cuz the max session length is 100
        num_policies=num_policies,
        num_agents=num_agents,
        env=vec_env,
        verbose=1,
        tensorboard_log=f"/mnt/e/BrainHack-TIL25/rppo_logs/{trial_name}",
        **config
    )

    # TODO: make it multi agent Set where to save the model
    checkpoint_callbacks = [[CheckpointCallback(
        save_freq=10000,                    # Save every n steps
        save_path=f"/mnt/e/BrainHack-TIL25/checkpoints/rppo_agent_{i}/{trial_name}",         # Target directory
        name_prefix=f"{EXPERIMENT_NAME}_novice_{GLOB_NOVICE}_run_{trial_name}"
    )] for i in range(num_policies)]

    model.learn(
        total_timesteps=200000, 
        callbacks=checkpoint_callbacks)
    
    save_path = f"/mnt/e/BrainHack-TIL25/checkpoints/rppo/{trial_name}/final_rppo_model_for_run_{trial_name}"
    model.save(save_path)

    # .load instantiates a new instance of the model. This is to simulate how you would run inference for this model.
    # instantiate the model, and do rollouts without action noise or randomness or sampling from replay buffer.
    gridworld, vec_env = make_new_vec_gridworld(render_mode=None, num_vec_envs=1, env_type=GLOB_ENV)
    eval_model = IndependentRecurrentPPO.load(
        policy="MultiInputLstmPolicy",
        path=save_path,
        num_policies=num_policies,
        num_agents=num_agents,
        env=vec_env,
        n_steps=n_steps,
        verbose=1,
        tensorboard_log=f"/mnt/e/BrainHack-TIL25/rppo_logs/{trial_name}",
        **config
    )

    reset_obs = vec_env.reset()
    # hijack collect_rollouts function to evaluate for us.
    all_scout_score, all_guard_score = [], []
    num_eval_rounds = 10
    for _ in range(num_eval_rounds):
        eval_model.collect_rollouts(
            last_obs=reset_obs,
            n_rollout_steps=eval_model.n_steps * eval_model.num_envs,  # rollout increments timesteps by number of envs
        )
        
        scout_pol_rewards = np.sum(model.policies[1].rollout_buffer.rewards) / eval_model.num_scout_envs
        guard_pol_rewards = np.sum(model.policies[0].rollout_buffer.rewards) / eval_model.num_guard_envs
        all_scout_score.append(scout_pol_rewards)
        all_guard_score.append(guard_pol_rewards)
        print('scout_pol_rewards', scout_pol_rewards)
        print('guard_pol_rewards', guard_pol_rewards)

    mean_scout_score = sum(all_scout_score) / len(all_scout_score)
    mean_guard_score = sum(all_guard_score) / len(all_guard_score)

    # results_dict = evaluate(model, num_rounds=100, render_mode=None)
    mean_all_score = (mean_scout_score +  mean_guard_score) / 2

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
                "learning_rate": tune.loguniform(1e-5, 1e-3),
                "gamma": tune.uniform(0.80, 0.999),
                "n_steps": tune.choice([256, 512, 1024]),
                "batch_size": tune.choice([64, 128]),
                "n_epochs": tune.choice([5, 7, 10]),
                "vf_coef": tune.uniform(0.10, 0.80),
                "ent_coef": tune.loguniform(1e-6, 1e-4),
                "gae_lambda": tune.uniform(0.70, 0.99),
            })
        tuner = Tuner(
            tune.with_resources(train, resources={"cpu": 6.0, "memory": 8 * 1024 ** 3}),
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=10,
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
        gridworld, vec_env = make_new_vec_gridworld(render_mode=None, num_vec_envs=1, env_type=GLOB_ENV)
        save_path = '/mnt/e/BrainHack-TIL25/checkpoints/dqn_agent_0/train_8b2ec_00005/adv_rppo_test_normalenv_novice_False_run_train_8b2ec_00005_80000_steps.zip'
        num_policies = 2
        num_agents = 4
        n_steps = 100
        eval_model = IndependentRecurrentPPO(
            policy="MultiInputLstmPolicy",
            num_policies=num_policies,
            num_agents=num_agents,
            env=vec_env,
            n_steps=n_steps,
            verbose=1,
        )
        # load guard
        eval_model.load_policy_id(
            path='/mnt/e/BrainHack-TIL25/checkpoints/dqn_agent_0/train_8b2ec_00005/adv_rppo_test_normalenv_novice_False_run_train_8b2ec_00005_80000_steps.zip',
            policy_id=0
        )
        # load scout
        eval_model.load_policy_id(
            path='/mnt/e/BrainHack-TIL25/checkpoints/dqn_agent_1/train_8b2ec_00005/adv_rppo_test_normalenv_novice_False_run_train_8b2ec_00005_80000_steps.zip',
            policy_id=0
        )

        reset_obs = vec_env.reset()
        # hijack collect_rollouts function to evaluate for us.
        all_scout_score, all_guard_score = [], []
        num_eval_rounds = 10
        for _ in range(num_eval_rounds):
            eval_model.collect_rollouts(
                last_obs=reset_obs,
                n_rollout_steps=eval_model.n_steps * eval_model.num_envs,  # rollout increments timesteps by number of envs
            )
            print('n_rollout_steps=eval_model.n_steps * eval_model.num_envs', eval_model.n_steps * eval_model.num_envs)
            scout_pol_rewards = np.sum(eval_model.policies[1].rollout_buffer.rewards) / eval_model.num_scout_envs
            guard_pol_rewards = np.sum(eval_model.policies[0].rollout_buffer.rewards) / eval_model.num_guard_envs
            all_scout_score.append(scout_pol_rewards)
            all_guard_score.append(guard_pol_rewards)
            print('scout_pol_rewards', scout_pol_rewards)
            print('guard_pol_rewards', guard_pol_rewards)

        mean_scout_score = sum(all_scout_score) / len(all_scout_score)
        mean_guard_score = sum(all_guard_score) / len(all_guard_score)

        # results_dict = evaluate(model, num_rounds=100, render_mode=None)
        mean_all_score = (mean_scout_score +  mean_guard_score) / 2
        

