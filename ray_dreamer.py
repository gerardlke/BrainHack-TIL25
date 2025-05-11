import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  
os.environ["TF_DISABLE_MKL"] = "1"

# a fairly annoying thing happened when running on my local machine, where the default LayerNorm value for tensorflow led to a float value
# that was not supported by GPU optimized MKL ops which work in float16. A work around is to turn it off for now, just to
# get the end to end training code working.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import ray
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config

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
    print('env config??', env_config)
    # return gridworld.env(env_wrappers=[], render_mode=None, novice=True)
    return env

register_env("custom_env", env_creator)

config = DreamerV3Config().environment(
    env="custom_env",
    ).training(
        model_size="S",  # Choose model size: XS, S, M, L, XL
        training_ratio=1024,
        batch_size_B=16,)

print('config', config)

trainer = config.build()
for i in range(1000):
    result = trainer.train()
    print(f"Iteration {i}: {result}")


