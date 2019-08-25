from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym


import ray
from ray import tune
from ray.tune import grid_search
from ray.tune.registry import register_env

import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

import matplotlib.pyplot as plt
import numpy as np
from gym_unity.envs import UnityEnv



if __name__ == "__main__":

    class UnityEnvWrapper(gym.Env):
        def __init__(self, env_config):
            self.vector_index = env_config.vector_index
            self.worker_index = env_config.worker_index
            self.worker_id = env_config["unity_worker_id"] + env_config.worker_index
            # Name of the Unity environment binary to launch
            env_name = '/home/jim/projects/unity_ray/basic_env_linux/basic_env_linux'
            self.env = UnityEnv(env_name, worker_id=self.worker_id, use_visual=False, multiagent=False, no_graphics=True) #
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

        def reset(self):
            return self.env.reset()

        def step(self, action):
            return self.env.step(action)

    register_env("unity_env", lambda config: UnityEnvWrapper(config))



    ray.init()

    tune.run(
        "PPO",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": "unity_env",
            "num_workers": 0, #
            "env_config":{
                "unity_worker_id": 52
            },
            "train_batch_size": 500,
        },
        checkpoint_at_end=True,
    )

