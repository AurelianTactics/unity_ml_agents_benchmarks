from gym_unity.envs import UnityEnv
import gym
import numpy as np
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import ACER
from stable_baselines import DQN
from stable_baselines import TRPO
from stable_baselines import ACKTR
from stable_baselines import GAIL
from stable_baselines import HER
from stable_baselines.bench import Monitor
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
import os
import time
import pickle


def make_env(env_id, log_dir, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = gym.make(env_id)
        # env.seed(seed + rank)
        env = UnityEnv(env_id, worker_id=rank, use_visual=False, no_graphics=True)
        #env.seed(seed + rank)
        env = Monitor(env, log_dir, allow_early_resets=True)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == "__main__":
    env_id = "/home/jim/projects/unity_ray/basic_env_linux/basic_env_linux"
    worker_id = 19
    env = UnityEnv(env_id, worker_id=worker_id, use_visual=False, no_graphics=True)
    # Create log dir
    time_int = int(time.time())
    log_dir = "stable_results/basic_env_{}/".format(time_int)
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    num_env = 2

    #env = SubprocVecEnv([make_env(env_id, log_dir, i+worker_id) for i in range(num_env)])
    model_class = DQN
    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE
    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                                                verbose=1)
    model.learn(total_timesteps=1000)
    model.save(log_dir+"model")

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    model = HER.load(log_dir+"model", env=env)

    #evaluate agent
    episodes = 100
    ep_r = []
    ep_l = []
    for e in range(episodes):
        obs = env.reset()
        total_r = 0.
        total_l = 0.
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            total_l += 1.
            total_r += rewards
            if dones:
                break
        ep_r.append(total_r)
        ep_l.append(total_l)
    print("episode mean reward: {:0.3f} mean length: {:0.3f}".format(np.mean(ep_r), np.mean(ep_l)))
    with open('{}_eval.pkl'.format(log_dir), 'wb') as f:
        pickle.dump(ep_r, f)
        pickle.dump(ep_l, f)

    env.close()

