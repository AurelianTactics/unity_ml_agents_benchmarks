from gym_unity.envs import UnityEnv
import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
import os
import time
import pickle

if __name__ == "__main__":
    env_name = "/home/jim/projects/unity_ray/basic_env_linux/basic_env_linux"
    env = UnityEnv(env_name, worker_id=2, use_visual=False)

    # Create log dir
    time_int = int(time.time())
    log_dir = "stable_results/basic_env_{}/".format(time_int)
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir, allow_early_resets=True)

    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)

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
            obs, reward, done, info = env.step(action)
            total_l += 1.
            total_r += reward
            if done:
                break
        ep_r.append(total_r)
        ep_l.append(total_l)
    print("episode mean reward: {:0.3f} mean length: {:0.3f}".format(np.mean(ep_r), np.mean(ep_l)))
    with open('{}_eval.pkl'.format(log_dir), 'wb') as f:
        pickle.dump(ep_r, f)
        pickle.dump(ep_l, f)

    env.close()
    model.save(log_dir+"model")
