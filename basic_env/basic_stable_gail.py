from gym_unity.envs import UnityEnv
import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import GAIL, SAC
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.bench import Monitor
import os
import time
import pickle

if __name__ == "__main__":
    worker_id = 32
    env_name = "/home/jim/projects/unity_ray/basic_env_linux/basic_env_linux"
    env = UnityEnv(env_name, worker_id=worker_id, use_visual=False) #

    # Create log dir
    time_int = int(time.time())
    log_dir = "stable_results/basic_env_{}/".format(time_int)
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir, allow_early_resets=True)

    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    # Generate expert trajectories (train expert)
    model = PPO2(MlpPolicy, env, verbose=1)
    generate_expert_traj(model, 'expert_basic_env', n_timesteps=10000, n_episodes=1000)
    env.close()

    print("Ending expert training, training with GAIL")
    # Load the expert dataset
    worker_id += 1
    env = UnityEnv(env_name, worker_id=worker_id, use_visual=False)  # , no_graphics=True
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    dataset = ExpertDataset(expert_path='expert_basic_env.npz', traj_limitation=10, verbose=1)

    model = GAIL("MlpPolicy", env, dataset, verbose=1)
    model.learn(total_timesteps=30000)
    model.save(log_dir + "model")
    print("evaluating agent")
    #evaluate agent
    episodes = 100
    ep_r = []
    ep_l = []
    for e in range(episodes):
        obs = env.reset()
        total_r = 0.
        total_l = 0.
        while total_l < 200:
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

