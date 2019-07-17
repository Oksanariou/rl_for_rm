import gym
import numpy as np
import os

from visualization_and_metrics import average_n_episodes

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, rewards, env, states, model

    if n_steps % 10 == 0:
        print(n_steps)
        policy = model.predict(states)[0]
        policy = np.array([env.A[k] for k in policy])
        rewards.append(average_n_episodes(env, policy, 10000))
    n_steps += 1
    return True


def env_builder():
    # Parameters of the environment
    data_collection_points = 100
    micro_times = 5
    capacity = 50
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    return gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points,
                    capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)


if __name__ == '__main__':
    global env, rewards, n_steps, model, states

    rewards, n_steps = [], 0

    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    env = env_builder()
    env_vec = DummyVecEnv([lambda: env])

    states = [k for k in range(env.T * env.C)]

    model = TRPO(MlpPolicy, env_vec, verbose=0)
    model.learn(total_timesteps=250000, callback=callback)

