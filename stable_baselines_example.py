import gym
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

from visualization_and_metrics import visualize_policy_FL

if __name__ == '__main__':
    # env = gym.make('FrozenLake-v0')

    data_collection_points = 4
    micro_times = 3
    capacity = 4
    actions = tuple(k for k in range(50, 231, 50))
    alpha = 0.8
    lamb = 0.7
    env = gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    env = DummyVecEnv([lambda: env])

    model = DQN(MlpPolicy, env, verbose=1, prioritized_replay=False)
    model.learn(total_timesteps=50000)
    print("Model learnt")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    policy = []
    for k in range(16):
        action, _ = model.predict(k)
        policy.append(action)
    print(np.array(policy).reshape(4,4))