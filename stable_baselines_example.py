import gym
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

from visualization_and_metrics import visualize_policy_FL

if __name__ == '__main__':
    # env = gym.make('FrozenLake-v0')

    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 50))
    alpha = 0.8
    lamb = 0.7
    env = gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)
    T, C, A = env.T, env.C, env.action_space.n

    env = DummyVecEnv([lambda: env])

    model = DQN(MlpPolicy, env, gamma=0.99, learning_rate=0.0005, buffer_size=400000, exploration_fraction=0.2,
                 exploration_final_eps=0.02, train_freq=1, batch_size=100, checkpoint_freq=10000, checkpoint_path=None,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False, verbose=1, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
    model.learn(total_timesteps=400000)
    print("Model learnt")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    states = [k for k in range(T*C)]
    policy, q_values, _ = model.step_model.step(states, deterministic = True)
    policy = policy.reshape(T, C)
    print(q_values.reshape(T, C, A))