import gym
import numpy as np
import os
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.bench import Monitor
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN, A2C, ACER
from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from visualization_and_metrics import visualisation_value_RM, visualize_policy_RM, average_n_episodes
import matplotlib.pyplot as plt


def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, rewards, steps
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          rewards.append(mean_reward)
          steps.append(n_steps)
  n_steps += 1
  return True

if __name__ == '__main__':
    rewards, n_steps, steps = [], 0, []

    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    # env = gym.make('FrozenLake-v0')

    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 50))
    alpha = 0.8
    lamb = 0.7
    env = gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)
    env = Monitor(env, log_dir, allow_early_resets=True)

    # V, P_ref = dynamic_programming_env_DCP(env)
    # V = V.reshape(env.T * env.C)
    # visualisation_value_RM(V, env.T, env.C)
    # visualize_policy_RM(P_ref, env.T, env.C)
    # P_DP = P_ref.reshape(env.T * env.C)
    # print("Average reward over 10000 episodes : " + str(average_n_episodes(env, P_DP, 10000, 0.01)))

    env_vec = DummyVecEnv([lambda: env])

    total_timesteps=40_000
    states = [k for k in range(env.T*env.C)]

    # DQN
    model = DQN(MlpPolicy, env_vec, gamma=0.99, learning_rate=0.0005, buffer_size=400000, exploration_fraction=0.2,
                 exploration_final_eps=0.02, train_freq=1, batch_size=1000, checkpoint_freq=10000, checkpoint_path=None,
                 learning_starts=1000, target_network_update_freq=50, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False, verbose=1, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    policy, q_values, _ = model.step_model.step(states, deterministic=True)

    # Policy optimization
    # n_cpu = 2
    # env_subproc = SubprocVecEnv([lambda: env for i in range(n_cpu)])
    # model = A2C(MlpPolicy, env_subproc, verbose=1)
    # model.learn(total_timesteps=total_timesteps)
    # policy = model.step_model.step(states, deterministic=True)[0]

    # policy = np.array([env.A[k] for k in policy])
    # visualize_policy_RM(policy, env.T, env.C)
    # print("Average reward over 10000 episodes : " + str(average_n_episodes(env, policy, 10000)))
    # # print(q_values.reshape(env.T, env.C, env.action_space.n))

    plt.plot(steps, rewards)
    plt.show()
