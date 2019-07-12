import gym
import numpy as np
import os
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.bench import Monitor
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN, A2C, ACER
from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from visualization_and_metrics import visualisation_value_RM, visualize_policy_RM, average_n_episodes
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from experience import tune_parameter


def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, rewards, steps, env, nb_avg_revenue, states, model
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      policy, q_values, _ = model.step_model.step(states, deterministic=True)
      policy = np.array([env.A[k] for k in policy])
      rewards.append(average_n_episodes(env, policy, nb_avg_revenue))
      steps.append(n_steps)
  n_steps += 1
  return True

def run_n_times(env, parameters_dict, nb_runs, nb_timesteps):
    V, P_ref = dynamic_programming_env_DCP(env)
    P_DP = P_ref.reshape(env.T * env.C)

    env_vec = DummyVecEnv([lambda: env])

    global states
    states = [k for k in range(env.T * env.C)]

    list_of_rewards = []

    for k in range(nb_runs):
        log_dir = "/tmp/gym/"
        os.makedirs(log_dir, exist_ok=True)

        global rewards, n_steps, steps, model
        rewards, n_steps, steps = [], 0, []

        model = DQN(MlpPolicy, env_vec)
        for key in parameters_dict:
            model.__setattr__(key, parameters_dict[key])

        model.learn(total_timesteps=nb_timesteps, callback=callback)

        list_of_rewards.append(rewards)


    nb_collection_points = len(rewards)

    all_rewards_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]

    for k in range(len(list_of_rewards)):
        rewards = list_of_rewards[k]
        for i in range(nb_collection_points):
            all_rewards_combined_at_each_collection_point[i].append(rewards[i])

    mean_revenues = [np.mean(list) for list in all_rewards_combined_at_each_collection_point]
    std_revenues = [sem(list) for list in all_rewards_combined_at_each_collection_point]
    confidence_revenues = [std_revenues[k] * t.ppf((1 + 0.95) / 2, nb_collection_points - 1) for k in
                           range(nb_collection_points)]
    min_revenues = [mean_revenues[k] - confidence_revenues[k] for k in range(nb_collection_points)]
    max_revenues = [mean_revenues[k] + confidence_revenues[k] for k in range(nb_collection_points)]

    fig = plt.figure()
    plt.plot(steps, mean_revenues, color="gray", label='DQL mean revenue')
    plt.fill_between(steps, min_revenues, max_revenues, label='95% confidence interval', color="gray", alpha=0.2)
    plt.plot(steps, [average_n_episodes(env, P_DP, nb_avg_revenue)] * len(steps), label="DP Revenue")
    plt.legend()
    plt.ylabel("Revenue computed over "+str(nb_avg_revenue)+" episodes")
    plt.xlabel("Number of timesteps")
    return fig

def tune_parameter(env, general_dir_name, parameter, parameter_values, parameters_dict, nb_timesteps, nb_runs, model=None, init_with_true_Q_table=False):

    # results_dir_name = "../Daily meetings/Stabilization experiences/" + parameter
    # os.mkdir(general_dir_name + "/" + parameter)

    for k in parameter_values:
        parameters_dict[parameter] = k
        experience_dir_name = parameter + " = " + str(parameters_dict[parameter])
        fig = run_n_times(env, parameters_dict, nb_runs, nb_timesteps)
        plt.savefig(general_dir_name + "/" + parameter + "/" + experience_dir_name + ".png")

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':

    nb_avg_revenue = 10000

    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 50))
    alpha = 0.8
    lamb = 0.7

    env = gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)
    # env = Monitor(env, log_dir, allow_early_resets=True)

    env_id = "gym_RMDCPDiscrete:RMDCPDiscrete-v0"
    num_cpu = 2
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # V, P_ref = dynamic_programming_env_DCP(env)
    # V = V.reshape(env.T * env.C)
    # visualisation_value_RM(V, env.T, env.C)
    # visualize_policy_RM(P_ref, env.T, env.C)
    # P_DP = P_ref.reshape(env.T * env.C)

    # DQN
    parameters_dict = {}
    parameters_dict["gamma"] = 0.99
    parameters_dict["learning_rate"] = 0.0005
    parameters_dict["buffer_size"] = 400000
    parameters_dict["exploration_fraction"] = 0.2
    parameters_dict["exploration_final_eps"] = 0.02
    parameters_dict["train_freq"] = 1
    parameters_dict["batch_size"] = 100
    parameters_dict["checkpoint_freq"] = 10000
    parameters_dict["checkpoint_path"] = None
    parameters_dict["learning_starts"] = 1000
    parameters_dict["target_network_update_freq"] = 50
    parameters_dict["prioritized_replay"] = False
    parameters_dict["prioritized_replay_alpha"] = 0.6
    parameters_dict["prioritized_replay_beta0"] = 0.4
    parameters_dict["prioritized_replay_beta_iters"] = None
    parameters_dict["prioritized_replay_eps"] = 1e-6
    parameters_dict["param_noise"] = False
    parameters_dict["verbose"] = 1
    parameters_dict["tensorboard_log"] = None


    # model = DQN(MlpPolicy, env_vec, gamma=0.99, learning_rate=0.0005, buffer_size=400000, exploration_fraction=0.2,
    #              exploration_final_eps=0.02, train_freq=1, batch_size=100, checkpoint_freq=10000, checkpoint_path=None,
    #              learning_starts=1000, target_network_update_freq=50, prioritized_replay=False,
    #              prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
    #              prioritized_replay_eps=1e-6, param_noise=False, verbose=1, tensorboard_log=None,
    #              _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)

    # model.learn(total_timesteps=total_timesteps, callback=callback)
    # policy, q_values, _ = model.step_model.step(states, deterministic=True)

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

    total_timesteps=10_000
    nb_runs = 5

    general_dir_name = "../Results"
    # os.mkdir(general_dir_name) #Creation of the folder where the results of the experience will be stocked

    parameter = "learning_rate"
    parameter_values = [1e-3, 1e-2]
    tune_parameter(env, general_dir_name, parameter, parameter_values, parameters_dict, total_timesteps, nb_runs)

    parameter = "target_network_update_freq"
    parameter_values = [10, 50, 100, 500]
    tune_parameter(env, general_dir_name, parameter, parameter_values, parameters_dict, total_timesteps, nb_runs)