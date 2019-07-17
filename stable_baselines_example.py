import gym
import numpy as np
import os
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.bench import Monitor
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN, A2C, ACER
from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from visualization_and_metrics import visualisation_value_RM, visualize_policy_RM, average_n_episodes
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from pathlib import Path
import sys
import ast
from functools import partial
from multiprocessing import Pool
import glob
import csv


def agent_builder(env_vec, parameters_dict):
    return DQN(MlpPolicy, env_vec, weights=parameters_dict["weights"],
               original_weights=parameters_dict["original_weights"],
               gamma=parameters_dict["gamma"], learning_rate=parameters_dict["learning_rate"],
               buffer_size=parameters_dict["buffer_size"],
               exploration_fraction=parameters_dict["exploration_fraction"],
               exploration_final_eps=parameters_dict["exploration_final_eps"],
               train_freq=parameters_dict["train_freq"], batch_size=parameters_dict["batch_size"],
               checkpoint_freq=parameters_dict["checkpoint_freq"],
               checkpoint_path=parameters_dict["checkpoint_path"], learning_starts=parameters_dict["learning_starts"],
               target_network_update_freq=parameters_dict["target_network_update_freq"],
               prioritized_replay=parameters_dict["prioritized_replay"],
               prioritized_replay_alpha=parameters_dict["prioritized_replay_alpha"],
               prioritized_replay_beta0=parameters_dict["prioritized_replay_beta0"],
               prioritized_replay_beta_iters=parameters_dict["prioritized_replay_beta_iters"],
               prioritized_replay_eps=parameters_dict["prioritized_replay_eps"],
               param_noise=parameters_dict["param_noise"], verbose=parameters_dict["verbose"],
               policy_kwargs=parameters_dict["policy_kwargs"],
               tensorboard_log=parameters_dict["tensorboard_log"])


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, rewards, env, states, model
    if n_steps == 0:
        policy, q_values, _ = model.step_model.step(states, deterministic=True)
        policy = np.array([env.A[k] for k in policy])
        rewards.append(average_n_episodes(env, policy, 10000))
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        policy, q_values, _ = model.step_model.step(states, deterministic=True)
        policy = np.array([env.A[k] for k in policy])
        rewards.append(average_n_episodes(env, policy, 10000))
    n_steps += 1
    return True


def run_once(parameters_dict, nb_timesteps, general_dir_name, parameter, value, k):
    global env, rewards, n_steps, steps, model, states

    env = parameters_dict["env_builder"]()
    env_vec = DummyVecEnv([lambda: env])

    rewards, n_steps = [], 0

    states = [k for k in range(env.T * env.C)]

    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    model = agent_builder(env_vec, parameters_dict)
    model.learn(total_timesteps=nb_timesteps, callback=callback)

    np.save(general_dir_name / parameter / str(value) / ("Run" + str(k) + ".npy"), rewards)


def run_n_times(parameters_dict, nb_timesteps, general_dir_name, parameter, number_of_runs, value):
    (general_dir_name / parameter / str(value)).mkdir(parents=True, exist_ok=True)

    f = partial(run_once, parameters_dict, nb_timesteps, general_dir_name, parameter, value)

    with Pool(number_of_runs) as pool:
        pool.map(f, range(number_of_runs))


def collect_list_of_mean_revenues(general_dir_name, parameter, value):
    list_of_rewards = []
    for np_name in glob.glob('../' + general_dir_name.name + '/' + parameter + '/' + str(value) + '/*.np[yz]'):
        list_of_rewards.append(list(np.load(np_name)))

    nb_collection_points = len(list_of_rewards[0])

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

    return mean_revenues, min_revenues, max_revenues


def plot_revenues(parameters_dict, nb_timesteps, mean_revenues, min_revenues, max_revenues):
    env = parameters_dict["env_builder"]()
    V, P_ref = dynamic_programming_env_DCP(env)
    P_DP = P_ref.reshape(env.T * env.C)

    steps = [0]
    for k in range(1000 - 1, nb_timesteps, 1000):
        steps.append(k)

    fig = plt.figure()
    plt.plot(steps, mean_revenues, color="gray", label='DQL mean revenue')
    plt.fill_between(steps, min_revenues, max_revenues, label='95% confidence interval', color="gray", alpha=0.2)
    plt.plot(steps, [average_n_episodes(env, P_DP, 10000)] * len(steps), label="DP Revenue")
    plt.legend()
    plt.ylabel("Revenue computed over 10000 episodes")
    plt.xlabel("Number of timesteps")
    return fig


def tune_parameter(general_dir_name, parameter, parameter_values, parameters_dict, nb_timesteps, number_of_runs):
    # results_dir_name = "../Daily meetings/Stabilization experiences/" + parameter
    # os.mkdir(general_dir_name + "/" + parameter)
    (general_dir_name / parameter).mkdir(parents=True, exist_ok=True)
    parameter_path = Path(parameter)

    for value in parameter_values:
        print(parameter + " = "+str(value))
        parameters_dict[parameter] = value

        if parameter == "policy_kwargs":
            value = value['dueling']

        experience_dir_name = parameter + " = " + str(value)

        run_n_times(parameters_dict, nb_timesteps, general_dir_name, parameter, number_of_runs, value)
        mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(general_dir_name, parameter,
                                                                                  value)
        fig = plot_revenues(parameters_dict, nb_timesteps, mean_revenues, min_revenues, max_revenues)
        plt.savefig('../' + general_dir_name.name + '/' + parameter + '/' + experience_dir_name + '.png')

        mean_revenue, speed = compute_metric(mean_revenues)
        metrics_file_name = '../' + general_dir_name.name + '/metrics_file.csv'
        save_metrics(metrics_file_name, parameter, value, mean_revenue, speed)


def compute_metric(list_of_revenues):
    starting_point_slope = 0
    ending_point_slope = 5

    starting_point_mean = 20

    mean_revenue = np.mean([list_of_revenues[-starting_point_mean:]])
    speed = (list_of_revenues[ending_point_slope] - list_of_revenues[starting_point_slope]) / (
                ending_point_slope - starting_point_slope)

    return mean_revenue, speed

def save_metrics(metrics_file_name, parameter, value, mean_revenues, speed):
    file_exists = os.path.isfile(metrics_file_name)
    fieldnames = ['parameter', 'value', 'average', 'slope']

    if file_exists:
        with open(metrics_file_name, 'a') as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            writer.writerow({'parameter': parameter, 'value': value, 'average': mean_revenues, 'slope': speed})
    else:
        with open(metrics_file_name, 'a') as metrics_file:
            writer = csv.DictWriter(metrics_file, delimiter=',', lineterminator='\n', fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'parameter': parameter, 'value': value, 'average': mean_revenues, 'slope': speed})

def compare_plots(general_dir_name, parameter, values, nb_timesteps):
    plt.figure()

    for value in values:
        if parameter == "policy_kwargs":
            value = value["dueling"]
        steps = [0]
        for k in range(1000 - 1, nb_timesteps, 1000):
            steps.append(k)
        mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(general_dir_name, parameter, value)
        plt.plot(steps, mean_revenues, label=str(value))
        plt.fill_between(steps, min_revenues, max_revenues, alpha=0.2)

    plt.legend()
    plt.ylabel("Revenue computed over 10000 episodes")
    plt.xlabel("Number of timesteps")
    plt.title(parameter)

    plt.savefig('../' + general_dir_name.name + '/' + parameter + '.png')

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


def compute_weights(env):
    compute_weight = lambda x: 1 + max(1. * x[0] / env.T, 1. * x[1] / env.C)
    weights = [(s, compute_weight((env.to_coordinate(s)[0], env.to_coordinate(s)[1]))) for s in range(env.T*env.C)]
    return weights


if __name__ == '__main__':

    parameters_dict = {}
    parameters_dict["env_builder"] = env_builder
    parameters_dict["gamma"] = 0.99
    parameters_dict["learning_rate"] = 0.0001
    parameters_dict["buffer_size"] = 10000
    parameters_dict["exploration_fraction"] = 0.4
    parameters_dict["exploration_final_eps"] = 0.01
    parameters_dict["train_freq"] = 1
    parameters_dict["batch_size"] = 100
    parameters_dict["checkpoint_freq"] = 10000
    parameters_dict["checkpoint_path"] = None
    parameters_dict["learning_starts"] = 100
    parameters_dict["target_network_update_freq"] = 50
    parameters_dict["prioritized_replay"] = False
    parameters_dict["prioritized_replay_alpha"] = 0.6
    parameters_dict["prioritized_replay_beta0"] = 0.4
    parameters_dict["prioritized_replay_beta_iters"] = None
    parameters_dict["prioritized_replay_eps"] = 1e-6
    parameters_dict["param_noise"] = False
    parameters_dict["verbose"] = 0
    # parameters_dict["tensorboard_log"] = "./../log_tensorboard/"
    parameters_dict["tensorboard_log"] = None
    parameters_dict["policy_kwargs"] = {"dueling" : False}
    parameters_dict["weights"] = False

    env = parameters_dict["env_builder"]()

    parameters_dict["original_weights"] = compute_weights(env)

    results_path = Path("../Results_big_env")
    results_path.mkdir(parents=True, exist_ok=True)

    # Tuning of the parameters
    # parameter = sys.argv[1]
    # parameter_values_string = sys.argv[2]
    # print(parameter_values_string)
    # parameter_values = ast.literal_eval(parameter_values_string)

    parameter = "learning_rate"
    parameter_values = [1e-5, 1e-4, 1e-3]
    total_timesteps = 30000

    compare_plots(results_path, parameter, parameter_values, total_timesteps)

    total_timesteps = 30000
    nb_runs = 30

    # parameter = "target_network_update_freq"
    # parameter_values = [10, 50, 100, 500]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, total_timesteps, nb_runs)
    # compare_plots(results_path, parameter, parameter_values, total_timesteps)
    #
    # parameter = "buffer_size"
    # parameter_values = [1000, 10000, 20000, 30000]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, total_timesteps, nb_runs)
    # compare_plots(results_path, parameter, parameter_values, total_timesteps)
    #
    # parameter = "batch_size"
    # parameter_values = [10, 100, 1000]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, total_timesteps, nb_runs)
    # compare_plots(results_path, parameter, parameter_values, total_timesteps)
    #
    # parameter = "exploration_final_eps"
    # parameter_values = [0.5, 0.2, 0.01, 0.001]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, total_timesteps, nb_runs)
    # compare_plots(results_path, parameter, parameter_values, total_timesteps)
    #
    # parameter = "gamma"
    # parameter_values = [0.8, 0.9, 0.99, 0.999]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, total_timesteps, nb_runs)
    # compare_plots(results_path, parameter, parameter_values, total_timesteps)


