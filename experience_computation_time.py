import gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

import sys
import ast

from keras.losses import mean_squared_error, logcosh

from DQL.agent_time import DQNAgent_time_builder


def env_builder():
    # Parameters of the environment
    data_collection_points = 100
    micro_times = 5
    capacity = 50
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    return gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)


def parameter_dict_builder():
    parameters_dict = {}
    parameters_dict["env_builder"] = env_builder
    parameters_dict["gamma"] = 0.99
    parameters_dict["replay_method"] = "DDQL"
    parameters_dict["batch_size"] = 100
    parameters_dict["memory_size"] = 30000
    parameters_dict["mini_batch_size"] = 100
    parameters_dict["prioritized_experience_replay"] = False
    parameters_dict["target_model_update"] = 50
    parameters_dict["hidden_layer_size"] = 64
    parameters_dict["dueling"] = True
    parameters_dict["loss"] = logcosh
    parameters_dict["learning_rate"] = 1e-4
    parameters_dict["epsilon"] = 1.
    parameters_dict["epsilon_min"] = 1e-2
    parameters_dict["epsilon_decay"] = 0.9998
    parameters_dict["use_weights"] = False
    parameters_dict["use_optimal_policy"] = False
    parameters_dict["state_scaler"] = None
    parameters_dict["value_scaler"] = None
    parameters_dict["maximum_number_of_total_samples"] = 1e6
    return parameters_dict


def computation_time(results_dir_path, experience_path, nb_runs, parameter_values, maximum_number_of_total_samples):
    mean_computing_times, sum_computing_time = [], []
    (results_dir_path / experience_path).mkdir(parents=True, exist_ok=True)

    env = env_builder()

    for value in parameter_values:
        print(value)
        parameters_dict = parameter_dict_builder()

        parameters_dict["mini_batch_size"] = value
        parameters_dict["batch_size"] = value
        parameters_dict["maximum_number_of_total_samples"] = maximum_number_of_total_samples

        agent = DQNAgent_time_builder(env, parameters_dict)
        agent.fill_memory_buffer()
        agent.train_time(nb_runs)
        mean_computing_times.append(agent.mean_training_time)
        sum_computing_time.append(agent.sum_training_time)

    plt.figure()
    plt.plot(parameter_values, mean_computing_times)
    plt.xlabel('batch_size')
    plt.ylabel("Computation time")
    # plt.savefig(results_dir_path / experience_path / ('computation_time.png'))
    np.save('../' + results_dir_path.name + '/' + experience_path.name + '/computation_time.npy', [mean_computing_times, sum_computing_time])
    plt.savefig('../' + results_dir_path.name + '/' + experience_path.name + '/computation_time.png')

def plot_comparison_computing_times(results_path, experience_path_with_gpu, experience_path_without_gpu, batch_size_values, maximum_number_of_total_samples):
    with_gpu = np.load(results_path / experience_path_with_gpu / ("computation_time.npy"))
    without_gpu = np.load(results_path / experience_path_without_gpu / ("computation_time.npy"))

    mean_with_gpu, sum_with_gpu = with_gpu[0], with_gpu[1]
    mean_without_gpu, sum_without_gpu = without_gpu[0], without_gpu[1]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(batch_size_values, mean_with_gpu, label="with gpu")
    ax1.plot(batch_size_values, mean_without_gpu, label="without gpu")
    ax1.set_title("Average computing time")
    plt.ylabel("Computation time")
    plt.xlabel("batch size")
    plt.legend()

    ax2.plot(batch_size_values, sum_with_gpu, label="with gpu")
    ax2.plot(batch_size_values, sum_without_gpu, label="without gpu")
    ax2.set_title("Sum computing time")
    plt.ylabel("Computation time")
    plt.xlabel("batch size")
    plt.legend()

    plt.text(0, 1, "Maximum number of total samples: {}".format(maximum_number_of_total_samples), transform=ax1.transAxes)
    plt.savefig('../' + results_path.name + '/comparison_computation_time.png')

if __name__ == '__main__':
    # batch_size_values_string = sys.argv[1]
    # batch_size_values = ast.literal_eval(batch_size_values_string)
    #
    # maximum_number_of_total_samples_string = sys.argv[2]
    # maximum_number_of_total_samples = ast.literal_eval(maximum_number_of_total_samples_string)

    batch_size_values = [100, 500] + [k for k in range(1000, 20000, 1000)]
    maximum_number_of_total_samples = 2e6

    results_path = Path("../Our_DQN")
    results_path.mkdir(parents=True, exist_ok=True)

    experience_path_with_gpu = Path("with_gpu")
    experience_path_without_gpu = Path("without_gpu")

    computation_time(results_path, experience_path_with_gpu, maximum_number_of_total_samples, batch_size_values,
                     maximum_number_of_total_samples)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    computation_time(results_path, experience_path_without_gpu, maximum_number_of_total_samples, batch_size_values,
                     maximum_number_of_total_samples)

    plot_comparison_computing_times(results_path, experience_path_with_gpu, experience_path_without_gpu, batch_size_values, maximum_number_of_total_samples)
