# -*- coding: utf-8 -*-
import numpy as np
import gym
from keras.losses import mean_squared_error, logcosh
from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from gym_RMDCP.envs.RMDCP_env import ValueScaler
from DQL.agent import DQNAgent, DQNAgent_builder
from DQL.callbacks import TrueCompute, VDisplay, RevenueMonitor, RevenueDisplay, AgentMonitor, QCompute, QErrorDisplay, \
    QErrorMonitor, PolicyDisplay, MemoryMonitor, MemoryDisplay, BatchMonitor, BatchDisplay, TotalBatchDisplay, \
    SumtreeMonitor, SumtreeDisplay
from DQL.run_and_save_several_experiences import run_n_times_and_save, \
    compute_statistical_results_about_list_of_revenues, get_DP_revenue, get_DQL_with_true_Q_table_revenue, \
    extract_same_files_from_several_runs, plot_revenues
from keras.models import load_model
from pathlib import Path
import multiprocessing as mp
from keras.layers import K
import sys
import ast

import os

import timeit

import matplotlib.pyplot as plt

def plot_revenue_of_each_run(nb_runs, results_dir_name, experience_dir_name):
    list_of_revenues = extract_same_files_from_several_runs(nb_first_run=0, nb_last_run=nb_runs,
                                                            results_dir_name=results_dir_name,
                                                            experience_dir_name=experience_dir_name)
    x_axis = list_of_revenues[0]["revenue_compute"].replays

    for k in range(len(list_of_revenues)):
        fig = plt.figure()
        plt.plot(x_axis, list_of_revenues[k]["revenue_compute"].revenues)
        plt.savefig('../' + results_dir_name.name + '/' + experience_dir_name + '/' + str(k) + '.png')

def visualize_revenue_n_runs(nb_runs, results_dir_name, experience_dir_name, optimal_model_path, parameters_dict):
    list_of_revenues = extract_same_files_from_several_runs(nb_first_run=0, nb_last_run=nb_runs,
                                                            results_dir_name=results_dir_name,
                                                            experience_dir_name=experience_dir_name)

    x_axis, mean_revenues, min_revenues, max_revenues = compute_statistical_results_about_list_of_revenues(
        list_of_revenues)

    mean_revenue_DP = get_DP_revenue(results_dir_name, experience_dir_name)

    # mean_revenue_DQN_with_true_Q_table = get_DQL_with_true_Q_table_revenue(results_dir_name, experience_dir_name,
    #                                                                        load_model(optimal_model_path))
    references_dict = {}
    references_dict["DP revenue"] = mean_revenue_DP
    # references_dict["DQL with true Q-table initialization"] = mean_revenue_DQN_with_true_Q_table

    fig = plot_revenues(x_axis, mean_revenues, min_revenues, max_revenues, references_dict, list_of_revenues, parameters_dict)

    plt.savefig('../' + results_dir_name.name + '/' + experience_dir_name + '/' + experience_dir_name + '.png')


def launch_several_runs(parameters_dict, nb_episodes, nb_runs, results_dir_name, experience_dir_name,
                        optimal_model_path, init_with_true_Q_table):
    run_n_times_and_save(results_dir_name, experience_dir_name, parameters_dict, nb_runs, nb_episodes,
                         optimal_model_path, init_with_true_Q_table)
    visualize_revenue_n_runs(nb_runs, results_dir_name, experience_dir_name, optimal_model_path, parameters_dict)


def launch_one_run(parameters_dict, nb_episodes, optimal_model_path, init_with_true_Q_table):
    agent = DQNAgent_builder(parameters_dict["env_builder"](), parameters_dict)

    if init_with_true_Q_table:
        agent.set_model(load_model(optimal_model_path))
        agent.set_target()

    before_train = lambda episode: episode == 0
    every_episode = lambda episode: True
    while_training = lambda episode: episode % (nb_episodes / 20) == 0
    after_train = lambda episode: episode == nb_episodes - 1
    while_training_after_replay_has_started = lambda episode: len(agent.memory) > agent.batch_size and episode % (
            nb_episodes / 20) == 0

    true_compute = TrueCompute(before_train, agent)
    true_v_display = VDisplay(before_train, agent, true_compute)
    true_revenue = RevenueMonitor(before_train, agent, true_compute, 10000, name="true_revenue")

    agent_monitor = AgentMonitor(while_training, agent)

    q_compute = QCompute(while_training, agent)
    # v_display = VDisplay(after_train, agent, q_compute)
    # policy_display = PolicyDisplay(after_train, agent, q_compute)

    # q_error = QErrorMonitor(while_training, agent, true_compute, q_compute)
    # q_error_display = QErrorDisplay(after_train, agent, q_error)

    revenue_compute = RevenueMonitor(while_training, agent, q_compute, 10000)
    # revenue_display = RevenueDisplay(after_train, agent, revenue_compute, true_revenue)

    # memory_monitor = MemoryMonitor(while_training, agent)
    # memory_display = MemoryDisplay(after_train, agent, memory_monitor)
    #
    # batch_monitor = BatchMonitor(while_training_after_replay_has_started, agent)
    # batch_display = BatchDisplay(after_train, agent, batch_monitor)
    # total_batch_display = TotalBatchDisplay(after_train, agent, batch_monitor)
    #
    # sumtree_monitor = SumtreeMonitor(while_training_after_replay_has_started, agent)
    # sumtree_display = SumtreeDisplay(after_train, agent, sumtree_monitor)

    callbacks = [
        true_compute, true_v_display, true_revenue,
                 agent_monitor,
                 q_compute,
                 # v_display, policy_display,
                 # q_error, q_error_display,
                 revenue_compute,
                 # revenue_display,
                 # memory_monitor, memory_display,
                 # batch_monitor, batch_display, total_batch_display,
                 # sumtree_monitor, sumtree_display
                 ]

    agent.train(nb_episodes, callbacks)

    plt.plot(revenue_compute.replays, revenue_compute.revenues)
    plt.ylabel("Revenues")
    plt.xlabel("Number of replays")
    plt.show()

    return agent, callbacks


def tune_parameter(general_dir_name, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs,
                   optimal_model_path, init_with_true_Q_table):
    # results_dir_name = "../Daily meetings/Stabilization experiences/" + parameter
    results_dir_name = general_dir_name / parameter
    results_dir_name.mkdir(parents=True, exist_ok=True)

    for k in parameter_values:
        print("Running with "+parameter+" = "+str(k))
        parameters_dict[parameter] = k
        experience_dir_name = parameter + " = " + str(parameters_dict[parameter])
        launch_several_runs(parameters_dict, nb_episodes, nb_runs, results_dir_name, experience_dir_name,
                            optimal_model_path, init_with_true_Q_table)


def save_optimal_model(parameters_dict, model_name):
    agent = DQNAgent_builder(parameters_dict["env_builder"](), parameters_dict)
    agent.init_network_with_true_Q_table()
    print("Saving optimal model")
    agent.model.save(model_name)

def plot_computation_times(parameter, parameter_values, nb_runs, results_dir_name):
    computation_times = []
    for value in parameter_values:
        computation_times_value = []
        experience_dir_name = str(value)
        for k in range(nb_runs):
            run_path = results_dir_name / experience_dir_name / ('Run_' + str(k))
            for file_path in run_path.iterdir():
                file_name = file_path.stem
                if file_name == ("computation_time"+str(k)):
                    computation_times_value.append(np.load(file_path))
        computation_times.append(np.mean(computation_times_value))

    plt.figure()
    plt.plot(parameter_values, computation_times)
    plt.xlabel(parameter)
    plt.ylabel("Computation time")
    plt.savefig('../' + results_dir_name.name + '/computation_time.png')


def env_builder():
    # Parameters of the environment
    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 20))
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
    parameters_dict["dueling"] = False
    parameters_dict["loss"] = logcosh
    parameters_dict["learning_rate"] = 1e-4
    parameters_dict["epsilon"] = 1.
    parameters_dict["epsilon_min"] = 1e-2
    parameters_dict["epsilon_decay"] = 0.9998
    parameters_dict["use_weights"] = False
    parameters_dict["use_optimal_policy"] = False
    parameters_dict["state_scaler"] = None
    parameters_dict["value_scaler"] = ValueScaler
    parameters_dict["maximum_number_of_total_samples"] = 10000000000
    return parameters_dict



if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # if len(sys.argv) != 3:
    #     print("Specify parameter and parameter values.")
    #     exit(0)
    # Parameters of the agent

    # Loading the model with the optimal weights which will be used to initialize the network of the agent if init_with_true_Q_table
    parameters_dict = parameter_dict_builder()
    dueling_model_name = "DQL/model_initialized_with_true_q_table.h5"
    # save_optimal_model(parameters_dict, dueling_model_name)

    optimal_model_path = dueling_model_name
    init_with_true_Q_table = False

    # Parameters of the experience
    nb_episodes = 40000
    nb_runs = 30

    results_path = Path("../Our DQN")
    results_path.mkdir(parents=True, exist_ok=True)
    experience_dir_name = "linear activation function"

    launch_several_runs(parameters_dict, nb_episodes, nb_runs, results_path, experience_dir_name,
                        optimal_model_path, init_with_true_Q_table)
    # parameter = "mini_batch_size"
    # parameter_values = [10, 100]
    # plot_revenue_of_each_run(nb_runs, results_path, experience_dir_name)
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, optimal_model_path,
    #                init_with_true_Q_table)
    # plot_computation_times(parameter, parameter_values, nb_runs, results_path)

    launch_several_runs(parameters_dict, nb_episodes, nb_runs, results_path, experience_dir_name, optimal_model_path,
                        init_with_true_Q_table)

    # experience_dir_name = "control_experiment"
    # visualize_revenue_n_runs(nb_runs, results_path, experience_dir_name, optimal_model_path)
    #
    # parameters_dict = parameter_dict_builder()
    # experience_dir_name = "no_extension"
    # launch_several_runs(parameters_dict, nb_episodes, nb_runs, results_path, experience_dir_name, optimal_model_path, init_with_true_Q_table)

    # parameters_dict = parameter_dict_builder()
    # parameters_dict["dueling"] = True
    # experience_dir_name = "dueling"
    # launch_several_runs(parameters_dict, nb_episodes, nb_runs, results_path, experience_dir_name, optimal_model_path, init_with_true_Q_table)
    #
    # parameters_dict = parameter_dict_builder()
    # parameters_dict["use_weights"] = True
    # experience_dir_name = "weights"
    # launch_several_runs(parameters_dict, nb_episodes, nb_runs, results_path, experience_dir_name, optimal_model_path, init_with_true_Q_table)
    #
    # parameters_dict = parameter_dict_builder()
    # parameters_dict["use_weights"] = True
    # parameters_dict["dueling"] = True
    # parameter = "epsilon_decay"
    # parameter_values = [0.9, 0.99, 0.999, 0.9999, 0.99999]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, optimal_model_path, init_with_true_Q_table)
    #
    # parameters_dict = parameter_dict_builder()
    # parameters_dict["use_weights"] = True
    # parameters_dict["dueling"] = True
    # parameter = "hidden_layer_size"
    # parameter_values = [10, 30, 65, 100, 200]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, optimal_model_path, init_with_true_Q_table)
    #
    #
    # parameters_dict = parameter_dict_builder()
    # parameters_dict["use_weights"] = True
    # parameters_dict["dueling"] = True
    # parameter = "learning_rate"
    # parameter_values = [1e-5, 1e-4, 1e-3, 1e-2]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, optimal_model_path, init_with_true_Q_table)


    # launch_one_run(parameters_dict, nb_episodes, dueling_model_name, init_with_true_Q_table)

    # Tuning of the parameters
    # parameter = "learning_rate"
    # parameter = sys.argv[1]
    # parameter_values_string = sys.argv[2]
    # parameter_values = ast.literal_eval(parameter_values_string)
    # parameter_values = [1e-5, 1e-4, 1e-3, 1e-2]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, optimal_model_path,
    #                init_with_true_Q_table)


    # parameter = "mini_batch_size"
    # parameter_values = [1000, 500, 100, 10]
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, model, init_with_true_Q_table)
    #
    # results_dir_name = "../Daily meetings/Stabilization experiences/" + parameter
    # experience_dir_name = parameter + " = " + str(1e-5)
    # visualize_revenue_n_runs(1, results_dir_name, experience_dir_name, model)
    #
    # launch_one_run(parameters_dict, nb_episodes, model, init_with_true_Q_table)
