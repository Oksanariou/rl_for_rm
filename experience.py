# -*- coding: utf-8 -*-
import numpy as np
import gym
from keras.losses import mean_squared_error, logcosh
from dynamic_programming_env_DCP import dynamic_programming_env_DCP

from DQL.agent import DQNAgent
from DQL.callbacks import TrueCompute, VDisplay, RevenueMonitor, RevenueDisplay, AgentMonitor, QCompute, QErrorDisplay, \
    QErrorMonitor, PolicyDisplay, MemoryMonitor, MemoryDisplay, BatchMonitor, BatchDisplay, TotalBatchDisplay, \
    SumtreeMonitor, SumtreeDisplay
from DQL.run_and_save_several_experiences import run_n_times_and_save, \
    compute_statistical_results_about_list_of_revenues, get_DP_revenue, get_DQL_with_true_Q_table_revenue, \
    extract_same_files_from_several_runs, plot_revenues
from keras.models import load_model
from keras.layers import K

import os

import timeit

import matplotlib.pyplot as plt

def visualize_revenue_n_runs(nb_runs, results_dir_name, experience_dir_name, model):
    list_of_revenues = extract_same_files_from_several_runs(nb_first_run=0, nb_last_run=nb_runs,
                                                            results_dir_name=results_dir_name,
                                                            experience_dir_name=experience_dir_name)

    x_axis, mean_revenues, min_revenues, max_revenues = compute_statistical_results_about_list_of_revenues(
        list_of_revenues)

    mean_revenue_DP = get_DP_revenue(results_dir_name, experience_dir_name)
    mean_revenue_DQN_with_true_Q_table = get_DQL_with_true_Q_table_revenue(results_dir_name, experience_dir_name, model)
    references_dict = {}
    references_dict["DP revenue"] = mean_revenue_DP
    references_dict["DQL with true Q-table initialization"] = mean_revenue_DQN_with_true_Q_table

    fig = plot_revenues(x_axis, mean_revenues, min_revenues, max_revenues, references_dict)

    plt.savefig(results_dir_name + "/" + experience_dir_name + "/" + experience_dir_name + ".png")

def launch_several_runs(parameters_dict, nb_episodes, nb_runs, results_dir_name, experience_dir_name, model, init_with_true_Q_table):

    run_n_times_and_save(results_dir_name, experience_dir_name, parameters_dict, nb_runs, nb_episodes,
                         model, init_with_true_Q_table)
    visualize_revenue_n_runs(nb_runs, results_dir_name, experience_dir_name, model)

def launch_one_run(parameters_dict, nb_episodes, model, init_with_true_Q_table):

    agent = DQNAgent(parameters_dict["env"])

    for key in parameters_dict:
        agent.__setattr__(key, parameters_dict[key])
    agent.model = agent._build_model()
    agent.target_model = agent._build_model()

    if init_with_true_Q_table:
        agent.set_model(model)
        agent.set_target()

    before_train = lambda episode: episode == 0
    every_episode = lambda episode: True
    while_training = lambda episode: episode % (nb_episodes / 20) == 0
    after_train = lambda episode: episode == nb_episodes - 1
    while_training_after_replay_has_started = lambda episode: len(agent.memory) > agent.batch_size and episode % (
            nb_episodes / 20) == 0

    true_compute = TrueCompute(before_train, agent)
    true_v_display = VDisplay(before_train, agent, true_compute)
    true_revenue = RevenueMonitor(before_train, agent, true_compute, 10_000, name="true_revenue")

    agent_monitor = AgentMonitor(while_training, agent)

    q_compute = QCompute(while_training, agent)
    # v_display = VDisplay(after_train, agent, q_compute)
    # policy_display = PolicyDisplay(after_train, agent, q_compute)

    q_error = QErrorMonitor(while_training, agent, true_compute, q_compute)
    q_error_display = QErrorDisplay(after_train, agent, q_error)

    revenue_compute = RevenueMonitor(while_training, agent, q_compute, 10_000)
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
                 q_error, q_error_display,
                 revenue_compute,
                 # revenue_display,
                 # memory_monitor, memory_display,
                 # batch_monitor, batch_display, total_batch_display,
                 # sumtree_monitor, sumtree_display
                 ]

    agent.train(nb_episodes, callbacks)

    plt.plot(revenue_compute.replays, revenue_compute.revenues)
    plt.ylim(820, 1070)
    plt.ylabel("Revenues")
    plt.xlabel("Number of replays")
    plt.show()
    plt.show()


def tune_parameter(general_dir_name, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, model, init_with_true_Q_table):

    # results_dir_name = "../Daily meetings/Stabilization experiences/" + parameter
    results_dir_name = general_dir_name + "/" + parameter
    os.mkdir(results_dir_name)

    for k in parameter_values:
        parameters_dict[parameter] = k
        experience_dir_name = parameter + " = " + str(parameters_dict[parameter])
        launch_several_runs(parameters_dict, nb_episodes, nb_runs, results_dir_name, experience_dir_name, model, init_with_true_Q_table)

def save_optimal_model(parameters_dict, model_name):
    agent = DQNAgent(parameters_dict["env"])
    for key in parameters_dict:
        agent.__setattr__(key, parameters_dict[key])
    agent.model = agent._build_model()
    agent.target_model = agent._build_model()
    agent.init_network_with_true_Q_table()

    agent.model.save(model_name)

if __name__ == '__main__':
    #Parameters of the environment
    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)


    #Parameters of the agent
    init_with_true_Q_table = False

    parameters_dict = {}
    parameters_dict["env"] = env
    parameters_dict["replay_method"] = "DDQL"
    parameters_dict["batch_size"] = 32
    parameters_dict["memory_size"] = 6_000
    parameters_dict["mini_batch_size"] = 100
    parameters_dict["prioritized_experience_replay"] = False
    parameters_dict["target_model_update"] = 90
    parameters_dict["hidden_layer_size"] = 50
    parameters_dict["dueling"] = True
    parameters_dict["loss"] = mean_squared_error
    parameters_dict["learning_rate"] = 1e-3
    parameters_dict["epsilon"] = 1e-2
    parameters_dict["epsilon_min"] = 1e-2
    parameters_dict["epsilon_decay"] = 1
    parameters_dict["use_weights"] = True
    parameters_dict["use_optimal_policy"] = False


    #Loading the model with the optimal weights which will be used to initialize the network of the agent if init_with_true_Q_table
    dueling_model_name = "DQL/model_initialized_with_true_q_table.h5"
    # save_optimal_model(dueling_model_name)
    model = load_model(dueling_model_name)


    #Parameters of the experience
    nb_episodes = 10_000
    nb_runs = 20

    general_dir_name = "../Results"
    os.mkdir(general_dir_name) #Creation of the folder where the results of the experience will be stocked

    #Tuning of the parameters
    parameter = "learning_rate"
    parameter_values = [1e-5, 1e-4, 1e-3, 1e-2]
    tune_parameter(general_dir_name, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, model, init_with_true_Q_table)

    parameter = "epsilon"
    parameter_values = [1e-4, 1e-3, 1e-2, 1e-1]
    tune_parameter(general_dir_name, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, model, init_with_true_Q_table)

    parameter = "mini_batch_size"
    parameter_values = [1000, 500, 100, 10]
    tune_parameter(general_dir_name, parameter, parameter_values, parameters_dict, nb_episodes, nb_runs, model, init_with_true_Q_table)

    results_dir_name = "../Daily meetings/Stabilization experiences/" + parameter
    experience_dir_name = parameter + " = " + str(1e-5)
    visualize_revenue_n_runs(1, results_dir_name, experience_dir_name, model)

    launch_one_run(parameters_dict, nb_episodes, model, init_with_true_Q_table)



