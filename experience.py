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


if __name__ == '__main__':
    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    # Parameters:
    parameters_dict = {}
    parameters_dict["env"] = env
    parameters_dict["replay_method"] = "DDQL"
    parameters_dict["batch_size"] = 32
    parameters_dict["memory_size"] = 6_000
    parameters_dict["mini_batch_size"] = 1_000
    parameters_dict["prioritized_experience_replay"] = False
    parameters_dict["target_model_update"] = 90
    parameters_dict["hidden_layer_size"] = 50
    parameters_dict["dueling"] = True
    parameters_dict["loss"] = mean_squared_error
    parameters_dict["learning_rate"] = 1e-5
    parameters_dict["epsilon"] = 0.0
    parameters_dict["epsilon_min"] = 0.0
    parameters_dict["epsilon_decay"] = 0.9995
    parameters_dict["state_weights"] = True

    # minibatch_size = int(parameters_dict["memory_size"] * percent_minibatch_size)
    # parameters_dict["mini_batch_size"] = minibatch_size

    agent = DQNAgent(env=parameters_dict["env"],
                     # state_scaler=env.get_state_scaler(), value_scaler=env.get_value_scaler(),
                     replay_method=parameters_dict["replay_method"], batch_size=parameters_dict["batch_size"],
                     memory_size=parameters_dict["memory_size"], mini_batch_size=parameters_dict["mini_batch_size"],
                     prioritized_experience_replay=parameters_dict["prioritized_experience_replay"],
                     target_model_update=parameters_dict["target_model_update"],
                     hidden_layer_size=parameters_dict["hidden_layer_size"], dueling=parameters_dict["dueling"],
                     loss=parameters_dict["loss"], learning_rate=parameters_dict["learning_rate"],
                     epsilon=parameters_dict["epsilon"], epsilon_min=parameters_dict["epsilon_min"],
                     epsilon_decay=parameters_dict["epsilon_decay"],
                     state_weights=parameters_dict["state_weights"])

    nb_episodes = 2_000
    nb_runs = 10

    # agent.init_target_network_with_true_Q_table()
    model_name = "DQL/model_initialized_with_true_q_table.h5"
    model = load_model(model_name)
    agent.set_model(model)
    agent.set_target()
    # agent.init_network_with_true_Q_table()


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
    v_display = VDisplay(after_train, agent, q_compute)
    policy_display = PolicyDisplay(after_train, agent, q_compute)

    q_error = QErrorMonitor(after_train, agent, true_compute, q_compute)
    q_error_display = QErrorDisplay(after_train, agent, q_error)

    revenue_compute = RevenueMonitor(while_training, agent, q_compute, 10_000)
    revenue_display = RevenueDisplay(after_train, agent, revenue_compute, true_revenue)

    memory_monitor = MemoryMonitor(while_training, agent)
    memory_display = MemoryDisplay(after_train, agent, memory_monitor)

    batch_monitor = BatchMonitor(while_training_after_replay_has_started, agent)
    batch_display = BatchDisplay(after_train, agent, batch_monitor)
    total_batch_display = TotalBatchDisplay(after_train, agent, batch_monitor)

    sumtree_monitor = SumtreeMonitor(while_training_after_replay_has_started, agent)
    sumtree_display = SumtreeDisplay(after_train, agent, sumtree_monitor)

    # callbacks = [true_compute, true_v_display, true_revenue,
    #              agent_monitor,
    #              q_compute, v_display, policy_display,
    #              q_error, q_error_display,
    #              revenue_compute, revenue_display,
    #              memory_monitor, memory_display,
    #              batch_monitor, batch_display, total_batch_display,
    #              sumtree_monitor, sumtree_display]
    callbacks = [true_compute, true_revenue,
                 q_compute, revenue_compute]

    agent.train(nb_episodes, callbacks)

    results_dir_name = "../Daily meetings/Short experiences/Experience 9"
    #
    # experience_dir_name = "Experience 1 of Optuna - cst epsilon = 0.001"
    # experience_dir_name = "Experience 2 of Optuna - cst epsilon = 0.001"
    # experience_dir_name = "Epsilon = 0, learning rate = 1e-5, mini batch size = 1_000"
    # experience_dir_name = "Epsilon = 0, learning rate = 1e-4, mini batch size = 1_000"
    # experience_dir_name = "Epsilon = 0, learning rate = 1e-3, mini batch size = 1_000"
    # experience_dir_name = "Epsilon = 0.01, learning rate = 1e-5, mini batch size = 1_000"
    # experience_dir_name = "Initialize network with outside model - Epsilon=0.01, lr=1e-4, mini batch size = 500"
    experience_dir_name = "Initialize network with outside model - Epsilon=0.01, lr=1e-4, mini batch size = 500"

    # experience_dir_name = "Replay"
    # experience_dir_name = "DQL"
    # experience_dir_name = "Double_DQL"
    # experience_dir_name = "Dueling_Network_Architecture"
    # experience_dir_name = "Init_network_with_true_Q-table"
    # experience_dir_name = "State_initialization"
    # experience_dir_name = "Prioritized_experience_replay"
    # experience_dir_name = "Use_weights_with_reset"

    # experience_dir_name = "Tuning_target_update"
    # experience_dir_name = "Tuning_learning_rate"
    # experience_dir_name = "Tuning_memory_size"
    # experience_dir_name = "Tuning_batch_size"

    # experience_dir_name = "Rainbow"


    callbacks_before_train = [true_compute, true_revenue]
    callbacks_after_train = [q_compute, revenue_compute]

    run_n_times_and_save(results_dir_name, experience_dir_name, parameters_dict,
                         number_of_runs=nb_runs, nb_episodes=nb_episodes, callbacks_before_train=callbacks_before_train,
                         callbacks_after_train=callbacks_after_train, model=model, init_with_true_Q_table=True)

    list_of_revenues = extract_same_files_from_several_runs(nb_first_run=0, nb_last_run=5,
                                                            results_dir_name=results_dir_name,
                                                            experience_dir_name=experience_dir_name)

    x_axis, mean_revenues, min_revenues, max_revenues = compute_statistical_results_about_list_of_revenues(
        list_of_revenues)

    mean_revenue_DP = get_DP_revenue(results_dir_name, experience_dir_name)
    mean_revenue_DQN_with_true_Q_table = get_DQL_with_true_Q_table_revenue(results_dir_name, experience_dir_name)
    references_dict = {}
    references_dict["DP revenue"] = mean_revenue_DP
    references_dict["DQL with true Q-table initialization"] = mean_revenue_DQN_with_true_Q_table

    plot_revenues(x_axis, mean_revenues, min_revenues, max_revenues, references_dict)


