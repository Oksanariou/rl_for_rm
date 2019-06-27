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

def main():
    data_collection_points = 4
    micro_times = 3
    capacity = 4
    actions = tuple(k for k in range(50, 231, 50))
    alpha = 0.8
    lamb = 0.7

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    # Parameters:
    parameters_dict = {}
    parameters_dict["env"] = env
    parameters_dict["replay_method"] = "DDQL"
    parameters_dict["batch_size"] = 30
    parameters_dict["memory_size"] = 5000
    parameters_dict["prioritized_experience_replay"] = False
    parameters_dict["target_model_update"] = 100
    parameters_dict["hidden_layer_size"] = 50
    parameters_dict["dueling"] = False
    parameters_dict["loss"] = mean_squared_error
    parameters_dict["learning_rate"] = 0.001
    parameters_dict["epsilon"] = 1.0
    parameters_dict["epsilon_min"] = 0.02
    parameters_dict["epsilon_decay"] = 0.9995
    parameters_dict["state_weights"] = None

    agent = DQNAgent(env=parameters_dict["env"],
                     # state_scaler=env.get_state_scaler(), value_scaler=env.get_value_scaler(),
                     replay_method=parameters_dict["replay_method"], batch_size=parameters_dict["batch_size"],
                     memory_size=parameters_dict["memory_size"],
                     prioritized_experience_replay=parameters_dict["prioritized_experience_replay"],
                     target_model_update=parameters_dict["target_model_update"],
                     hidden_layer_size=parameters_dict["hidden_layer_size"], dueling=parameters_dict["dueling"],
                     loss=parameters_dict["loss"], learning_rate=parameters_dict["learning_rate"],
                     epsilon=parameters_dict["epsilon"], epsilon_min=parameters_dict["epsilon_min"],
                     epsilon_decay=parameters_dict["epsilon_decay"],
                     state_weights=parameters_dict["state_weights"])

    nb_episodes = 10_000

    # agent.init_target_network_with_true_Q_table()
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

    agent_monitor = AgentMonitor(every_episode, agent)

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

    callbacks = [true_compute, true_v_display, true_revenue,
                 agent_monitor,
                 q_compute, v_display, policy_display,
                 q_error, q_error_display,
                 revenue_compute, revenue_display,
                 memory_monitor, memory_display,
                 batch_monitor, batch_display, total_batch_display,
                 sumtree_monitor, sumtree_display]

    agent.train(nb_episodes, callbacks)

    results_dir_name = "DQL-Results"
    experience_dir_name = "Initialize_networks_with_true_Q_table"

    callbacks_before_train = [true_compute, true_revenue]
    callbacks_after_train = [q_compute, q_error, revenue_compute]

    run_n_times_and_save(results_dir_name, experience_dir_name, parameters_dict,
                         number_of_runs=2, nb_episodes=nb_episodes, callbacks_before_train=callbacks_before_train,
                         callbacks_after_train=callbacks_after_train, init_with_true_Q_table=True)

    list_of_revenues = extract_same_files_from_several_runs(nb_first_run=0, nb_last_run=2,
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
