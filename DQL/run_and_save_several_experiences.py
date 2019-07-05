# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import dill as pickle

from visualization_and_metrics import average_n_episodes, q_to_policy_RM
from scipy.stats import sem, t

from DQL.agent import DQNAgent
from DQL.callbacks import TrueCompute, RevenueMonitor, QCompute, QErrorMonitor


def run_n_times_and_save(results_dir_name, experience_dir_name, parameters_dict, number_of_runs, nb_episodes,
                         callbacks_before_train, callbacks_after_train, init_with_true_Q_table=False):
    os.mkdir(results_dir_name + '/' + experience_dir_name)

    pickle_out = open(results_dir_name + '/' + experience_dir_name + "/Environment", "wb")
    pickle.dump(parameters_dict["env"], pickle_out)
    pickle_out.close()

    for callback in callbacks_before_train:
        callback._run()
        pickle_out = open(results_dir_name + '/' + experience_dir_name + "/" + callback.name, "wb")
        pickle.dump(callback, pickle_out)
        pickle_out.close()

    for k in range(number_of_runs):
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

        if init_with_true_Q_table:
            agent.init_network_with_true_Q_table()

        for callback in callbacks_after_train:
            callback.reset(agent)

        run_dir_name = results_dir_name + '/' + experience_dir_name + '/Run_' + str(k)
        os.mkdir(run_dir_name)

        agent.train(nb_episodes, callbacks_before_train + callbacks_after_train)

        pickle_out = open(run_dir_name + "/" + "agent", "wb")
        pickle.dump(agent, pickle_out)
        pickle_out.close()

        for callback in callbacks_after_train:
            pickle_out = open(run_dir_name + "/" + callback.name, "wb")
            pickle.dump(callback, pickle_out)
            pickle_out.close()


def extract_files_from_a_run(results_dir_name, experience_dir_name, run_number, list_of_file_names):
    """
        Input: Name of the experience, number of the run from which we want to extract the files, list of the names of the files that we want to extract
        Output: Dictionary containing the files of one run which names correspond to the names in callback_name
    """
    dict_of_files = {}
    for dir_name in sorted(os.listdir(results_dir_name + '/' + experience_dir_name)):
        if dir_name == "Run_" + str(run_number):
            for file_name in os.listdir(results_dir_name + '/' + experience_dir_name + "/Run_" + str(run_number)):
                if file_name in list_of_file_names:
                    print("Collecting " + file_name + "...")
                    pickle_in = open(
                        results_dir_name + '/' + experience_dir_name + "/Run_" + str(run_number) + "/" + file_name,
                        "rb")
                    dict_of_files[file_name] = pickle.load(pickle_in)
                    pickle_in.close()
    return (dict_of_files)


def extract_files_from_experience(results_dir_name, experience_dir_name, list_of_file_names):
    dict_of_files = {}
    for file_name in os.listdir(results_dir_name + '/' + experience_dir_name):
        if file_name in list_of_file_names:
            print("Collecting " + file_name + "...")
            pickle_in = open(results_dir_name + '/' + experience_dir_name + "/" + file_name, "rb")
            dict_of_files[file_name] = pickle.load(pickle_in)
            pickle_in.close()
    return (dict_of_files)


def extract_same_files_from_several_runs(nb_first_run, nb_last_run, results_dir_name, experience_dir_name,
                                         file_name="revenue_compute"):
    """
        Input: Number of the first run from which we want to get the file, number of the last run from which we want to get the file, name of the experience, name of the file that we want to extract
        Output: List containing as many dictionaries as the number of runs from which we want to extract the file, with each dictionary containing the file that we want to extract
    """
    list_of_files = []
    for k in range(nb_first_run, nb_last_run):
        list_of_files.append(extract_files_from_a_run(results_dir_name, experience_dir_name, k, [file_name]))
    return list_of_files


def compute_statistical_results_about_list_of_revenues(list_of_revenues, file_name="revenue_compute", confidence=0.95):
    nb_collection_points = len(list_of_revenues[0][file_name].revenues)
    nb_episodes = (list_of_revenues[0][file_name].replays[-1] - list_of_revenues[0][file_name].replays[
        -2]) * nb_collection_points

    x_axis = [k for k in range(0, nb_episodes, nb_episodes // nb_collection_points)]

    all_revenues_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]
    for k in range(len(list_of_revenues)):
        revenues = list_of_revenues[k][file_name].revenues
        for i in range(nb_collection_points):
            all_revenues_combined_at_each_collection_point[i].append(revenues[i])

    mean_revenues = [np.mean(list) for list in all_revenues_combined_at_each_collection_point]
    std_revenues = [sem(list) for list in all_revenues_combined_at_each_collection_point]
    confidence_revenues = [std_revenues[k] * t.ppf((1 + confidence) / 2, nb_collection_points - 1) for k in
                           range(nb_collection_points)]
    min_revenues = [mean_revenues[k] - confidence_revenues[k] for k in range(nb_collection_points)]
    max_revenues = [mean_revenues[k] + confidence_revenues[k] for k in range(nb_collection_points)]

    return x_axis, mean_revenues, min_revenues, max_revenues


def get_DP_revenue(results_dir_name, experience_dir_name, file_name="true_revenue"):
    true_revenue = extract_files_from_experience(results_dir_name, experience_dir_name, [file_name])[file_name]
    DP_revenue = true_revenue.revenues[0]

    return DP_revenue


def get_DQL_with_true_Q_table_revenue(results_dir_name, experience_dir_name, run_number=0, agent_name="agent"):
    agent = extract_files_from_a_run(results_dir_name, experience_dir_name, run_number, [agent_name])[agent_name]
    agent.init_network_with_true_Q_table()
    Q_table_init = agent.compute_q_table()
    policy_init = q_to_policy_RM(agent.env, Q_table_init)
    revenue = average_n_episodes(agent.env, policy_init.flatten(), 10000, agent.epsilon_min)

    return revenue


def plot_revenues(x_axis, mean_revenues, min_revenues, max_revenues, references_dict, comparison=[]):
    plt.plot(x_axis, mean_revenues, color="gray", label='DQL mean revenue')
    plt.fill_between(x_axis, min_revenues, max_revenues, label='95% confidence interval', color="gray", alpha=0.2)

    if len(comparison) > 0:
        plt.plot(x_axis, comparison)

    for y_name in references_dict:
        plt.plot(x_axis, [references_dict[y_name]] * len(x_axis), label=y_name)

    plt.legend()
    plt.ylabel("Revenues")
    plt.xlabel("Number of episodes")
    plt.show()
