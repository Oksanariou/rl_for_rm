# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from functools import partial
from multiprocessing import Pool
from keras.models import load_model

from visualization_and_metrics import average_n_episodes, q_to_policy_RM
from scipy.stats import sem, t

from DQL.agent import DQNAgent
from DQL.callbacks import TrueCompute, RevenueMonitor, QCompute


def pickle_that(obj, path):
    with path.open("wb") as f:
        pickle.dump(obj, f)


def unpickle_that(path):
    with path.open("rb") as f:
        return pickle.load(f)


def run_once_and_save(experience_path, parameters_dict, nb_episodes, optimal_model_path, init_with_true_Q_table, k):
    print("Run {}".format(k))

    run_path = experience_path / ('Run_' + str(k))
    run_path.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent(parameters_dict["env_builder"]())
    for key in parameters_dict:
        agent.__setattr__(key, parameters_dict[key])
    agent.model = agent._build_model()
    agent.target_model = agent._build_model()

    if init_with_true_Q_table:
        agent.set_model(load_model(optimal_model_path))
        agent.set_target()
        # agent.init_network_with_true_Q_table()

    before_train = lambda episode: episode == 0
    while_training = lambda episode: episode % (nb_episodes / 20) == 0

    true_compute = TrueCompute(before_train, agent)
    true_revenue = RevenueMonitor(before_train, agent, true_compute, 10000, name="true_revenue")
    q_compute = QCompute(while_training, agent)
    revenue_compute = RevenueMonitor(while_training, agent, q_compute, 10000)

    callbacks = [true_compute, true_revenue, q_compute, revenue_compute]

    agent.train(nb_episodes, callbacks)

    if k == 0:
        pickle_that(true_compute, experience_path / true_compute.name)
        pickle_that(true_revenue, experience_path / true_revenue.name)

    pickle_that(agent, run_path / 'agent')
    pickle_that(q_compute, run_path / q_compute.name)
    pickle_that(revenue_compute, run_path / revenue_compute.name)


def run_n_times_and_save(results_dir_name, experience_dir_name, parameters_dict, number_of_runs, nb_episodes,
                         optimal_model_path, init_with_true_Q_table=False):
    experience_path = results_dir_name / experience_dir_name
    experience_path.mkdir(parents=True, exist_ok=True)

    pickle_that(parameters_dict["env_builder"](), experience_path / 'Environment')

    f = partial(run_once_and_save, experience_path, parameters_dict, nb_episodes, optimal_model_path,
                init_with_true_Q_table)
    with Pool(number_of_runs) as pool:
        pool.map(f, range(number_of_runs))


def extract_files_from_a_run(results_dir_name, experience_dir_name, run_number, list_of_file_names):
    """
        Input: Name of the experience, number of the run from which we want to extract the files, list of the names of the files that we want to extract
        Output: Dictionary containing the files of one run which names correspond to the names in callback_name
    """

    run_path = results_dir_name / experience_dir_name / ('Run_' + str(run_number))

    dict_of_files = {}

    for file_path in run_path.iterdir():
        file_name = file_path.stem

        if file_name in list_of_file_names:
            print("Collecting " + file_name + "...")
            dict_of_files[file_name] = unpickle_that(file_path)

    return (dict_of_files)


def extract_files_from_experience(results_dir_name, experience_dir_name, list_of_file_names):
    experience_path = results_dir_name / experience_dir_name

    dict_of_files = {}
    for file_path in experience_path.iterdir():
        if not file_path.is_file():
            continue
        file_name = file_path.stem
        print("Collecting " + file_name + "...")
        dict_of_files[file_name] = unpickle_that(file_path)

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

    x_axis = list_of_revenues[0][file_name].replays
    # x_axis = [k for k in range(0, nb_episodes, nb_episodes // nb_collection_points)]

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


def get_DQL_with_true_Q_table_revenue(results_dir_name, experience_dir_name, model, run_number=0, agent_name="agent"):
    agent = extract_files_from_a_run(results_dir_name, experience_dir_name, run_number, [agent_name])[agent_name]
    agent.set_model(model)
    agent.set_target()
    Q_table_init = agent.compute_q_table()
    policy_init = q_to_policy_RM(agent.env, Q_table_init)
    revenue = average_n_episodes(agent.env, policy_init.flatten(), 10000, agent.epsilon_min)

    return revenue


def plot_revenues(x_axis, mean_revenues, min_revenues, max_revenues, references_dict, comparison=[]):
    fig = plt.figure()

    plt.plot(x_axis, mean_revenues, color="gray", label='DQL mean revenue')
    plt.fill_between(x_axis, min_revenues, max_revenues, label='95% confidence interval', color="gray", alpha=0.2)

    if len(comparison) > 0:
        plt.plot(x_axis, comparison)

    for y_name in references_dict:
        plt.plot(x_axis, [references_dict[y_name]] * len(x_axis), label=y_name)
        # plt.ylim(820, 1070)

    plt.legend()
    plt.ylabel("Revenues")
    plt.xlabel("Number of replays")
    # plt.show()
    return fig
