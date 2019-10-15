import gym
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import glob

from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from visualization_and_metrics import average_n_episodes, visualizing_epsilon_decay
from keras_rl_experience import run_once
from q_learning import q_learning
import random

from functools import partial
from multiprocessing import Pool


def env_builder(parameters_dict):
    actions = tuple(k for k in range(parameters_dict["action_min"], parameters_dict["action_max"],
                                     parameters_dict["action_offset"]))
    return gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=parameters_dict["data_collection_points"],
                    capacity=parameters_dict["capacity"],
                    micro_times=parameters_dict["micro_times"], actions=actions, alpha=parameters_dict["alpha"],
                    lamb=parameters_dict["lambda"], compute_P_matrix=parameters_dict["compute_P_matrix"],
                    transition_noise_percentage=parameters_dict["transition_noise_percentage"],
                    parameter_noise_percentage=parameters_dict["parameter_noise_percentage"])


def env_parameters():
    parameters_dict = {}
    parameters_dict["data_collection_points"] = 101
    parameters_dict["micro_times"] = 1
    parameters_dict["capacity"] = 51
    parameters_dict["action_min"] = 50
    parameters_dict["action_max"] = 231
    parameters_dict["action_offset"] = 20
    parameters_dict["lambda"] = 0.7
    parameters_dict["alpha"] = 0.7
    parameters_dict["compute_P_matrix"] = True
    parameters_dict["transition_noise_percentage"] = 0
    parameters_dict["parameter_noise_percentage"] = 0
    return parameters_dict


def agent_parameters_dict_DQL():
    parameters_dict = {}
    parameters_dict["nb_steps_warmup"] = 1000
    parameters_dict["enable_double_dqn"] = True
    parameters_dict["enable_dueling_network"] = True
    parameters_dict["target_model_update"] = 100
    parameters_dict["batch_size"] = 128
    parameters_dict["hidden_layer_size"] = 100
    parameters_dict["layers_nb"] = 3
    parameters_dict["memory_buffer_size"] = 50000
    parameters_dict["epsilon"] = 0.2
    parameters_dict["learning_rate"] = 1e-4
    return parameters_dict


def agent_parameters_dict_QL(nb_steps):
    parameters_dict = {}
    parameters_dict["alpha"] = 0.8
    parameters_dict["alpha_min"] = 0.0001
    parameters_dict["nb_steps"] = nb_steps
    parameters_dict["alpha_decay"] = 10 ** (np.log(parameters_dict["alpha_min"] / parameters_dict["alpha"]) / (
            3 * parameters_dict["nb_steps"]))  # 0.99999
    parameters_dict["gamma"] = 0.99
    parameters_dict["epsilon"] = 1
    parameters_dict["epsilon_min"] = 0.01
    parameters_dict["epsilon_decay"] = 10 ** (np.log(parameters_dict["epsilon_min"] / parameters_dict["epsilon"]) / (
            3 * parameters_dict["nb_steps"]))  # 0.99999
    return parameters_dict


def run_once_QL(env_builder, env_parameters_dict, parameters_dict, nb_episodes, experience_name, callback_frequency, k):
    env = env_builder(env_parameters_dict)
    q_table, nb_episodes_list, diff_with_policy_opt_list, M, trajectories, revenues_QL = q_learning(env=env, alpha=
    parameters_dict["alpha"], alpha_min=parameters_dict["alpha_min"], alpha_decay=parameters_dict["alpha_decay"], gamma=
                                                                                                    parameters_dict[
                                                                                                        "gamma"],
                                                                                                    nb_episodes=nb_episodes,
                                                                                                    epsilon=
                                                                                                    parameters_dict[
                                                                                                        "epsilon"],
                                                                                                    epsilon_min=
                                                                                                    parameters_dict[
                                                                                                        "epsilon_min"],
                                                                                                    epsilon_decay=
                                                                                                    parameters_dict[
                                                                                                        "epsilon_decay"],
                                                                                                    callback_frequency=callback_frequency)
    np.save(experience_name / ("Run" + str(k) + ".npy"), revenues_QL)


def run_once_random(env_builder, env_parameters_dict, experience_name, k):
    env = env_builder(env_parameters_dict)
    random_P = np.array([random.randint(0, env.nA - 1) for k in range(env.nS)]).reshape(env.T, env.C)
    revenue = env.average_n_episodes(random_P, 10000)
    np.save(experience_name / ("Run" + str(k) + ".npy"), revenue[0])


if __name__ == '__main__':
    nb_runs = 20
    callback_frequency = 2

    env_param = env_parameters()
    env_param["data_collection_points"] = 13
    env_param["capacity"] = 6
    print("capacity = {}".format(env_param["capacity"] - 1))
    env = env_builder(env_param)
    initial_true_V, initial_true_P = dynamic_programming_env_DCP(env)
    initial_true_revenues, initial_true_bookings = average_n_episodes(env, initial_true_P, 10000)

    experience_name_random = Path("../Results/Random_capacity_" + str(env_param["capacity"]))
    experience_name_random.mkdir(parents=True, exist_ok=True)
    # f = partial(run_once_random, env_builder, env_param, experience_name_random)
    # with Pool(nb_runs) as pool:
    #     pool.map(f, range(nb_runs))
    list_of_rewards_random = []
    for np_name in glob.glob(str(experience_name_random) + '/*.np[yz]'):
        list_of_rewards_random.append(np.load(np_name, allow_pickle=True))
    average_initial_random_revenue = np.mean(list_of_rewards_random)
    initial_random_percentage = (average_initial_random_revenue / initial_true_revenues) * 100
    print("Random percentage of true revenue = {}".format(initial_random_percentage))

    nb_timesteps = 80001
    absc = [k for k in range(0, nb_timesteps, nb_timesteps // callback_frequency)]

    experience_name_DQL = Path("../Results/DQL_capacity_" + str(env_param["capacity"]))
    experience_name_DQL.mkdir(parents=True, exist_ok=True)
    param_dict_DQL = agent_parameters_dict_DQL()
    # f = partial(run_once, env_builder, env_param, param_dict_DQL, nb_timesteps, experience_name_DQL, callback_frequency)
    # with Pool(nb_runs) as pool:
    #     pool.map(f, range(nb_runs))
    # for k in range(nb_runs):
    #     run_once(env_builder, env_param, param_dict_DQL, nb_timesteps, experience_name_DQL, callback_frequency, k)
    list_of_rewards_DQL, mean_revenues_DQL, mean_bookings_DQL, min_revenues_DQL, max_revenues_DQL = env.collect_revenues(
        experience_name_DQL)
    average_initial_DQL_revenue = mean_revenues_DQL[-1]
    initial_DQL_percentage = (average_initial_DQL_revenue / initial_true_revenues) * 100
    print("DQL percentage of true revenue = {}".format(initial_DQL_percentage))

    experience_name_QL = Path("../Results/QL_capacity_" + str(env_param["capacity"]))
    experience_name_QL.mkdir(parents=True, exist_ok=True)
    param_dict_QL = agent_parameters_dict_QL(nb_timesteps)
    f = partial(run_once_QL, env_builder, env_param, param_dict_QL, nb_timesteps, experience_name_QL,
                callback_frequency)
    # with Pool(nb_runs) as pool:
    #     pool.map(f, range(nb_runs))
    # for k in range(nb_runs):
    #     run_once_QL(env_builder, env_param, param_dict_QL, nb_timesteps, experience_name_QL, callback_frequency, k)
    list_of_rewards_QL, mean_revenues_QL, mean_bookings_QL, min_revenues_QL, max_revenues_QL = env.collect_revenues(
        experience_name_QL)
    average_initial_QL_revenue = mean_revenues_QL[-1]
    initial_QL_percentage = (average_initial_QL_revenue / initial_true_revenues) * 100
    print("QL percentage of true revenue = {}".format(initial_QL_percentage))

    capacities = [k for k in range(10, 151, 10)] + [k for k in range(180, 241, 30)]
    optimal_revenues = [initial_true_revenues]
    DQL_percentage = [average_initial_DQL_revenue]
    DQL_min_revenues = [min_revenues_DQL[-1]]
    DQL_max_revenues = [max_revenues_DQL[-1]]
    QL_percentage = [average_initial_QL_revenue]
    QL_min_revenues = [min_revenues_QL[-1]]
    QL_max_revenues = [max_revenues_QL[-1]]
    random_percentage = [average_initial_random_revenue]
    random_min_revenues = [np.min(list_of_rewards_random)]
    random_max_revenues = [np.max(list_of_rewards_random)]

    for capacity in capacities:
        print("capacity = {}".format(capacity))
        env_param = env_parameters()
        env_param["data_collection_points"] = (2 * capacity) + 1
        env_param["capacity"] = capacity + 1
        env = env_builder(env_param)
        true_V, true_P = dynamic_programming_env_DCP(env)
        true_revenues, true_bookings = average_n_episodes(env, true_P, 10000)
        optimal_revenues.append(true_revenues)

        experience_name_random = Path("../Results/Random_capacity_" + str(env_param["capacity"]))
        experience_name_random.mkdir(parents=True, exist_ok=True)
        f = partial(run_once_random, env_builder, env_param, experience_name_random)
        # with Pool(nb_runs) as pool:
        #     pool.map(f, range(nb_runs))
        list_of_rewards_random = []
        for np_name in glob.glob(str(experience_name_random) + '/*.np[yz]'):
            list_of_rewards_random.append(np.load(np_name, allow_pickle=True))
        random_percentage.append(np.mean(list_of_rewards_random))
        random_min_revenues.append(np.min(list_of_rewards_random))
        random_max_revenues.append(np.max(list_of_rewards_random))
        print("Random percentage of true revenue = {}".format((np.mean(list_of_rewards_random) / true_revenues) * 100))

        experience_name_DQL = Path("../Results/DQL_capacity_" + str(env_param["capacity"]))
        experience_name_DQL.mkdir(parents=True, exist_ok=True)
        param_dict_DQL = agent_parameters_dict_DQL()
        f = partial(run_once, env_builder, env_param, param_dict_DQL, nb_timesteps, experience_name_DQL,
                    callback_frequency)
        # with Pool(nb_runs) as pool:
        #     pool.map(f, range(nb_runs))
        # for k in range(nb_runs):
        #     run_once(env_builder, env_param, param_dict_DQL, nb_timesteps, experience_name_DQL, callback_frequency, k)
        list_of_rewards_DQL, mean_revenues_DQL, mean_bookings_DQL, min_revenues_DQL, max_revenues_DQL = env.collect_revenues(
            experience_name_DQL)
        average_DQL_revenue = mean_revenues_DQL[-1]
        DQL_percentage.append(average_DQL_revenue)
        DQL_min_revenues.append(min_revenues_DQL[-1])
        DQL_max_revenues.append(max_revenues_DQL[-1])
        print("DQL percentage of true revenue = {}".format((average_DQL_revenue / true_revenues) * 100))

        experience_name_QL = Path("../Results/QL_capacity_" + str(env_param["capacity"]))
        experience_name_QL.mkdir(parents=True, exist_ok=True)
        param_dict_QL = agent_parameters_dict_QL(nb_timesteps)
        f = partial(run_once_QL, env_builder, env_param, param_dict_QL, nb_timesteps, experience_name_QL,
                    callback_frequency)
        # with Pool(nb_runs) as pool:
        #     pool.map(f, range(nb_runs))
        # for k in range(nb_runs):
        #     run_once_QL(env_builder, env_param, param_dict_QL, nb_timesteps, experience_name_QL, callback_frequency, k)
        list_of_rewards_QL, mean_revenues_QL, mean_bookings_QL, min_revenues_QL, max_revenues_QL = env.collect_revenues(
            experience_name_QL)
        average_QL_revenue = mean_revenues_QL[-1]
        QL_percentage.append(average_QL_revenue)
        QL_min_revenues.append(min_revenues_QL[-1])
        QL_max_revenues.append(max_revenues_QL[-1])
        print("QL percentage of true revenue = {}".format((average_QL_revenue / true_revenues) * 100))

    plt.figure()
    total_capacities = [5] + capacities
    plt.plot(total_capacities, (QL_percentage / optimal_revenues) * 100, label="QL", color="c")
    plt.plot(total_capacities, (DQL_percentage / optimal_revenues) * 100, label="DQL", color="y")
    plt.plot(total_capacities, (random_percentage / optimal_revenues) * 100, label="Random policy", color="r")
    plt.xlabel("Capacity")
    plt.ylabel("Percentage of performance \n on smallest capacity")
    plt.fill_between(total_capacities, (QL_min_revenues / optimal_revenues) * 100,
                     (QL_max_revenues / optimal_revenues) * 100, color="c", alpha=0.2)
    plt.fill_between(total_capacities, (DQL_min_revenues / optimal_revenues) * 100,
                     (DQL_max_revenues / optimal_revenues) * 100, color="y", alpha=0.2)
    plt.fill_between(total_capacities, (random_min_revenues / optimal_revenues) * 100,
                     (random_max_revenues / optimal_revenues) * 100, color="r", alpha=0.2)
    plt.legend()
    plt.savefig("../Results2/" + "precision_as_a_function_of_C_and_T.png")

    plt.figure()
    total_capacities = [5] + capacities
    plt.plot(total_capacities, (((QL_percentage / optimal_revenues) * 100) / initial_QL_percentage)*100, label="QL",
             color="c")
    plt.plot(total_capacities, (((DQL_percentage / optimal_revenues) * 100) / initial_DQL_percentage)*100, label="DQL",
             color="y")
    plt.plot(total_capacities, (((random_percentage / optimal_revenues) * 100) / initial_random_percentage)*100,
             label="Random policy", color="r")
    plt.xlabel("Capacity")
    plt.ylabel("Percentage of performance \n on smallest capacity")
    plt.fill_between(total_capacities, (((QL_min_revenues / optimal_revenues) * 100) / initial_QL_percentage)*100,
                     (((QL_max_revenues / optimal_revenues) * 100) / initial_QL_percentage)*100, color="c", alpha=0.2)
    plt.fill_between(total_capacities, (((DQL_min_revenues / optimal_revenues) * 100) / initial_DQL_percentage)*100,
                     (((DQL_max_revenues / optimal_revenues) * 100) / initial_DQL_percentage)*100, color="y", alpha=0.2)
    plt.fill_between(total_capacities, (((random_min_revenues / optimal_revenues) * 100) / initial_random_percentage)*100,
                     (((random_max_revenues / optimal_revenues) * 100) / initial_random_percentage)*100, color="r", alpha=0.2)
    plt.legend()
    plt.savefig("../Results2/" + "scaling_as_a_function_of_C_and_T.png")

    env_param = env_parameters()
    env_param["action_offset"] = 100
    env = env_builder(env_param)
    initial_true_V, initial_true_P = dynamic_programming_env_DCP(env)
    initial_true_revenues, initial_true_bookings = average_n_episodes(env, initial_true_P, 10000)

    experience_name_random = Path("../Results3/Random_action_offset_" + str(env_param["action_offset"]))
    experience_name_random.mkdir(parents=True, exist_ok=True)
    f = partial(run_once_random, env_builder, env_param, experience_name_random)
    with Pool(nb_runs) as pool:
        pool.map(f, range(nb_runs))
    list_of_rewards_random = []
    for np_name in glob.glob(str(experience_name_random) + '/*.np[yz]'):
        list_of_rewards_random.append(np.load(np_name, allow_pickle=True))
    average_initial_random_revenue = np.mean(list_of_rewards_random)
    initial_random_percentage = (average_initial_random_revenue / initial_true_revenues) * 100
    print("Random percentage of true revenue = {}".format(initial_random_percentage))

    nb_timesteps = 80001
    absc = [k for k in range(0, nb_timesteps, nb_timesteps // callback_frequency)]

    experience_name_DQL = Path("../Results3/DQL_action_offset_" + str(env_param["action_offset"]))
    experience_name_DQL.mkdir(parents=True, exist_ok=True)
    param_dict_DQL = agent_parameters_dict_DQL()
    f = partial(run_once, env_builder, env_param, param_dict_DQL, nb_timesteps, experience_name_DQL, callback_frequency)
    with Pool(nb_runs) as pool:
        pool.map(f, range(nb_runs))
    # for k in range(nb_runs):
    #     run_once(env_builder, env_param, param_dict_DQL, nb_timesteps, experience_name_DQL, callback_frequency, k)
    list_of_rewards_DQL, mean_revenues_DQL, mean_bookings_DQL, min_revenues_DQL, max_revenues_DQL = env.collect_revenues(
        experience_name_DQL)
    average_initial_DQL_revenue = mean_revenues_DQL[-1]
    initial_DQL_percentage = (average_initial_DQL_revenue / initial_true_revenues) * 100
    print("DQL percentage of true revenue = {}".format(initial_DQL_percentage))

    experience_name_QL = Path("../Results3/QL_action_offset_" + str(env_param["action_offset"]))
    experience_name_QL.mkdir(parents=True, exist_ok=True)
    param_dict_QL = agent_parameters_dict_QL(nb_timesteps)
    f = partial(run_once_QL, env_builder, env_param, param_dict_QL, nb_timesteps, experience_name_QL,
                callback_frequency)
    with Pool(nb_runs) as pool:
        pool.map(f, range(nb_runs))
    # for k in range(nb_runs):
    #     run_once_QL(env_builder, env_param, param_dict_QL, nb_timesteps, experience_name_QL, callback_frequency, k)
    list_of_rewards_QL, mean_revenues_QL, mean_bookings_QL, min_revenues_QL, max_revenues_QL = env.collect_revenues(
        experience_name_QL)
    average_initial_QL_revenue = mean_revenues_QL[-1]
    initial_QL_percentage = (average_initial_QL_revenue / initial_true_revenues) * 100
    print("QL percentage of true revenue = {}".format(initial_QL_percentage))

    capacities = [k for k in range(10, 151, 10)]
    action_offsets = [70, 50, 40, 30, 20, 15, 12, 10, 9, 7]
    number_of_actions = [(env_param["action_max"] - env_param["action_min"]) // k + 1 for k in action_offsets]
    DQL_percentage = [100]
    DQL_min_revenues = [(min_revenues_DQL[-1] / average_initial_DQL_revenue) * 100]
    DQL_max_revenues = [(max_revenues_DQL[-1] / average_initial_DQL_revenue) * 100]
    QL_percentage = [100]
    QL_min_revenues = [(min_revenues_QL[-1] / average_initial_QL_revenue) * 100]
    QL_max_revenues = [(max_revenues_QL[-1] / average_initial_QL_revenue) * 100]
    random_percentage = [100]
    random_min_revenues = [(np.min(list_of_rewards_random) / average_initial_random_revenue) * 100]
    random_max_revenues = [(np.max(list_of_rewards_random) / average_initial_random_revenue) * 100]

    for action_offset in action_offsets:
        print("action offset :{}".format(action_offset))
        env_param = env_parameters()
        env_param["action_offset"] = action_offset
        env = env_builder(env_param)
        true_V, true_P = dynamic_programming_env_DCP(env)
        true_revenues, true_bookings = average_n_episodes(env, true_P, 10000)

        experience_name_random = Path("../Results3/Random_action_offset_" + str(env_param["action_offset"]))
        experience_name_random.mkdir(parents=True, exist_ok=True)
        f = partial(run_once_random, env_builder, env_param, experience_name_random)
        with Pool(nb_runs) as pool:
            pool.map(f, range(nb_runs))
        list_of_rewards_random = []
        for np_name in glob.glob(str(experience_name_random) + '/*.np[yz]'):
            list_of_rewards_random.append(np.load(np_name, allow_pickle=True))
        random_percentage.append(
            (((np.mean(list_of_rewards_random) / true_revenues) * 100) / initial_random_percentage) * 100)
        random_min_revenues.append(
            (((np.min(list_of_rewards_random) / true_revenues) * 100) / initial_random_percentage) * 100)
        random_max_revenues.append(
            (((np.max(list_of_rewards_random) / true_revenues) * 100) / initial_random_percentage) * 100)

        experience_name_DQL = Path("../Results3/DQL_action_offset_" + str(env_param["action_offset"]))
        experience_name_DQL.mkdir(parents=True, exist_ok=True)
        param_dict_DQL = agent_parameters_dict_DQL()
        f = partial(run_once, env_builder, env_param, param_dict_DQL, nb_timesteps, experience_name_DQL,
                    callback_frequency)
        with Pool(nb_runs) as pool:
            pool.map(f, range(nb_runs))
        # for k in range(nb_runs):
        #     run_once(env_builder, env_param, param_dict_DQL, nb_timesteps, experience_name_DQL, callback_frequency, k)
        list_of_rewards_DQL, mean_revenues_DQL, mean_bookings_DQL, min_revenues_DQL, max_revenues_DQL = env.collect_revenues(
            experience_name_DQL)
        average_DQL_revenue = mean_revenues_DQL[-1]
        DQL_percentage.append((((average_DQL_revenue / true_revenues) * 100) / initial_DQL_percentage) * 100)
        DQL_min_revenues.append((((min_revenues_DQL[-1] / true_revenues) * 100) / initial_DQL_percentage) * 100)
        DQL_max_revenues.append((((max_revenues_DQL[-1] / true_revenues) * 100) / initial_DQL_percentage) * 100)

        experience_name_QL = Path("../Results3/QL_action_offset_" + str(env_param["action_offset"]))
        experience_name_QL.mkdir(parents=True, exist_ok=True)
        param_dict_QL = agent_parameters_dict_QL(nb_timesteps)
        f = partial(run_once_QL, env_builder, env_param, param_dict_QL, nb_timesteps, experience_name_QL,
                    callback_frequency)
        with Pool(nb_runs) as pool:
            pool.map(f, range(nb_runs))
        # for k in range(nb_runs):
        #     run_once_QL(env_builder, env_param, param_dict_QL, nb_timesteps, experience_name_QL, callback_frequency, k)
        list_of_rewards_QL, mean_revenues_QL, mean_bookings_QL, min_revenues_QL, max_revenues_QL = env.collect_revenues(
            experience_name_QL)
        average_QL_revenue = mean_revenues_QL[-1]
        QL_percentage.append((((average_QL_revenue / true_revenues) * 100) / initial_QL_percentage) * 100)
        QL_min_revenues.append((((min_revenues_QL[-1] / true_revenues) * 100) / initial_QL_percentage) * 100)
        QL_max_revenues.append((((max_revenues_QL[-1] / true_revenues) * 100) / initial_QL_percentage) * 100)
    plt.figure()
    total_number_of_actions = [2] + number_of_actions
    plt.plot(total_number_of_actions, QL_percentage, label="QL", color="c")
    plt.plot(total_number_of_actions, DQL_percentage, label="DQL", color="y")
    plt.plot(total_number_of_actions, random_percentage, label="Random policy", color="r")
    plt.xlabel("Number of actions")
    plt.ylabel("Percentage of performance \n on smallest number of actions")
    plt.fill_between(total_number_of_actions, QL_min_revenues, QL_max_revenues, color="c", alpha=0.2)
    plt.fill_between(total_number_of_actions, DQL_min_revenues, DQL_max_revenues, color="y", alpha=0.2)
    plt.fill_between(total_number_of_actions, random_min_revenues, random_max_revenues, color="r", alpha=0.2)
    plt.savefig("../Results3/" + "scaling_as_a_function_of_actions_nb.png")

    # # DQL
    # nb_timesteps_DQL = 80001
    # absc = [k for k in range(0, nb_timesteps_DQL, nb_timesteps_DQL // callback_frequency)]
    # experience_name_DQL = Path("../Results/DQL")
    # experience_name_DQL.mkdir(parents=True, exist_ok=True)
    # param_dict_DQL = agent_parameters_dict_DQL()
    # # for k in range(nb_runs):
    # #     run_once(env_builder, param_dict, nb_timesteps, experience_name, callback_frequency, k)
    #
    # list_of_rewards_DQL, mean_revenues_DQL, mean_bookings_DQL, min_revenues_DQL, max_revenues_DQL = env.collect_revenues(
    #     experience_name_DQL)
    #
    # # QL
    # nb_timesteps_QL = 400001
    # experience_name_QL = Path("../Results/QL")
    # experience_name_QL.mkdir(parents=True, exist_ok=True)
    # param_dict_QL = agent_parameters_dict_QL()
    # for k in range(nb_runs):
    #     run_once_QL(env_builder, param_dict_QL, nb_timesteps_QL, experience_name_QL, callback_frequency, k)
    #
    # list_of_rewards_QL, mean_revenues_QL, mean_bookings_QL, min_revenues_QL, max_revenues_QL = env.collect_revenues(
    #     experience_name_QL)
    # # visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    # # visualizing_epsilon_decay(nb_episodes, alpha, alpha_min, alpha_decay)
    #
    # # Visualizing revenues
    # plt.figure()
    # width = 1 / 4
    # plt.bar(["Optimal \n solution"], true_revenues, width=width, label="{}".format(round(true_revenues)))
    # plt.bar(["QL"], np.mean(mean_revenues_QL[-1]), width=width, label="{}".format(round(mean_revenues_QL[-1])))
    # plt.bar(["DQL"], np.mean(mean_revenues_DQL[-1]), width=width, label="{}".format(round(mean_revenues_DQL[-1])))
    # plt.xlabel("Strategies")
    # plt.ylabel("Average revenue \n (computed on 10000 flights)")
    # plt.title("Revenues produced by the optimal policies \n of the different strategies")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1))
    # plt.show()
    #
    # # Visualizig bookings
    # plt.figure()
    # width = 5
    # plt.bar(np.array(env.A) - width, true_bookings, width=width,
    #         label="Optimal solution - Load factor of {:.2}".format(np.sum(true_bookings) / (env.C - 1)))
    # plt.bar(np.array(env.A), mean_bookings_QL[-1], width=width,
    #         label="QL - Load factor of {:.2}".format(np.sum(mean_bookings_QL[-1]) / (env.C - 1)))
    # plt.bar(np.array(env.A) + width, mean_bookings_DQL[-1], width=width,
    #         label="DQL - Load factor of {:.2}".format(np.sum(mean_bookings_DQL[-1]) / (env.C - 1)))
    # plt.xlabel("Prices")
    # plt.legend(loc='upper center', bbox_to_anchor=(0.7, 1))
    # plt.ylabel("Average number of bookings \n (computed on 10000 flights)")
    # plt.title("Number of bookings produced by the policies of the different strategies \n Demand ratio of {}".format(
    #     env.lamb * (env.T - 1) / (env.C - 1)))
    # plt.xticks(env.A)
    # plt.show()
