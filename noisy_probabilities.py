import numpy as np
import matplotlib.pyplot as plt
import gym
from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from dynamic_programming_env import dynamic_programming_collaboration
from visualization_and_metrics import average_n_episodes


def multiagent_env_parameters_dict():
    parameters_dict = {}
    parameters_dict["micro_times"] = 100
    parameters_dict["capacity1"] = 20
    parameters_dict["capacity2"] = 20
    parameters_dict["action_min"] = 50
    parameters_dict["action_max"] = 231
    parameters_dict["action_offset"] = 30
    parameters_dict["demand_ratio"] = 0.65
    parameters_dict["beta"] = 0.04
    parameters_dict["k_airline1"] = 5.
    parameters_dict["k_airline2"] = 5.
    parameters_dict["nested_lamb"] = 0.3
    return parameters_dict


def singleagent_env_parameters_dict():
    parameters_dict = {}
    parameters_dict["data_collection_points"] = 100
    parameters_dict["micro_times"] = 1
    parameters_dict["capacity"] = 20
    parameters_dict["action_min"] = 50
    parameters_dict["action_max"] = 231
    parameters_dict["action_offset"] = 30
    parameters_dict["lambda"] = 0.8
    parameters_dict["alpha"] = 0.7
    parameters_dict["noise_average"] = 0
    parameters_dict["noise_std_deviation"] = 0
    parameters_dict["compute_P_matrix"] = True
    parameters_dict["noise_percentage"] = 0
    return parameters_dict


def multiagent_env_builder(parameters_dict):
    prices_flight1 = [k for k in range(parameters_dict["action_min"], parameters_dict["action_max"] + 1,
                                       parameters_dict["action_offset"])]
    prices_flight2 = [k for k in range(parameters_dict["action_min"], parameters_dict["action_max"] + 1,
                                       parameters_dict["action_offset"])]
    lamb = parameters_dict["demand_ratio"] * (parameters_dict["capacity1"] + parameters_dict["capacity2"]) / \
           parameters_dict["micro_times"]

    return gym.make('gym_CollaborationGlobal3DMultiDiscrete:CollaborationGlobal3DMultiDiscrete-v0',
                    micro_times=parameters_dict["micro_times"],
                    capacity1=parameters_dict["capacity1"],
                    capacity2=parameters_dict["capacity2"],
                    prices=[prices_flight1, prices_flight2], beta=parameters_dict["beta"],
                    k_airline1=parameters_dict["k_airline1"],
                    k_airline2=parameters_dict["k_airline2"],
                    lamb=lamb,
                    nested_lamb=parameters_dict["nested_lamb"])


def singleagent_env_builder(parameters_dict):
    actions = tuple(k for k in range(parameters_dict["action_min"], parameters_dict["action_max"],
                                     parameters_dict["action_offset"]))
    return gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=parameters_dict["data_collection_points"],
                    capacity=parameters_dict["capacity"],
                    micro_times=parameters_dict["micro_times"], actions=actions, alpha=parameters_dict["alpha"],
                    lamb=parameters_dict["lambda"], noise_average=parameters_dict["noise_average"],
                    noise_std_deviation=parameters_dict["noise_std_deviation"],
                    compute_P_matrix=parameters_dict["compute_P_matrix"],
                    noise_percentage=parameters_dict["noise_percentage"])


if __name__ == '__main__':
    env_param = singleagent_env_parameters_dict()
    env_param["noise_percentage"] = 0
    real_env = singleagent_env_builder(env_param)

    if real_env.observation_space.shape[0] == 2:
        true_V, true_P = dynamic_programming_env_DCP(real_env)
        true_revenues, true_bookings = average_n_episodes(real_env, true_P, 10000)
    else:
        true_V, true_P = dynamic_programming_collaboration(real_env)
        true_revenue1, true_revenue2, true_bookings, true_bookings_flight1, true_bookings_flight2, true_prices_proposed_flight1, true_prices_proposed_flight2 = real_env.average_n_episodes(
            true_P, 10000)

    noise_percentages = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    average_error_percentage_on_parameters = (noise_percentages / 2) * 100
    percents_of_optimal_revenue = []
    average_nb = 10
    for noise_percentage in noise_percentages:
        print(noise_percentage)
        temporary_percents = []
        for k in range(average_nb):
            env_param["noise_percentage"] = noise_percentage
            env = singleagent_env_builder(env_param)

            V, P = dynamic_programming_env_DCP(env)
            revenues, bookings = average_n_episodes(real_env, P, 10000)
            temporary_percents.append((revenues / true_revenues) * 100)
        percents_of_optimal_revenue.append(np.mean(temporary_percents))

    plt.figure()
    plt.plot(average_error_percentage_on_parameters, percents_of_optimal_revenue)
    plt.xlabel("Average percentage error on the transition probabilities")
    plt.ylabel("Percentage of the optimal revenue")
    plt.show()

