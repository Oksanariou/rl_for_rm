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
    parameters_dict["compute_P_matrix"] = True
    parameters_dict["transition_noise_percentage"] = 0
    parameters_dict["parameter_noise_percentage"] = 0
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
                    lamb=parameters_dict["lambda"], compute_P_matrix=parameters_dict["compute_P_matrix"],
                    transition_noise_percentage=parameters_dict["transition_noise_percentage"],
                    parameter_noise_percentage=parameters_dict["parameter_noise_percentage"])


if __name__ == '__main__':
    env_param = singleagent_env_parameters_dict()
    real_env = singleagent_env_builder(env_param)

    if real_env.observation_space.shape[0] == 2:
        true_V, true_P = dynamic_programming_env_DCP(real_env)
        true_revenues, true_bookings = average_n_episodes(real_env, true_P, 10000)
    else:
        true_V, true_P = dynamic_programming_collaboration(real_env)
        true_revenue1, true_revenue2, true_bookings, true_bookings_flight1, true_bookings_flight2, true_prices_proposed_flight1, true_prices_proposed_flight2 = real_env.average_n_episodes(
            true_P, 10000)

    noise_percentages = np.array([k for k in range(0, 60, 5)]) / 100
    mean_percents_of_optimal_revenue = []
    max_percents_of_optimal_revenue = []
    min_percents_of_optimal_revenue = []
    mean_error = []
    average_nb = 20
    for noise_percentage in noise_percentages:
        print(noise_percentage)
        temporary_percents = []
        temporary_error = []
        for k in range(average_nb):
            env_param = singleagent_env_parameters_dict()
            # env_param["transition_noise_percentage"] = noise_percentage
            env_param["parameter_noise_percentage"] = noise_percentage
            env = singleagent_env_builder(env_param)

            V, P = dynamic_programming_env_DCP(env)
            revenues, bookings = average_n_episodes(real_env, P, 10000)
            temporary_percents.append((revenues / true_revenues) * 100)
            # error = np.mean(abs(np.array(real_env.probas) - np.array(env.probas)) / np.array(real_env.probas))
            error = np.mean([abs(real_env.lamb - env.lamb)/real_env.lamb, abs(real_env.alpha - env.alpha)/real_env.alpha])
            temporary_error.append(error)
        mean_percents_of_optimal_revenue.append(np.mean(temporary_percents))
        max_percents_of_optimal_revenue.append(np.max(temporary_percents))
        min_percents_of_optimal_revenue.append(np.min(temporary_percents))
        mean_error.append(np.mean(temporary_error))

    plt.figure()
    plt.plot(noise_percentages, mean_percents_of_optimal_revenue)
    plt.fill_between(noise_percentages, min_percents_of_optimal_revenue, max_percents_of_optimal_revenue,
                     label='95% confidence interval', alpha=0.2)
    plt.xlabel("Average percentage error on the parameters")
    # plt.xlabel("Noise")
    plt.ylabel("Percentage of the optimal revenue")
    plt.show()
