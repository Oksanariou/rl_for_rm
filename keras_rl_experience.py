import numpy as np
import gym
import keras
import random
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob

from dynamic_programming_env import dynamic_programming_collaboration
from q_learning import q_to_v

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent

from dynamic_programming_env_DCP import dynamic_programming_env_DCP

from visualization_and_metrics import average_n_episodes, q_to_policy_RM
from ACKTR_experience import plot_revenues
from functools import partial
from multiprocessing import Pool
from Collaboration_Competition.keras_rl_multi_agent import fit_multi_agent, combine_two_policies, observation_split_2D, \
    action_merge, observation_split_3D
from scipy.stats import sem, t


def global_env_builder(parameters_dict):
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
                    nested_lamb=parameters_dict["nested_lamb"],
                    parameter_noise_percentage=parameters_dict["parameter_noise_percentage"])


def env_builder():
    # Parameters of the environment
    data_collection_points = 101
    micro_times = 1
    capacity = 51

    action_min = 50
    action_max = 231
    action_offset = 20

    actions = tuple(k for k in range(action_min, action_max, action_offset))
    alpha = 0.65
    lamb = 0.65

    return gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)


class callback(keras.callbacks.Callback):
    def __init__(self, env, nb_timesteps, period):
        super(callback, self).__init__()
        self.env = env
        self.nb_timesteps = nb_timesteps
        self.period = period

    def on_train_begin(self, logs={}):
        self.rewards = []

    def on_batch_end(self, batch, logs={}):
        if ((self.model.step % (self.nb_timesteps // self.period)) == 0):
            Q_table = [self.model.compute_q_values([state]) for state in self.env.states]
            policy = [np.argmax(q) for q in Q_table]
            policy = np.asarray(policy).reshape(self.env.observation_space.nvec)
            self.rewards.append(self.env.average_n_episodes(policy, 1000))


class callback_multiagent(keras.callbacks.Callback):
    def __init__(self, env, nb_timesteps, period, configuration_name):
        super(callback_multiagent, self).__init__()
        self.env = env
        self.nb_timesteps = nb_timesteps
        self.period = period
        self.configuration_name = configuration_name

    def on_train_begin(self, logs={}):
        self.rewards = []

    def on_batch_end(self, batch, logs={}):
        if ((self.model[0].step % (self.nb_timesteps // self.period)) == 0):
            print(self.model[0].step)
            if parameters[self.configuration_name]["shape"][0] == 3:
                states1, states2 = self.env.states, self.env.states
                dim1, dim2 = (self.env.T, self.env.C1, self.env.C2), (self.env.T, self.env.C1, self.env.C2)
            else:
                states1, states2 = self.env.states1, self.env.states2
                dim1, dim2 = (self.env.T, self.env.C1), (self.env.T, self.env.C2)
            Q_table1 = [self.model[0].compute_q_values([state]) for state in states1]
            policy1 = [np.argmax(q) for q in Q_table1]
            policy1 = np.asarray(policy1).reshape(dim1)
            Q_table2 = [self.model[1].compute_q_values([state]) for state in states2]
            policy2 = [np.argmax(q) for q in Q_table2]
            policy2 = np.asarray(policy2).reshape(dim2)
            combined_policy = combine_two_policies(self.env, policy1, policy2,
                                                   parameters[self.configuration_name]["observation_split"],
                                                   parameters[self.configuration_name]["action_merge"])
            self.rewards.append(self.env.average_n_episodes(combined_policy, 1000))


def build_model(nb_actions, shape, hidden_layer_size, layers_nb):
    model = Sequential()
    model.add(Flatten(input_shape=((1,) + shape)))
    for k in range(layers_nb):
        model.add(Dense(hidden_layer_size))
        model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model


def agent_parameters_dict():
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
    parameters_dict["parameter_noise_percentage"] = 0
    return parameters_dict


def run_once(env_builder, env_parameters_dict, parameters_dict, nb_timesteps, experience_name, period, k):
    env = env_builder(env_parameters_dict)
    model = build_model(env.nA, env.observation_space.shape, parameters_dict["hidden_layer_size"],
                        parameters_dict["layers_nb"])
    memory = SequentialMemory(limit=parameters_dict["memory_buffer_size"], window_length=1)
    policy = EpsGreedyQPolicy(eps=parameters_dict["epsilon"])
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory,
                   nb_steps_warmup=parameters_dict["nb_steps_warmup"],
                   enable_double_dqn=parameters_dict["enable_double_dqn"],
                   enable_dueling_network=parameters_dict["enable_dueling_network"],
                   target_model_update=parameters_dict["target_model_update"], policy=policy)
    dqn.compile(Adam(lr=parameters_dict["learning_rate"]), metrics=['mae'])
    rewards = callback(env, nb_timesteps, period)
    history = dqn.fit(env, nb_steps=nb_timesteps, visualize=False, verbose=2, callbacks=[rewards])
    np.save(experience_name / ("Run" + str(k) + ".npy"), rewards.rewards)


def run_once_multiagent(env_parameters, agent_parameter_dict, configuration_name, nb_timesteps, experience_name,
                        callback_frequency, k):
    env = global_env_builder(env_parameters)
    model1 = build_model(len(env.prices[0]), parameters[configuration_name]["shape"],
                         agent_parameter_dict["hidden_layer_size"],
                         agent_parameter_dict["layers_nb"])
    memory1 = SequentialMemory(limit=agent_parameter_dict["memory_buffer_size"], window_length=1)
    policy1 = EpsGreedyQPolicy(eps=agent_parameter_dict["epsilon"])
    dqn1 = DQNAgent(model=model1, nb_actions=len(env.prices[0]), memory=memory1,
                    nb_steps_warmup=agent_parameter_dict["nb_steps_warmup"],
                    enable_double_dqn=agent_parameter_dict["enable_double_dqn"],
                    enable_dueling_network=agent_parameter_dict["enable_dueling_network"],
                    target_model_update=agent_parameter_dict["target_model_update"], policy=policy1)
    dqn1.compile(Adam(lr=agent_parameter_dict["learning_rate"]), metrics=['mae'])

    memory2 = SequentialMemory(limit=agent_parameter_dict["memory_buffer_size"], window_length=1)
    policy2 = EpsGreedyQPolicy(eps=agent_parameter_dict["epsilon"])
    model2 = build_model(len(env.prices[0]), parameters[configuration_name]["shape"],
                         agent_parameter_dict["hidden_layer_size"],
                         agent_parameter_dict["layers_nb"])
    dqn2 = DQNAgent(model=model2, nb_actions=len(env.prices[1]), memory=memory2,
                    nb_steps_warmup=agent_parameter_dict["nb_steps_warmup"],
                    enable_double_dqn=agent_parameter_dict["enable_double_dqn"],
                    enable_dueling_network=agent_parameter_dict["enable_dueling_network"],
                    target_model_update=agent_parameter_dict["target_model_update"], policy=policy2)
    dqn2.compile(Adam(lr=agent_parameter_dict["learning_rate"]), metrics=['mae'])

    callback = callback_multiagent(env, nb_timesteps, callback_frequency, configuration_name)
    history = fit_multi_agent(agents=[dqn1, dqn2], global_env=env, nb_steps=nb_timesteps,
                              callbacks=[callback],
                              fully_collaborative=parameters[configuration_name]["fully_collaborative"],
                              observation_split=parameters[configuration_name]["observation_split"],
                              action_merge=parameters[configuration_name]["action_merge"])
    np.save(experience_name / ("Run" + str(k) + ".npy"), callback.rewards)


def run_n_times(env_parameters_dict, experience_name, env_builder, parameters_dict, nb_timesteps, number_of_runs, period):
    (experience_name).mkdir(parents=True, exist_ok=True)

    f = partial(run_once, env_builder, env_parameters_dict, parameters_dict, nb_timesteps, experience_name, period)

    with Pool(number_of_runs) as pool:
        pool.map(f, range(number_of_runs))


def plot_comparison(experience_name, parameters, env, absc, optimal_revenue):
    comparison_mean_revenues = []
    for parameter in parameters:
        parameter_name = experience_name / Path(str(parameter))
        list_of_rewards, mean_revenues = env.collect_revenues(parameter_name)
        comparison_mean_revenues.append(mean_revenues)
    fig = plt.figure()
    for k in range(len(parameters)):
        plt.plot(absc, comparison_mean_revenues[k], label=str(parameters[k]))
    plt.legend()
    plt.title(experience_name.name)
    plt.plot(absc, [optimal_revenue] * len(absc), label="Optimal solution")
    plt.ylabel("Average revenue on 10000 flights")
    plt.xlabel("Number of episodes")
    plt.savefig(str(experience_name) + "/" + experience_name.name + '.png')


def multi_agent_experience(demand_ratios, configuration_name, nb_timesteps, callback_frequency, number_of_runs):
    for dr in demand_ratios:
        env_param = multiagent_env_parameters_dict()
        env_param["demand_ratio"] = dr

        experience_name = Path("../Results/" + configuration_name + "/" + str(dr))
        (experience_name).mkdir(parents=True, exist_ok=True)

        agent_param = agent_parameters_dict()

        # for k in range(number_of_runs):
        #     run_once_multiagent(env_param, agent_param, configuration_name, nb_timesteps, experience_name, callback_frequency,k)

        f = partial(run_once_multiagent, env_param, agent_param, configuration_name, nb_timesteps, experience_name,
                    callback_frequency)

        with Pool(number_of_runs) as pool:
            pool.map(f, range(number_of_runs))


def parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues,
                         absc, nb_runs, period):
    for parameter_value in parameter_values:
        param_dict = agent_parameters_dict()
        param_dict[parameter_name] = parameter_value
        parameter_value_name = experience_name / Path(str(parameter_value))
        parameter_value_name.mkdir(parents=True, exist_ok=True)
        # for k in range(nb_runs):
        #     run_once(env_builder, param_dict, nb_timesteps, parameter_value_name, period, k)
        run_n_times(parameter_value_name, env_builder, param_dict, nb_timesteps, nb_runs, period)

        list_of_rewards, mean_revenues = env_builder().collect_revenues(parameter_value_name)
        env_builder().plot_collected_data(mean_revenues, list_of_rewards, absc, true_revenues)
        plt.title(parameter_name + " = " + str(parameter_value))
        plt.savefig(str(parameter_value_name) + "/" + parameter_name + " = " + parameter_value_name.name + '.png')


def run_once_random(env_builder, env_parameters_dict, experience_name, real_env, k):
    env = env_builder(env_parameters_dict)
    V, P = dynamic_programming_collaboration(env)
    revenue = real_env.average_n_episodes(P, 10000)
    np.save(experience_name / ("Run" + str(k) + ".npy"), revenue[0] + revenue[1])


if __name__ == '__main__':
    nb_runs = 20
    # mp.set_start_method('spawn')
    env_parameters_dict = multiagent_env_parameters_dict()
    env = global_env_builder(env_parameters_dict)
    # env = env_builder()

    if env.observation_space.shape[0] == 2:
        true_V, true_P = dynamic_programming_env_DCP(env)
        true_revenues, true_bookings = average_n_episodes(env, true_P, 10000)
    else:
        true_V, true_P = dynamic_programming_collaboration(env)
        true_revenue1, true_revenue2, true_bookings, true_bookings_flight1, true_bookings_flight2, true_prices_proposed_flight1, true_prices_proposed_flight2 = env.average_n_episodes(
            true_P, 10000)

    # nb_timesteps = 80001
    # callback_frequency = 10
    # absc = [k for k in range(0, nb_timesteps, nb_timesteps // callback_frequency)]
    # nb_runs = 20
    # experience_name = Path("../Results/Best_run")
    # experience_name.mkdir(parents=True, exist_ok=True)
    # param_dict = agent_parameters_dict()
    # for k in range(nb_runs):
    #     run_once(env_builder, param_dict, nb_timesteps, experience_name, callback_frequency, k)
    #
    # list_of_rewards, mean_revenues, mean_bookings, min_revenues, max_revenues = env.collect_revenues(experience_name)
    # fig = plt.figure()
    # plt.plot(absc, mean_revenues, label="DQN revenue")
    # plt.legend()
    # plt.title(experience_name.name)
    # plt.plot(absc, [true_revenues] * len(absc), label="Optimal solution")
    # plt.fill_between(absc, min_revenues, max_revenues, label='95% confidence interval', alpha=0.2)
    # plt.ylabel("Average revenue on 10000 flights")
    # plt.xlabel("Number of steps")
    # plt.savefig(str(experience_name) + "/" + experience_name.name + '.png')

    parameters = {}
    parameters["2D_individual_rewards"] = {"shape": (2,), "observation_split": observation_split_2D,
                                           "fully_collaborative": False, "action_merge": action_merge, "color": "c"}
    parameters["2D_shared_rewards"] = {"shape": (2,), "observation_split": observation_split_2D,
                                       "fully_collaborative": True, "action_merge": action_merge, "color": "y"}
    parameters["3D_individual_rewards"] = {"shape": (3,), "observation_split": observation_split_3D,
                                           "fully_collaborative": False, "action_merge": action_merge, "color": "black"}
    parameters["3D_shared_rewards"] = {"shape": (3,), "observation_split": observation_split_3D,
                                       "fully_collaborative": True, "action_merge": action_merge, "color": "m"}

    # configuration = "2D_shared_rewards"
    configuration = "2D_individual_rewards"

    # model1 = build_model(len(env.prices[0]), parameters[configuration]["shape"], param_dict["hidden_layer_size"],
    #                      param_dict["layers_nb"])
    # memory1 = SequentialMemory(limit=param_dict["memory_buffer_size"], window_length=1)
    # policy1 = EpsGreedyQPolicy(eps=param_dict["epsilon"])
    # dqn1 = DQNAgent(model=model1, nb_actions=len(env.prices[0]), memory=memory1,
    #                 nb_steps_warmup=param_dict["nb_steps_warmup"],
    #                 enable_double_dqn=param_dict["enable_double_dqn"],
    #                 enable_dueling_network=param_dict["enable_dueling_network"],
    #                 target_model_update=param_dict["target_model_update"], policy=policy1)
    # dqn1.compile(Adam(lr=param_dict["learning_rate"]), metrics=['mae'])
    #
    # memory2 = SequentialMemory(limit=param_dict["memory_buffer_size"], window_length=1)
    # policy2 = EpsGreedyQPolicy(eps=param_dict["epsilon"])
    # model2 = build_model(len(env.prices[0]), parameters[configuration]["shape"], param_dict["hidden_layer_size"],
    #                      param_dict["layers_nb"])
    # dqn2 = DQNAgent(model=model2, nb_actions=len(env.prices[1]), memory=memory2,
    #                 nb_steps_warmup=param_dict["nb_steps_warmup"],
    #                 enable_double_dqn=param_dict["enable_double_dqn"],
    #                 enable_dueling_network=param_dict["enable_dueling_network"],
    #                 target_model_update=param_dict["target_model_update"], policy=policy2)
    # dqn2.compile(Adam(lr=param_dict["learning_rate"]), metrics=['mae'])
    #
    # callback = callback_multiagent(env, nb_timesteps, callback_frequency, configuration)
    # history = fit_multi_agent(agents=[dqn1, dqn2], global_env=env, nb_steps=nb_timesteps,
    #                           callbacks=[callback],
    #                           fully_collaborative=parameters[configuration]["fully_collaborative"],
    #                           observation_split=parameters[configuration]["observation_split"],
    #                           action_merge=parameters[configuration]["action_merge"])

    demand_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
    # demand_ratios = [0.5, 0.6]
    configuration_names = ["2D_individual_rewards", "2D_shared_rewards"]
    nb_timesteps = 100001
    callback_frequency = 10
    number_of_runs = 20
    # for configuration_name in configuration_names:
    #     multi_agent_experience(demand_ratios, configuration_name, nb_timesteps, callback_frequency, number_of_runs)

    # for configuration_name in configuration_names:
    #     for dr_idx in range(len(demand_ratios)):
    #         env_param = multiagent_env_parameters_dict()
    #         env_param["demand_ratio"] = demand_ratios[dr_idx]
    #         env = global_env_builder(env_param)
    #         true_V, true_P = dynamic_programming_collaboration(env)
    #         true_revenue1, true_revenue2, true_bookings, true_bookings_flight1, true_bookings_flight2, true_prices_proposed_flight1, true_prices_proposed_flight2 = env.average_n_episodes(
    #             true_P, 10000)
    #         plt.figure()
    #         plt.plot(absc, [true_revenue1 + true_revenue2] * len(absc), 'g--', label="Optimal solution")
    #         experience_name = Path("../Results/"+configuration_name+"/"+str(demand_ratios[dr_idx]))
    #         list_of_rewards, mean_revenues1, mean_revenues2, mean_bookings, mean_bookings1, mean_bookings2, mean_prices_proposed1, mean_prices_proposed2 = env.collect_list_of_mean_revenues_and_bookings(experience_name)
    #         list_of_rewards = np.array(list_of_rewards)
    #         for reward in list_of_rewards:
    #             plt.plot(absc, np.array(reward[:,0]) + np.array(reward[:,1]), alpha=0.2, color=parameters[configuration_name]["color"])
    #         plt.plot(absc, np.array(mean_revenues1) + np.array(mean_revenues2), color=parameters[configuration_name]["color"])
    #         plt.legend(loc='best')
    #         plt.xlabel("Number of steps")
    #         plt.ylabel("Average revenue on {} flights".format(10000))
    #         plt.savefig("../Results/"+configuration_name+"/"+str(demand_ratios[dr_idx])+"/"+str(demand_ratios[dr_idx])+"_revenues.png")
    #
    #         plt.figure()
    #         width = 5
    #         bookings1, bookings2 = mean_bookings1[-1], mean_bookings2[-1]
    #         prices_proposed1, prices_proposed2 = mean_prices_proposed1[-1], mean_prices_proposed2[-1]
    #         plt.bar(np.array(env.prices_flight2) + 2*width/3, bookings2, width, color="blue", label="Bookings flight 2")
    #         plt.bar(np.array(env.prices_flight1) + 2*width/3, bookings1, width, color="orange", label="Bookings flight 1", bottom=bookings2)
    #         plt.bar(np.array(env.prices_flight2) - 2*width/3, prices_proposed2, width, color="blue", alpha = 0.3, label="Prices proposed flight 2")
    #         plt.bar(np.array(env.prices_flight1) - 2*width/3, prices_proposed1, width, color="orange", alpha = 0.3, label="Prices proposed flight 1", bottom=prices_proposed2)
    #         plt.xlabel("Prices")
    #         plt.ylabel("Average computed on 10000 flights")
    #         plt.title("Overall load factor: {:.2}".format((np.sum(bookings2) + np.sum(bookings2)) / (env.C1 + env.C2)))
    #         plt.legend()
    #         plt.xticks(env.prices_flight1)
    #         plt.savefig("../Results/"+configuration_name+"/"+str(demand_ratios[dr_idx])+"/"+str(demand_ratios[dr_idx])+"_mean_bookings.png")
    nb_collection_points = len(demand_ratios)
    true_revenues = []
    list_final_revenues = []
    list_mean_final_revenues = []
    single_agent_min_revenues = []
    single_agent_max_revenues = []
    for dr_idx in range(len(demand_ratios)):
        print(dr_idx)
        env_param = multiagent_env_parameters_dict()
        env_param["demand_ratio"] = demand_ratios[dr_idx]
        env = global_env_builder(env_param)
        true_V, true_P = dynamic_programming_collaboration(env)
        true_revenue1, true_revenue2, true_bookings, true_bookings_flight1, true_bookings_flight2, true_prices_proposed_flight1, true_prices_proposed_flight2 = env.average_n_episodes(
            true_P, 10000)
        true_revenues.append(true_revenue1+true_revenue2)
        experience_name = Path("../Results/single_global_agent_" + str(dr_idx))
        experience_name.mkdir(parents=True, exist_ok=True)
        run_n_times(env_param, experience_name, global_env_builder, env_param, nb_timesteps, number_of_runs, callback_frequency)
        list_of_rewards, mean_revenues1, mean_revenues2, mean_bookings, mean_bookings1, mean_bookings2, mean_prices_proposed1, mean_prices_proposed2 = env.collect_list_of_mean_revenues_and_bookings(experience_name)
        list_of_rewards = np.array(list_of_rewards)
        for reward in list_of_rewards:
            list_final_revenues[dr_idx].append(
                ((reward[:, 0][-1] + reward[:, 1][-1]) / (true_revenue1 + true_revenue2)) * 100)

        difference_to_true_revenue = ((mean_revenues1[-1] + mean_revenues2[-1]) / (
                true_revenue1 + true_revenue2)) * 100
        list_mean_final_revenues.append(difference_to_true_revenue)

    std_revenues = [sem(list) for list in list_final_revenues]
    confidence_revenues = [std_revenues[k] * t.ppf((1 + 0.95) / 2, nb_collection_points - 1) for k in
                           range(nb_collection_points)]
    min_revenues = [list_mean_final_revenues[k] - confidence_revenues[k] for k in range(nb_collection_points)]
    max_revenues = [list_mean_final_revenues[k] + confidence_revenues[k] for k in range(nb_collection_points)]
    plt.figure()
    plt.plot(demand_ratios, list_mean_final_revenues, color="r",)
    plt.fill_between(demand_ratios, min_revenues, max_revenues, color="r", alpha=0.2)
    plt.legend(loc='best')
    plt.xlabel("Demand ratio")
    plt.ylabel("Percentage of optimal revenue \n average on {} flights".format(10000))
    plt.savefig('../Results/single_agent_multi_flights.png')
    # axes.set_ylim([0, 265])



    # plt.figure()
    # nb_collection_points = len(demand_ratios)
    # for configuration_name in configuration_names:
    #     list_mean_final_revenues = []
    #     if configuration_name == "2D_individual_rewards":
    #         list_difference_to_true_revenue_parameter_noise = []
    #         list_difference_to_true_revenue_parameter_noise_min = []
    #         list_difference_to_true_revenue_parameter_noise_max = []
    #         list_difference_to_true_revenue_parameter_noise_mnl = []
    #         list_difference_to_true_revenue_parameter_noise_min_mnl = []
    #         list_difference_to_true_revenue_parameter_noise_max_mnl = []
    #         list_final_revenues = [[] for k in range(len(demand_ratios))]
    #     for dr_idx in range(len(demand_ratios)):
    #         print(dr_idx)
    #         env_param = multiagent_env_parameters_dict()
    #         env_param["demand_ratio"] = demand_ratios[dr_idx]
    #         env = global_env_builder(env_param)
    #         true_V, true_P = dynamic_programming_collaboration(env)
    #         true_revenue1, true_revenue2, true_bookings, true_bookings_flight1, true_bookings_flight2, true_prices_proposed_flight1, true_prices_proposed_flight2 = env.average_n_episodes(
    #             true_P, 10000)
    #
    #         if configuration_name == "2D_individual_rewards":
    #             env_param["parameter_noise_percentage"] = 0.2
    #             differences_to_true_revenue_parameter_noise = []
    #
    #             experience_name_noise = Path("../Results/Noise_on_parameters_"+str(dr_idx))
    #             experience_name_noise.mkdir(parents=True, exist_ok=True)
    #             # for k in range(nb_runs):
    #             #     run_once_random(global_env_builder, env_param, experience_name_noise, env, k)
    #             for np_name in glob.glob(str(experience_name_noise) + '/*.np[yz]'):
    #                 differences_to_true_revenue_parameter_noise.append(
    #                     (np.load(np_name, allow_pickle=True) / (true_revenue1 + true_revenue2)) * 100)
    #
    #             env_param["nested_lamb"] = 1.
    #             experience_name_noise_mnl = Path("../Results/Noise_on_parameters_mnl_"+str(dr_idx))
    #             experience_name_noise_mnl.mkdir(parents=True, exist_ok=True)
    #             differences_to_true_revenue_parameter_noise_mnl = []
    #             # for k in range(nb_runs):
    #             #     run_once_random(global_env_builder, env_param, experience_name_noise_mnl, env, k)
    #             for np_name in glob.glob(str(experience_name_noise_mnl) + '/*.np[yz]'):
    #                 differences_to_true_revenue_parameter_noise_mnl.append(
    #                     (np.load(np_name, allow_pickle=True) / (true_revenue1 + true_revenue2)) * 100)
    #
    #             list_difference_to_true_revenue_parameter_noise_mnl.append(
    #                 np.mean(differences_to_true_revenue_parameter_noise_mnl))
    #             list_difference_to_true_revenue_parameter_noise_min_mnl.append(
    #                 np.min(differences_to_true_revenue_parameter_noise_mnl))
    #             list_difference_to_true_revenue_parameter_noise_max_mnl.append(
    #                 np.max(differences_to_true_revenue_parameter_noise_mnl))
    #             list_difference_to_true_revenue_parameter_noise.append(
    #                 np.mean(differences_to_true_revenue_parameter_noise))
    #             list_difference_to_true_revenue_parameter_noise_min.append(
    #                 np.min(differences_to_true_revenue_parameter_noise))
    #             list_difference_to_true_revenue_parameter_noise_max.append(
    #                 np.max(differences_to_true_revenue_parameter_noise))
    #
    #         experience_name = Path("../Results/" + configuration_name + "/" + str(demand_ratios[dr_idx]))
    #         experience_name.mkdir(parents=True, exist_ok=True)
    #         list_of_rewards, mean_revenues1, mean_revenues2, mean_bookings, mean_bookings1, mean_bookings2, mean_prices_proposed1, mean_prices_proposed2 = env.collect_list_of_mean_revenues_and_bookings(
    #             experience_name)
    #         difference_to_true_revenue = ((mean_revenues1[-1] + mean_revenues2[-1]) / (
    #                 true_revenue1 + true_revenue2)) * 100
    #
    #         list_mean_final_revenues.append(difference_to_true_revenue)
    #         list_of_rewards = np.array(list_of_rewards)
    #
    #         # plt.figure()
    #         # plt.plot(absc, [true_revenue1 + true_revenue2] * len(absc), 'g--', label="Optimal solution")
    #
    #         for reward in list_of_rewards:
    #             list_final_revenues[dr_idx].append(
    #                 ((reward[:, 0][-1] + reward[:, 1][-1]) / (true_revenue1 + true_revenue2)) * 100)
    #             # plt.plot(absc, np.array(reward[:, 0]) + np.array(reward[:, 1]), alpha=0.2,
    #             #          color=parameters[configuration_name]["color"])
    #
    #         # plt.plot(absc, np.array(mean_revenues1) + np.array(mean_revenues2),
    #         #          color=parameters[configuration_name]["color"])
    #         # plt.legend(loc='best')
    #         # plt.xlabel("Number of steps")
    #         # plt.ylabel("Average revenue on {} flights".format(10000))
    #         # plt.savefig("../Results/"+configuration_name+"/"+str(demand_ratios[dr_idx])+"/"+str(demand_ratios[dr_idx])+"_revenues.png")
    #         #
    #         # plt.figure()
    #         # width = 5
    #         # bookings1, bookings2 = mean_bookings1[-1], mean_bookings2[-1]
    #         # prices_proposed1, prices_proposed2 = mean_prices_proposed1[-1], mean_prices_proposed2[-1]
    #         # plt.bar(np.array(env.prices_flight2) + 2*width/3, bookings2, width, color="blue", label="Bookings flight 2")
    #         # plt.bar(np.array(env.prices_flight1) + 2*width/3, bookings1, width, color="orange", label="Bookings flight 1", bottom=bookings2)
    #         # plt.bar(np.array(env.prices_flight2) - 2*width/3, prices_proposed2, width, color="blue", alpha=0.3, label="Prices proposed flight 2")
    #         # plt.bar(np.array(env.prices_flight1) - 2*width/3, prices_proposed1, width, color="orange", alpha=0.3, label="Prices proposed flight 1", bottom=prices_proposed2)
    #         # plt.xlabel("Prices")
    #         # plt.ylabel("Average computed on 10000 flights")
    #         # plt.title("Overall load factor: {:.2}".format((np.sum(bookings2) + np.sum(bookings2)) / (env.C1 + env.C2)))
    #         # plt.legend()
    #         # plt.xticks(env.prices_flight1)
    #         # plt.savefig("../Results/"+configuration_name+"/"+str(demand_ratios[dr_idx])+"/"+str(demand_ratios[dr_idx])+"_mean_bookings.png")
    #
    #     std_revenues = [sem(list) for list in list_final_revenues]
    #     confidence_revenues = [std_revenues[k] * t.ppf((1 + 0.95) / 2, nb_collection_points - 1) for k in
    #                            range(nb_collection_points)]
    #     min_revenues = [list_mean_final_revenues[k] - confidence_revenues[k] for k in range(nb_collection_points)]
    #     max_revenues = [list_mean_final_revenues[k] + confidence_revenues[k] for k in range(nb_collection_points)]
    #     plt.plot(demand_ratios, list_mean_final_revenues, color=parameters[configuration_name]["color"],
    #              label=configuration_name)
    #     plt.fill_between(demand_ratios, min_revenues, max_revenues, color=parameters[configuration_name]["color"],
    #                      alpha=0.2)
    # plt.plot(demand_ratios, list_difference_to_true_revenue_parameter_noise, color="orange",
    #          label="Model-based with 20% \n noise in parameters")
    # plt.fill_between(demand_ratios, list_difference_to_true_revenue_parameter_noise_min,
    #                  list_difference_to_true_revenue_parameter_noise_max, color="orange", alpha=0.2)
    # plt.plot(demand_ratios, list_difference_to_true_revenue_parameter_noise_mnl, color="red",
    #          label="Model-based with 20% \n noise in parameters and \n MNL CCM assumption")
    # plt.fill_between(demand_ratios, list_difference_to_true_revenue_parameter_noise_min_mnl,
    #                  list_difference_to_true_revenue_parameter_noise_max_mnl, color="red", alpha=0.2)
    # plt.legend(loc='best')
    # plt.xlabel("Demand ratio")
    # plt.ylabel("Percentage of the optimal revenue \n average on {} flights".format(10000))
    # # axes.set_ylim([0, 265])
    # plt.savefig('../Results/multiagent_strategies_as_function_of_demand_ratios_variance_mnl.png')
    #
    # # plt.figure()
    # # for configuration_name in configuration_names:
    # #     list_mean_final_revenues = []
    # #     for dr in demand_ratios:
    # #         env_param = multiagent_env_parameters_dict()
    # #         env_param["demand_ratio"] = dr
    # #         env = global_env_builder(env_param)
    # #         true_V, true_P = dynamic_programming_collaboration(env)
    # #         true_revenue1, true_revenue2, true_bookings, true_bookings_flight1, true_bookings_flight2, true_prices_proposed_flight1, true_prices_proposed_flight2 = env.average_n_episodes(
    # #             true_P, 10000)
    # #         experience_name = Path("../Results/"+configuration_name+"/"+str(dr))
    # #         mean_revenues1, mean_revenues2, mean_bookings, mean_bookings1, mean_bookings2, mean_prices_proposed1, mean_prices_proposed2 = env.collect_list_of_mean_revenues_and_bookings(experience_name)
    # #         difference_to_true_revenue = ((mean_revenues1[-1] + mean_revenues2[-1])/(true_revenue1 + true_revenue2))*100
    # #         list_mean_final_revenues.append(difference_to_true_revenue)
    # #     plt.plot(demand_ratios, list_mean_final_revenues, label=configuration_name)
    # # plt.legend(loc='best')
    # # plt.xlabel("Demand ratio")
    # # plt.ylabel("Percentage of the optimal revenue \n average on {} flights".format(10000))
    # # # axes.set_ylim([0, 265])
    # # plt.savefig('../Results/multiagent_strategies_as_function_of_demand_ratios.png')
    #
    # # run_n_times(experience_name, env_builder, param_dict, nb_timesteps, nb_runs, callback_frequency)
    #
    # try:
    #     parameter_name = "enable_double_dqn"
    #     parameter_values = [True, False]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass

    # np.random.seed(123)
    # env.seed(123)

    # nb_actions = env.action_space.n
    # hidden_layer_size = 100
    #
    # model = Sequential()
    # model.add(Flatten(input_shape=((1,) + env.observation_space.shape)))
    # model.add(Dense(hidden_layer_size))
    # model.add(Activation('relu'))
    # model.add(Dense(hidden_layer_size))
    # model.add(Activation('relu'))
    # model.add(Dense(hidden_layer_size))
    # model.add(Activation('relu'))
    # model.add(Dense(nb_actions))
    # model.add(Activation('linear'))
    # print(model.summary())
    #
    # memory = SequentialMemory(limit=50000, window_length=1)
    # policy = EpsGreedyQPolicy(eps=.2)
    # dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
    #                enable_double_dqn=True, enable_dueling_network=True,
    #                target_model_update=1e-2, policy=policy, batch_size=32)
    # dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    #
    # rewards = callback()
    # history = dqn.fit(env, nb_steps=20000, visualize=False, verbose=2, callbacks=[rewards])

    # test_history = dqn1.test(env, nb_episodes=100, visualize=False)
    #
    # import matplotlib.pyplot as plt
    #
    # w = 100
    # moving_average1 = np.convolve(history.history['episode_reward1'], np.ones(w), 'valid') / w
    # plt.plot(list(range(0, w * len(moving_average1), w)), moving_average1, 'red', lw=4)
    # moving_average2 = np.convolve(history.history['episode_reward2'], np.ones(w), 'valid') / w
    # plt.plot(list(range(0, w * len(moving_average2), w)), moving_average2, 'blue', lw=4)
    # plt.ylim([0, 3000])
    # plt.show()
    # print("V(0,0)={}".format(max(dqn.compute_q_values([env.states[0]]))))
    # print("evaluated revenue={}".format(np.mean(test_history.history['episode_reward'])))
    #
    # Q_table = [dqn1.compute_q_values([state]) for state in env.states]
    # policy = [np.argmax(q) for q in Q_table]
    # policy = np.asarray(policy).reshape(env.observation_space.nvec)
    #
    # revenues, bookings = average_n_episodes(env, policy, 10000)
    # V = q_to_v(env, Q_table).reshape(env.observation_space.nvec)

    # revenues = np.array(callback.rewards)
    # axes = plt.gca()
    # plt.plot(absc, [true_revenue1 + true_revenue2] * len(absc), 'g--', label="Optimal solution")
    # plt.plot(absc, revenues[:, 0] + revenues[:, 1], color=parameters[configuration]["color"], label=configuration)
    # # plt.plot(absc, revenues[:, 0], color="orange", label="Flight1")
    # # plt.plot(absc, revenues[:, 1], color="blue", label="Flight2")
    # plt.legend(loc='best')
    # plt.xlabel("Number of steps")
    # plt.ylabel("Average revenue on {} flights".format(10000))
    # # axes.set_ylim([0, 265])
    # plt.show()
    #
    # indx_nb = 6
    # # bookings1 = revenues[:, 3][indx_nb]
    # # bookings2 = revenues[:, 4][indx_nb]
    # bookings1 = true_bookings_flight1
    # bookings2 = true_bookings_flight2
    #
    # # prices_proposed1 = revenues[:, 5][indx_nb]
    # # prices_proposed2 = revenues[:, 6][indx_nb]
    # prices_proposed1 = true_prices_proposed_flight1
    # prices_proposed2 = true_prices_proposed_flight2
    #
    # plt.figure()
    # width = 5
    # plt.bar(np.array(env.prices_flight2) + 2*width/3, bookings2, width, color="blue", label="Bookings flight 2")
    # plt.bar(np.array(env.prices_flight1) + 2*width/3, bookings1, width, color="orange", label="Bookings flight 1", bottom=bookings2)
    # plt.bar(np.array(env.prices_flight2) - 2*width/3, prices_proposed2, width, color="blue", alpha = 0.3, label="Prices proposed flight 2")
    # plt.bar(np.array(env.prices_flight1) - 2*width/3, prices_proposed1, width, color="orange", alpha = 0.3, label="Prices proposed flight 1", bottom=prices_proposed2)
    # plt.xlabel("Prices")
    # plt.ylabel("Average computed on 10000 flights")
    # plt.title("Overall load factor: {:.2}".format((np.sum(bookings2) + np.sum(bookings2)) / (env.C1 + env.C2)))
    # plt.legend()
    # plt.xticks(env.prices_flight1)
    # plt.show()
