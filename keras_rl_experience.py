import numpy as np
import gym
import keras
import matplotlib.pyplot as plt
from pathlib import Path
import os

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
import multiprocessing as mp

def global_env_builder():
    # Parameters of the environment
    micro_times = 100
    capacity1 = 20
    capacity2 = 20

    action_min = 50
    action_max = 231
    action_offset = 30

    actions_global = tuple((k, m) for k in range(action_min, action_max + 1, action_offset) for m in
                           range(action_min, action_max + 1, action_offset))

    demand_ratio = 1.8
    # lamb = demand_ratio * (capacity1 + capacity2) / micro_times
    lamb = 0.7

    beta = 0.04
    k_airline1 = 5.
    k_airline2 = 5.
    nested_lamb = 0.3

    return gym.make('gym_CollaborationGlobal3DMultiDiscrete:CollaborationGlobal3DMultiDiscrete-v0',
                    micro_times=micro_times,
                    capacity1=capacity1,
                    capacity2=capacity2,
                    actions=actions_global, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2,
                    lamb=lamb,
                    nested_lamb=nested_lamb)


def env_builder():
    # Parameters of the environment
    data_collection_points = 100
    micro_times = 1
    capacity = 50

    action_min = 50
    action_max = 231
    action_offset = 20

    actions = tuple(k for k in range(action_min, action_max, action_offset))
    alpha = 0.7
    lamb = 0.8

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


def build_model(env, hidden_layer_size, layers_nb):
    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=((1,) + env.observation_space.shape)))
    for k in range(layers_nb):
        model.add(Dense(hidden_layer_size))
        model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model

def parameters_dict():
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
    parameters_dict["learning_rate"] = 1e-3
    return parameters_dict

def run_once(env_builder, parameters_dict, nb_timesteps, experience_name, period, k):
    env = env_builder()
    model = build_model(env, parameters_dict["hidden_layer_size"], parameters_dict["layers_nb"])
    memory = SequentialMemory(limit=parameters_dict["memory_buffer_size"], window_length=1)
    policy = EpsGreedyQPolicy(eps=parameters_dict["epsilon"])
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=parameters_dict["nb_steps_warmup"],
                   enable_double_dqn=parameters_dict["enable_double_dqn"], enable_dueling_network=parameters_dict["enable_dueling_network"],
                   target_model_update=parameters_dict["target_model_update"], policy=policy)
    dqn.compile(Adam(lr=parameters_dict["learning_rate"]), metrics=['mae'])
    rewards = callback(env, nb_timesteps, period)
    history = dqn.fit(env, nb_steps=nb_timesteps, visualize=False, verbose=2, callbacks=[rewards])
    np.save(experience_name / ("Run" + str(k) + ".npy"), rewards.rewards)


def run_n_times(experience_name, env_builder, parameters_dict, nb_timesteps, number_of_runs, period):
    (experience_name).mkdir(parents=True, exist_ok=True)

    f = partial(run_once, env_builder, parameters_dict, nb_timesteps, experience_name, period)

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

def parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, period):
    for parameter_value in parameter_values:

        param_dict = parameters_dict()
        param_dict[parameter_name] = parameter_value
        parameter_value_name = experience_name / Path(str(parameter_value))
        parameter_value_name.mkdir(parents=True, exist_ok=True)
        # for k in range(nb_runs):
        #     run_once(env_builder, param_dict, nb_timesteps, parameter_value_name, period, k)
        run_n_times(parameter_value_name, env_builder, param_dict, nb_timesteps, nb_runs, period)
        list_of_rewards, mean_revenues = env_single_agent.collect_revenues(parameter_value_name)
        env_single_agent.plot_collected_data(mean_revenues, list_of_rewards, absc, true_revenues)
        plt.title(parameter_name + " = "+ str(parameter_value))
        plt.savefig(str(parameter_value_name) + "/" + parameter_name + " = " + parameter_value_name.name + '.png')

if __name__ == '__main__':
    # mp.set_start_method('spawn')

    env = global_env_builder()
    env_single_agent = env_builder()

    if env.observation_space.shape[0] == 2:
        true_V, true_P = dynamic_programming_env_DCP(env)
        true_revenues, true_bookings = average_n_episodes(env, true_P, 10000)
    else:
        true_V, true_P = dynamic_programming_collaboration(env)
        true_revenues, true_bookings = average_n_episodes(env, true_P, 10000)

    nb_timesteps = 100001
    callback_frequency = 10
    absc = [k for k in range(0, nb_timesteps, nb_timesteps // callback_frequency)]
    nb_runs = 20

    env_builder = global_env_builder

    # try:
    #     parameter_name = "enable_double_dqn"
    #     parameter_values = [True, False]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass
    # try:
    #     parameter_name = "batch_size"
    #     parameter_values = [32, 128, 256, 512, 1024]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_single_agent, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass
    # try:
    #     parameter_name = "hidden_layer_size"
    #     parameter_values = [10, 50, 100, 200, 300]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass
    # try:
    #     parameter_name = "layers_nb"
    #     parameter_values = [1, 2, 3, 4, 5]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass
    # try:
    #     parameter_name = "enable_dueling_network"
    #     parameter_values = [True, False]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass
    # try:
    #     parameter_name = "target_model_update"
    #     parameter_values = [0.01, 0.1, 10, 100, 1000]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass
    # try:
    #     parameter_name = "epsilon"
    #     parameter_values = [0.05, 0.1, 0.2, 0.3, 0.4]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass
    # try:
    #     parameter_name = "learning_rate"
    #     parameter_values = [1e-1, 1e-2, 1e-3, 1e-4]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass
    # try:
    #     parameter_name = "memory_buffer_size"
    #     parameter_values = [1000, 5000, 10000, 50000, 100000]
    #     experience_name = Path("../Results/global_env") / Path(parameter_name)
    #     experience_name.mkdir(parents=True, exist_ok=True)
    #     # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    #     plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)
    # except Exception:
    #     pass

    parameter_name = "memory_buffer_size"
    parameter_values = [1000, 5000, 10000, 50000]
    experience_name = Path("../Results/global_env") / Path(parameter_name)
    experience_name.mkdir(parents=True, exist_ok=True)
    # parameter_experience(experience_name, parameter_name, parameter_values, env_builder, nb_timesteps, true_revenues, absc, nb_runs, callback_frequency)
    plot_comparison(experience_name, parameter_values, env_single_agent, absc, true_revenues)


    # except Exception:
    #     pass
    # print("ok")

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
    # test_history = dqn.test(env, nb_episodes=100, visualize=False)
    #
    # import matplotlib.pyplot as plt
    #
    # w = 100
    # moving_average = np.convolve(history.history['episode_reward'], np.ones(w), 'valid') / w
    # plt.plot(list(range(0, w * len(moving_average), w)), moving_average, 'red', lw=4)
    # plt.show()
    # print("V(0,0)={}".format(max(dqn.compute_q_values([env.states[0]]))))
    # print("evaluated revenue={}".format(np.mean(test_history.history['episode_reward'])))
    #
    # Q_table = [dqn.compute_q_values([state]) for state in env.states]
    # policy = [np.argmax(q) for q in Q_table]
    # policy = np.asarray(policy).reshape(env.observation_space.nvec)
    # revenues, bookings = average_n_episodes(env, policy, 10000)
    # V = q_to_v(env, Q_table).reshape(env.observation_space.nvec)
