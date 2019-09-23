import gym
import numpy as np
import matplotlib.pyplot as plt
import os

from dynamic_programming_env import dynamic_programming_collaboration
from dynamic_programming_env_DCP import dynamic_programming_env_DCP

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACKTR
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from scipy.stats import sem, t
import glob


def ACKTR_agent_builder(env_vec):
    return ACKTR(MlpPolicy, env_vec, learning_rate=0.00001)


def collaboration_environment_parameters():
    env_parameters = {}
    env_parameters["micro_times"] = 100
    env_parameters["capacity1"] = 11
    env_parameters["capacity2"] = 11
    env_parameters["action_min"] = 10
    env_parameters["action_max"] = 230
    env_parameters["action_offset"] = 20
    env_parameters["actions"] = tuple((k, m) for k in
                                      range(env_parameters["action_min"], env_parameters["action_max"] + 1,
                                            env_parameters["action_offset"]) for m in
                                      range(env_parameters["action_min"], env_parameters["action_max"] + 1,
                                            env_parameters["action_offset"]))
    env_parameters["lamb"] = 0.4
    env_parameters["beta"] = 0.02
    env_parameters["k_airline1"] = 1.5
    env_parameters["k_airline2"] = 1.5
    env_parameters["nested_lamb"] = 0.3
    return env_parameters


def RMDCPDiscrete_environment_parameters():
    env_parameters = {}
    env_parameters["data_collection_points"] = 80
    env_parameters["micro_times"] = 5
    env_parameters["capacity"] = 50
    env_parameters["action_min"] = 50
    env_parameters["action_max"] = 230
    env_parameters["action_offset"] = 30
    env_parameters["actions"] = [k for k in range(env_parameters["action_min"], env_parameters["action_max"] + 1,
                                                  env_parameters["action_offset"])]
    env_parameters["lamb"] = 0.7
    env_parameters["alpha"] = 0.8
    return env_parameters


def RMDCPDiscrete_env_builder(env_parameters):
    return gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0',
                    micro_times=env_parameters["micro_times"],
                    data_collection_points=env_parameters["data_collection_points"],
                    capacity=env_parameters["capacity"],
                    actions=env_parameters["actions"], alpha=env_parameters["alpha"],
                    lamb=env_parameters["lamb"])


def CollaborationGlobal3D_env_builder(env_parameters):
    return gym.make('gym_CollaborationGlobal3D:CollaborationGlobal3D-v0',
                    micro_times=env_parameters["micro_times"],
                    capacity1=env_parameters["capacity1"],
                    capacity2=env_parameters["capacity2"],
                    actions=env_parameters["actions"], beta=env_parameters["beta"],
                    k_airline1=env_parameters["k_airline1"], k_airline2=env_parameters["k_airline2"],
                    lamb=env_parameters["lamb"],
                    nested_lamb=env_parameters["nested_lamb"])


def CollaborationGlobal3DMultiDiscrete_env_builder(env_parameters):
    return gym.make('gym_CollaborationGlobal3DMultiDiscrete:CollaborationGlobal3DMultiDiscrete-v0',
                    micro_times=env_parameters["micro_times"],
                    capacity1=env_parameters["capacity1"],
                    capacity2=env_parameters["capacity2"],
                    actions=env_parameters["actions"], beta=env_parameters["beta"],
                    k_airline1=env_parameters["k_airline1"], k_airline2=env_parameters["k_airline2"],
                    lamb=env_parameters["lamb"],
                    nested_lamb=env_parameters["nested_lamb"])


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, rewards, env, model, callback_frequency
    if n_steps == 0 or ((n_steps + 1) % callback_frequency == 0):
        print(n_steps)
        policy, _ = model.predict(env.states)
        rewards.append(env.average_n_episodes(policy, 10000))
        print(rewards)
    n_steps += 1
    return True


def run_once(env_builder, env_parameters, agent_builder, nb_timesteps, frequency, experience_name, k):
    global env, rewards, n_steps, steps, model, count, callback_frequency
    callback_frequency = frequency

    env = env_builder(env_parameters)
    env_vec = DummyVecEnv([lambda: env])

    rewards, n_steps = [], 0

    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    model = agent_builder(env_vec)
    model.learn(total_timesteps=nb_timesteps, callback=callback)

    np.save(experience_name / ("Run" + str(k) + ".npy"), rewards)


def run_n_times(experience_name, env_builder, env_parameter, agent_builder, nb_timesteps, number_of_runs,
                callback_frequency):
    (experience_name).mkdir(parents=True, exist_ok=True)

    f = partial(run_once, env_builder, env_parameter, agent_builder, nb_timesteps, callback_frequency, experience_name)

    with Pool(number_of_runs) as pool:
        pool.map(f, range(number_of_runs))


def collect_list_of_mean_revenues(experience_name):
    list_of_rewards = []
    for np_name in glob.glob(str(experience_name) + '/*.np[yz]'):
        list_of_rewards.append(list(np.load(np_name, allow_pickle=True)))

    nb_collection_points = len(list_of_rewards[0])

    all_rewards_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]

    for k in range(len(list_of_rewards)):
        rewards = list_of_rewards[k]
        for i in range(nb_collection_points):
            all_rewards_combined_at_each_collection_point[i].append(rewards[i][0])

    mean_revenues = [np.mean(list) for list in all_rewards_combined_at_each_collection_point]
    std_revenues = [sem(list) for list in all_rewards_combined_at_each_collection_point]
    confidence_revenues = [std_revenues[k] * t.ppf((1 + 0.95) / 2, nb_collection_points - 1) for k in
                           range(nb_collection_points)]
    min_revenues = [mean_revenues[k] - confidence_revenues[k] for k in range(nb_collection_points)]
    max_revenues = [mean_revenues[k] + confidence_revenues[k] for k in range(nb_collection_points)]

    return mean_revenues, min_revenues, max_revenues


def plot_revenues(mean_revenues, min_revenues, max_revenues, callback_frequency, optimal_revenue):
    steps = [0]
    for k in range(len(mean_revenues) - 1):
        steps.append(callback_frequency + steps[-1] - 1)

    fig = plt.figure()
    plt.plot(steps, mean_revenues, color="gray", label='ACKTR mean revenue')
    plt.fill_between(steps, min_revenues, max_revenues, label='95% confidence interval', color="gray", alpha=0.2)
    plt.plot(steps, [optimal_revenue] * len(steps), label="Optimal solution")
    plt.legend()
    plt.ylabel("Average revenue on 10000 flights")
    plt.xlabel("Number of episodes")
    return fig


if __name__ == '__main__':
    collab_env_parameters = collaboration_environment_parameters()
    RMDCP_env_parameters = RMDCPDiscrete_environment_parameters()
    learning_rate = 0.1
    RMDCP_env_parameters["learning_rate"] = learning_rate

    # env_discrete = CollaborationGlobal3D_env_builder(collab_env_parameters)
    # V, P_global = dynamic_programming_collaboration(env_discrete)
    # P_global = P_global.reshape(env_discrete.T * env_discrete.C1 * env_discrete.C2)
    # P_global = [int(a) for a in P_global]
    # optimal_revenue = V[0][0][0]

    env_rmdcp = RMDCPDiscrete_env_builder(RMDCP_env_parameters)
    V, P = dynamic_programming_env_DCP(env_rmdcp)
    P = P.reshape(env_rmdcp.T * env_rmdcp.C)
    P = [int(a) for a in P]
    optimal_revenue = V[0][0]

    nb_timesteps = 10000
    number_of_runs = 20
    callback_frequency = int((nb_timesteps/20)/10)
    # experience_name = Path("../Results/ACKTR/Collaboration_medium_env_dr_1_8")
    experience_name = Path("../Results/Test_ACKTR/lr_of_"+str(learning_rate))
    experience_name.mkdir(parents=True, exist_ok=True)

    run_n_times(experience_name, RMDCPDiscrete_env_builder, RMDCP_env_parameters,ACKTR_agent_builder, nb_timesteps, number_of_runs,
                callback_frequency)

    mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(experience_name)
    figure = plot_revenues(mean_revenues, min_revenues, max_revenues, callback_frequency, optimal_revenue)

    plt.savefig(str(experience_name) + "/" + experience_name.name + '.png')

    RMDCP_env_parameters = RMDCPDiscrete_environment_parameters()
    learning_rate = 0.01
    RMDCP_env_parameters["learning_rate"] = learning_rate

    # env_discrete = CollaborationGlobal3D_env_builder(collab_env_parameters)
    # V, P_global = dynamic_programming_collaboration(env_discrete)
    # P_global = P_global.reshape(env_discrete.T * env_discrete.C1 * env_discrete.C2)
    # P_global = [int(a) for a in P_global]
    # optimal_revenue = V[0][0][0]

    env_rmdcp = RMDCPDiscrete_env_builder(RMDCP_env_parameters)
    V, P = dynamic_programming_env_DCP(env_rmdcp)
    P = P.reshape(env_rmdcp.T * env_rmdcp.C)
    P = [int(a) for a in P]
    optimal_revenue = V[0][0]

    nb_timesteps = 10000
    number_of_runs = 20
    callback_frequency = int((nb_timesteps/20)/10)
    # experience_name = Path("../Results/ACKTR/Collaboration_medium_env_dr_1_8")
    experience_name = Path("../Results/Test_ACKTR/lr_of_"+str(learning_rate))
    experience_name.mkdir(parents=True, exist_ok=True)

    run_n_times(experience_name, CollaborationGlobal3D_env_builder, collab_env_parameters,ACKTR_agent_builder, nb_timesteps, number_of_runs,
                callback_frequency)

    mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(experience_name)
    figure = plot_revenues(mean_revenues, min_revenues, max_revenues, callback_frequency, optimal_revenue)

    plt.savefig(str(experience_name) + "/" + experience_name.name + '.png')

    RMDCP_env_parameters = RMDCPDiscrete_environment_parameters()
    learning_rate = 0.001
    RMDCP_env_parameters["learning_rate"] = learning_rate

    # env_discrete = CollaborationGlobal3D_env_builder(collab_env_parameters)
    # V, P_global = dynamic_programming_collaboration(env_discrete)
    # P_global = P_global.reshape(env_discrete.T * env_discrete.C1 * env_discrete.C2)
    # P_global = [int(a) for a in P_global]
    # optimal_revenue = V[0][0][0]

    env_rmdcp = RMDCPDiscrete_env_builder(RMDCP_env_parameters)
    V, P = dynamic_programming_env_DCP(env_rmdcp)
    P = P.reshape(env_rmdcp.T * env_rmdcp.C)
    P = [int(a) for a in P]
    optimal_revenue = V[0][0]

    nb_timesteps = 10000
    number_of_runs = 20
    callback_frequency = int((nb_timesteps/20)/10)
    # experience_name = Path("../Results/ACKTR/Collaboration_medium_env_dr_1_8")
    experience_name = Path("../Results/Test_ACKTR/lr_of_"+str(learning_rate))
    experience_name.mkdir(parents=True, exist_ok=True)

    run_n_times(experience_name, CollaborationGlobal3D_env_builder, collab_env_parameters,ACKTR_agent_builder, nb_timesteps, number_of_runs,
                callback_frequency)

    mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(experience_name)
    figure = plot_revenues(mean_revenues, min_revenues, max_revenues, callback_frequency, optimal_revenue)

    plt.savefig(str(experience_name) + "/" + experience_name.name + '.png')

    RMDCP_env_parameters = RMDCPDiscrete_environment_parameters()
    learning_rate = 0.0001
    RMDCP_env_parameters["learning_rate"] = learning_rate

    # env_discrete = CollaborationGlobal3D_env_builder(collab_env_parameters)
    # V, P_global = dynamic_programming_collaboration(env_discrete)
    # P_global = P_global.reshape(env_discrete.T * env_discrete.C1 * env_discrete.C2)
    # P_global = [int(a) for a in P_global]
    # optimal_revenue = V[0][0][0]

    env_rmdcp = RMDCPDiscrete_env_builder(RMDCP_env_parameters)
    V, P = dynamic_programming_env_DCP(env_rmdcp)
    P = P.reshape(env_rmdcp.T * env_rmdcp.C)
    P = [int(a) for a in P]
    optimal_revenue = V[0][0]

    nb_timesteps = 10000
    number_of_runs = 20
    callback_frequency = int((nb_timesteps/20)/10)
    # experience_name = Path("../Results/ACKTR/Collaboration_medium_env_dr_1_8")
    experience_name = Path("../Results/Test_ACKTR/lr_of_"+str(learning_rate))
    experience_name.mkdir(parents=True, exist_ok=True)

    run_n_times(experience_name, CollaborationGlobal3D_env_builder, collab_env_parameters,ACKTR_agent_builder, nb_timesteps, number_of_runs,
                callback_frequency)

    mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(experience_name)
    figure = plot_revenues(mean_revenues, min_revenues, max_revenues, callback_frequency, optimal_revenue)

    plt.savefig(str(experience_name) + "/" + experience_name.name + '.png')

    RMDCP_env_parameters = RMDCPDiscrete_environment_parameters()
    learning_rate = 0.00001
    RMDCP_env_parameters["learning_rate"] = learning_rate

    # env_discrete = CollaborationGlobal3D_env_builder(collab_env_parameters)
    # V, P_global = dynamic_programming_collaboration(env_discrete)
    # P_global = P_global.reshape(env_discrete.T * env_discrete.C1 * env_discrete.C2)
    # P_global = [int(a) for a in P_global]
    # optimal_revenue = V[0][0][0]

    env_rmdcp = RMDCPDiscrete_env_builder(RMDCP_env_parameters)
    V, P = dynamic_programming_env_DCP(env_rmdcp)
    P = P.reshape(env_rmdcp.T * env_rmdcp.C)
    P = [int(a) for a in P]
    optimal_revenue = V[0][0]

    nb_timesteps = 10000
    number_of_runs = 20
    callback_frequency = int((nb_timesteps/20)/10)
    # experience_name = Path("../Results/ACKTR/Collaboration_medium_env_dr_1_8")
    experience_name = Path("../Results/Test_ACKTR/lr_of_"+str(learning_rate))
    experience_name.mkdir(parents=True, exist_ok=True)

    run_n_times(experience_name, CollaborationGlobal3D_env_builder, collab_env_parameters,ACKTR_agent_builder, nb_timesteps, number_of_runs,
                callback_frequency)

    mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(experience_name)
    figure = plot_revenues(mean_revenues, min_revenues, max_revenues, callback_frequency, optimal_revenue)

    plt.savefig(str(experience_name) + "/" + experience_name.name + '.png')

    RMDCP_env_parameters = RMDCPDiscrete_environment_parameters()
    learning_rate = 0.000001
    RMDCP_env_parameters["learning_rate"] = learning_rate

    # env_discrete = CollaborationGlobal3D_env_builder(collab_env_parameters)
    # V, P_global = dynamic_programming_collaboration(env_discrete)
    # P_global = P_global.reshape(env_discrete.T * env_discrete.C1 * env_discrete.C2)
    # P_global = [int(a) for a in P_global]
    # optimal_revenue = V[0][0][0]

    env_rmdcp = RMDCPDiscrete_env_builder(RMDCP_env_parameters)
    V, P = dynamic_programming_env_DCP(env_rmdcp)
    P = P.reshape(env_rmdcp.T * env_rmdcp.C)
    P = [int(a) for a in P]
    optimal_revenue = V[0][0]

    nb_timesteps = 10000
    number_of_runs = 20
    callback_frequency = int((nb_timesteps/20)/10)
    # experience_name = Path("../Results/ACKTR/Collaboration_medium_env_dr_1_8")
    experience_name = Path("../Results/Test_ACKTR/lr_of_"+str(learning_rate))
    experience_name.mkdir(parents=True, exist_ok=True)

    run_n_times(experience_name, CollaborationGlobal3D_env_builder, collab_env_parameters,ACKTR_agent_builder, nb_timesteps, number_of_runs,
                callback_frequency)

    mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(experience_name)
    figure = plot_revenues(mean_revenues, min_revenues, max_revenues, callback_frequency, optimal_revenue)

    plt.savefig(str(experience_name) + "/" + experience_name.name + '.png')


