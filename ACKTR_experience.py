import gym
import numpy as np
import matplotlib.pyplot as plt
import os

from Collaboration_Competition.DQN_single_agent import learn_single_agent_collaboration_global, run_DQN_single_agent
from Collaboration_Competition.DQN_multi_agent import run_DQN_multi_agent
from dynamic_programming_env import dynamic_programming_collaboration
from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from visualization_and_metrics import average_n_episodes_collaboration_global_policy, average_n_episodes
from Collaboration_Competition.competition import plot_global_bookings_histograms
from Collaboration_Competition.q_learning_collaboration import q_learning_global

from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACKTR, DQN
from functools import partial
from multiprocessing import Pool


def ACKTR_agent_builder(env_vec):
    return ACKTR(MlpPolicy, env_vec)

def CollaborationGlobal3D_env_builder():
    micro_times = 5
    capacity1 = 2
    capacity2 = 2

    action_min = 50
    action_max = 160
    action_offset = 100
    actions_global = tuple((k, m) for k in range(action_min, action_max + 1, action_offset) for m in
                           range(action_min, action_max + 1, action_offset))
    actions_individual = tuple(k for k in range(action_min, action_max + 1, action_offset))

    demand_ratio = 1.8
    # lamb = demand_ratio * (capacity1 + capacity2) / micro_times
    lamb = 0.9

    beta = 0.04
    k_airline1 = 5
    k_airline2 = 5
    nested_lamb = 0.3
    return gym.make('gym_CollaborationGlobal3DMultiDiscrete:CollaborationGlobal3DMultiDiscrete-v0', micro_times=micro_times,
                    capacity1=capacity1,
                    capacity2=capacity2,
                    actions=actions_global, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2,
                    lamb=lamb,
                    nested_lamb=nested_lamb)

def parameters_builder_DQN():
    parameters_dict = {}
    parameters_dict["env_builder"] = CollaborationGlobal3D_env_builder
    parameters_dict["gamma"] = 0.99
    parameters_dict["learning_rate"] = 0.001
    parameters_dict["buffer_size"] = 500
    parameters_dict["exploration_fraction"] = 0.6
    parameters_dict["exploration_final_eps"] = 0.01
    parameters_dict["train_freq"] = 1
    parameters_dict["batch_size"] = 32
    parameters_dict["checkpoint_freq"] = 10000
    parameters_dict["checkpoint_path"] = None
    parameters_dict["learning_starts"] = 1
    parameters_dict["target_network_update_freq"] = 50
    parameters_dict["prioritized_replay"] = False
    parameters_dict["prioritized_replay_alpha"] = 0.6
    parameters_dict["prioritized_replay_beta0"] = 0.4
    parameters_dict["prioritized_replay_beta_iters"] = None
    parameters_dict["prioritized_replay_eps"] = 1e-6
    parameters_dict["param_noise"] = False
    parameters_dict["verbose"] = 0
    parameters_dict["tensorboard_log"] = None
    parameters_dict["policy_kwargs"] = {}
    parameters_dict["weights"] = False

    # env = parameters_dict["env_builder"]()
    parameters_dict["original_weights"] = None

    return parameters_dict

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, rewards, env, model, callback_frequency
    if n_steps == 0 or ((n_steps + 1) % callback_frequency == 0):
        print(n_steps)
        policy,_ = model.predict(env.states)
        rewards.append(env.average_n_episodes(policy, 10000))
    n_steps += 1
    return True

def run_once(env_builder, agent_builder, nb_timesteps, frequency, k):
    global env, rewards, n_steps, steps, model, count, callback_frequency
    callback_frequency = frequency

    env = env_builder()
    env_vec = DummyVecEnv([lambda: env])

    rewards, n_steps = [], 0

    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    model = agent_builder(env_vec)
    model.learn(total_timesteps=nb_timesteps, callback=callback)

def run_n_times(env_builder, agent_builder, nb_timesteps, number_of_runs, callback_frequency):

    f = partial(run_once, env_builder, agent_builder, nb_timesteps, callback_frequency)

    with Pool(number_of_runs) as pool:
        pool.map(f, range(number_of_runs))


if __name__ == '__main__':
    callback_frequency = 500
    total_timesteps = 30000
    number_of_runs = 10

    run_n_times(CollaborationGlobal3D_env_builder, ACKTR_agent_builder, total_timesteps, number_of_runs, callback_frequency)

    # Dynamic Programming
    # V, P_global = dynamic_programming_collaboration(collab_global_env)
    # P_global = P_global.reshape(collab_global_env.T * collab_global_env.C1 * collab_global_env.C2)
    # revenues_global, bookings_global = average_n_episodes_collaboration_global_policy(collab_global_env, P_global,
    #                                                                                   individual_3D_env1,
    #                                                                                   10000)

    # V, P_global = dynamic_programming_env_DCP(simple_env)
    # P_global = P_global.reshape(simple_env.T * simple_env.C)
    # revenues_global, bookings_global = average_n_episodes(simple_env, P_global,10000)

    # episodes = [k for k in range(0, total_timesteps + 1, callback_frequency)]

    # revenues = np.array(revenues)
    # plt.figure()
    # plt.plot(episodes, [revenues_global] * len(episodes), 'g--', label="Optimal P_Global")
    # plt.plot(episodes, np.array(revenues[:, 0]), label="DQN P_Global")
    # plt.legend()
    # plt.ylabel("Average revenue on 10000 flights")
    # plt.xlabel("Number of episodes")
    # plt.show()

    # revenues = np.array(revenues)
    # plt.figure()
    # plt.plot(episodes, [revenues_global[0] + revenues_global[1]] * len(episodes), 'g--', label="Optimal P_Global")
    # plt.plot(episodes, np.array(revenues[:, 0][:, 0]) + np.array(revenues[:, 0][:, 1]), label="DQN P_Global")
    # plt.legend()
    # plt.ylabel("Average revenue on 10000 flights")
    # plt.xlabel("Number of episodes")
    # plt.show()
    #
    # individual_env = gym.make('gym_CompetitionIndividual2D:CompetitionIndividual2D-v0', capacity=capacity1,
    #                           micro_times=micro_times, actions=actions_individual, lamb=lamb, beta=beta,
    #                           k=k_airline1,
    #                           nested_lamb=nested_lamb,
    #                           competition_aware=False)
    # plot_global_bookings_histograms(individual_env,
    #                                 [revenues[:, 1][:, 0][-1],
    #                                  revenues[:, 1][:, 1][-1]])
    # plot_global_bookings_histograms(individual_env,
    #                                 bookings_global)


    # plot_global_bookings_histograms(simple_env,
    #                                  [revenues[:, 1][-1],
    #                                  [0 for k in range(simple_env.nA)]])
    # plot_global_bookings_histograms(simple_env,
    #                                 [bookings_global, [0 for k in range(simple_env.nA)]])
