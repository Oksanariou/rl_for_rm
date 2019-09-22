import gym
import numpy as np
import matplotlib.pyplot as plt

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
from stable_baselines import A2C, SAC, PPO1, ACKTR, DQN


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

def simple_agent_env_builder():
    dcp = 100
    capacity = 50
    lamb = 0.65
    alpha = 0.65
    actions = [k for k in range(50, 231, 20)]
    return gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', micro_times=1, data_collection_points=dcp,
                          capacity=capacity, lamb=lamb, alpha=alpha, actions=actions)


def parameters_dict_builder():
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
    global n_steps, revenues, env, states, agent, callback_frequency
    if n_steps == 0:
        print(n_steps)
        policy, _ = agent.predict(states)
        policy = [env.A[a] for a in policy]
        # save_values(env, q_values, '../Results', str(n_steps))
        revenues.append(average_n_episodes(env, policy, 10000, agent))
    # Print stats every 1000 calls
    if (n_steps + 1) % callback_frequency == 0:
        print(n_steps)
        policy,_ = agent.predict(states)
        policy = [env.A[a] for a in policy]
        # save_values(env, q_values, '../Results', str(n_steps))
        revenues.append(average_n_episodes(env, policy, 10000, agent))
    n_steps += 1
    return True


if __name__ == '__main__':
    global callback_frequency, n_steps, revenues, states, agent, env
    callback_frequency = 500
    n_steps = 0
    revenues = []

    micro_times = 5
    capacity1 = 2
    capacity2 = 2

    action_min = 50
    action_max = 155
    action_offset = 100
    actions_global = tuple((k, m) for k in range(action_min, action_max + 1, action_offset) for m in
                           range(action_min, action_max + 1, action_offset))
    actions_individual = tuple(k for k in range(action_min, action_max + 1, action_offset))

    # demand_ratio = 1.8
    # lamb = demand_ratio * (capacity1 + capacity2) / micro_times
    lamb = 0.9

    beta = 0.04
    k_airline1 = 5
    k_airline2 = 5
    nested_lamb = 0.3

    collab_global_env = CollaborationGlobal3D_env_builder()

    env_md = gym.make('gym_CollaborationGlobal3DMultiDiscrete:CollaborationGlobal3DMultiDiscrete-v0', micro_times=micro_times,
                          capacity1=capacity1,
                          capacity2=capacity2,
                          actions=actions_global, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2,
                          lamb=lamb,
                          nested_lamb=nested_lamb)
    env_d = gym.make('gym_CollaborationGlobal3D:CollaborationGlobal3D-v0', micro_times=micro_times,
                          capacity1=capacity1,
                          capacity2=capacity2,
                          actions=actions_global, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2,
                          lamb=lamb,
                          nested_lamb=nested_lamb)

    # env = gym.make('gym_RMDCP:RMDCP-v0', micro_times=1, data_collection_points=10,
    #                       capacity=8, lamb=0.9, alpha=0.8, actions=[50, 100, 150, 200])
    env = env_d
    # states = [[t, x1, x2] for t in range(env.T) for x1 in range(env.C1) for x2 in range(env.C2)]
    states = [s for s in range(env.nS)]
    env_vec = DummyVecEnv([lambda: env])
    agent = ACKTR(MlpPolicy, env_vec)
    agent.learn(total_timesteps=30000, callback=callback)

    individual_3D_env1 = gym.make('gym_CollaborationIndividual3D:CollaborationIndividual3D-v0',
                                  micro_times=micro_times,
                                  capacity1=capacity1, capacity2=capacity2, actions=actions_individual)

    callback_frequency = 1000
    total_timesteps = 30000

    parameters_dict = parameters_dict_builder()
    parameters_dict1, parameters_dict2 = parameters_dict_builder(), parameters_dict_builder()

    # Dynamic Programming
    V, P_global = dynamic_programming_collaboration(collab_global_env)
    P_global = P_global.reshape(collab_global_env.T * collab_global_env.C1 * collab_global_env.C2)
    revenues_global, bookings_global = average_n_episodes_collaboration_global_policy(collab_global_env, P_global,
                                                                                      individual_3D_env1,
                                                                                      10000)

    # V, P_global = dynamic_programming_env_DCP(simple_env)
    # P_global = P_global.reshape(simple_env.T * simple_env.C)
    # revenues_global, bookings_global = average_n_episodes(simple_env, P_global,10000)

    # Run DQN with single agent
    revenues = run_DQN_single_agent(parameters_dict, callback_frequency, total_timesteps,
                                                       individual_3D_env1)
    # revenues = run_DQN_single_agent(parameters_dict, callback_frequency, total_timesteps, individual_3D_env1)

    # gamma = 0.99
    # # alpha, alpha_min, alpha_decay = 0.4, 0.02, 0.9999985
    # alpha, alpha_min, alpha_decay = 0.8, 0, 0.9999975
    # # alpha, alpha_min, alpha_decay = 0.2, 0, 0.9999975
    # beta, beta_min, beta_decay = alpha / 2, 0, 0.9999975
    # epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.9999975
    # total_timesteps_q_learning = 2000000
    # Q_table_global, revenues_QL_global, bookings_QL_global, _ = q_learning_global(collab_global_env, individual_3D_env1,
    #                                                                               alpha, alpha_min, alpha_decay,
    #                                                                               gamma,
    #                                                                               total_timesteps_q_learning,
    #                                                                               epsilon,
    #                                                                               epsilon_min, epsilon_decay)
    # revenues_QL_global = np.array(revenues_QL_global)

    episodes = [k for k in range(0, total_timesteps + 1, callback_frequency)]

    # revenues = np.array(revenues)
    # plt.figure()
    # plt.plot(episodes, [revenues_global] * len(episodes), 'g--', label="Optimal P_Global")
    # plt.plot(episodes, np.array(revenues[:, 0]), label="DQN P_Global")
    # plt.legend()
    # plt.ylabel("Average revenue on 10000 flights")
    # plt.xlabel("Number of episodes")
    # plt.show()

    revenues = np.array(revenues)
    plt.figure()
    plt.plot(episodes, [revenues_global[0] + revenues_global[1]] * len(episodes), 'g--', label="Optimal P_Global")
    plt.plot(episodes, np.array(revenues[:, 0][:, 0]) + np.array(revenues[:, 0][:, 1]), label="DQN P_Global")
    plt.legend()
    plt.ylabel("Average revenue on 10000 flights")
    plt.xlabel("Number of episodes")
    plt.show()

    individual_env = gym.make('gym_CompetitionIndividual2D:CompetitionIndividual2D-v0', capacity=capacity1,
                              micro_times=micro_times, actions=actions_individual, lamb=lamb, beta=beta,
                              k=k_airline1,
                              nested_lamb=nested_lamb,
                              competition_aware=False)
    plot_global_bookings_histograms(individual_env,
                                    [revenues[:, 1][:, 0][-1],
                                     revenues[:, 1][:, 1][-1]])
    plot_global_bookings_histograms(individual_env,
                                    bookings_global)


    # plot_global_bookings_histograms(simple_env,
    #                                  [revenues[:, 1][-1],
    #                                  [0 for k in range(simple_env.nA)]])
    # plot_global_bookings_histograms(simple_env,
    #                                 [bookings_global, [0 for k in range(simple_env.nA)]])
