import numpy as np
import gym
from dynamic_programming import dynamic_programming
from dynamic_programming_env import dynamic_programming_env
from q_learning import q_learning, q_to_v
from visualization_and_metrics import visualisation_value_RM, v_to_q, visualize_policy_RM, average_n_episodes, \
    visualizing_epsilon_decay, extract_policy_RM, plot_evolution_difference_between_policies, q_to_policy_RM, \
    reshape_matrix_of_visits, from_microtimes_to_DCP, average_and_std_deviation_n_episodes, plot_average_and_std_deviatione
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from actor_critic_keras import trainer, extract_values_policy
from deep_q_learning_tf_RM import dql
from value_iteration import value_iteration
from policy_iteration import policy_iteration
from actor_critic_to_solve_RM_game import train_actor_critic
from dynamic_programming_env_DCP import dynamic_programming_env_DCP
import time

if __name__ == '__main__':
    data_collection_points = 20
    micro_times = 5
    capacity = 20
    actions = tuple(k for k in range(50, 331, 20))
    alpha = 0.8
    lamb = 0.7

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)
    env_microtimes = gym.make('gym_RM:RM-v0', micro_times=data_collection_points, capacity=capacity, actions=actions, alpha=alpha, lamb=lamb)
    # print(env_DCP.P)
    # env_DCP.visualize_proba_actions()
    #
    # V, P_ref = dynamic_programming(env_microtimes.T, env_microtimes.C, env_microtimes.alpha, env_microtimes.lamb, env_microtimes.A)
    # visualisation_value_RM(V, env_microtimes.T, env_microtimes.C)
    # visualize_policy_RM(P_ref, env_microtimes.T, env_microtimes.C)
    # P_ref = P_ref.reshape(env_microtimes.T * env_microtimes.C)
    # print("Average reward over 1000 episodes : " + str(average_n_episodes(env_microtimes, P_ref, 1000)))

    # policy_DCP = from_microtimes_to_DCP(P_ref, env_microtimes, env_DCP)
    # visualize_policy_RM(policy_DCP, env_DCP.T, env_DCP.C)

    V, P_ref = dynamic_programming_env_DCP(env)
    V = V.reshape(env.T * env.C)
    visualisation_value_RM(V, env.T, env.C)
    visualize_policy_RM(P_ref, env.T, env.C)
    P_DP = P_ref.reshape(env.T * env.C)
    start_time = time.time()
    print("Average reward over 10000 episodes : " + str(average_n_episodes(env, P_DP, 10000, 0.01)))
    print("--- %s seconds ---" % (time.time() - start_time))

    nb_runs = 10_000
    mean, var = average_and_std_deviation_n_episodes(env, P_DP, nb_runs, epsilon = 0.0)
    print("Average and variance computed on {} runs : {} and {}".format(nb_runs, mean, var))

    list_of_nb_episodes = [k for k in range(10, 50_000, 500)]
    # list_of_nb_episodes = [10, 100, 1000, 5000, 8000, 10000, 20000, 30000, 50000]
    plot_average_and_std_deviatione(list_of_nb_episodes, env, P_DP, epsilon=0.)

    # q_model = v_to_q(env, V, 1)
    # v = q_to_v(env, q_model)
    # p = extract_policy_RM(env, v, 1)
    # print(q_model[0])
    # visualize_policy_RM(p, env.T, env.C)

    # gamma = 1
    # alpha = 0.05
    #
    # nb_episodes, nb_steps = 2000, 10000
    # epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.995
    #
    # visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    # q_table, rList = dql(env, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay, env.T, env.C)
    # print(q_table)
    # v = q_to_v(env, q_table)
    # visualisation_value_RM(v, env.T, env.C)
    # policy = extract_policy_RM(env, v, gamma)
    # visualize_policy_RM(policy, env.T, env.C)
    # print("Average reward over 1000 episodes : " + str(average_n_episodes(env, policy, 1000)))

    alpha, alpha_min, alpha_decay, gamma = 0.8, 0, 0.99999, 0.99
    nb_episodes = 600000
    epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.99999
    temp = 100

    visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    # visualizing_epsilon_decay(nb_episodes, alpha, alpha_min, alpha_decay)
    q_table, nb_episodes_list, diff_with_policy_opt_list, M, trajectories, revenues = q_learning(env, alpha, alpha_min, alpha_decay, gamma,
                                                                         nb_episodes, epsilon,
                                                                         epsilon_min, epsilon_decay, P_ref, temp)

    v = q_to_v(env, q_table)
    visualisation_value_RM(v, env.T, env.C)
    policy = q_to_policy_RM(env, q_table)
    visualize_policy_RM(policy, env.T, env.C)
    print("Average reward over 10000 episodes : " + str(average_n_episodes(env, policy, 10000)))
    plot_evolution_difference_between_policies(nb_episodes_list, diff_with_policy_opt_list)
    #
    # #M = M.reshape(env.nS, env.action_space.n)
    # X, Y, Z, values = reshape_matrix_of_visits(M, env)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # #plt.imshow(M, aspect='auto')
    # p = ax.scatter3D(X, Y, Z, c=values, cmap='hot')
    # fig.colorbar(p, ax=ax)
    # #plt.colorbar()
    # plt.show()
    #
    # plt.imshow(trajectories, aspect = 'auto')
    # plt.colorbar()
    # plt.show()

    max_iter = 100000
    epsilon = 1e-20
    gamma = 0.99

    v = value_iteration(env, max_iter, epsilon)
    visualisation_value_RM(v, env.T, env.C)
    P_VI = extract_policy_RM(env, v, gamma)
    visualize_policy_RM(P_VI, env.T, env.C)
    print("Average reward over 10000 episodes : " + str(average_n_episodes(env, P_VI, 10000)))
    #
    # max_iter = 100000
    # epsilon = 1e-20
    # gamma = 0.99
    #
    # policy = policy_iteration(env, gamma, max_iter, epsilon)
    # visualize_policy_RM(policy, env.T, env.C)
    #
    # print(difference_between_policies(policy, P_ref))

    #
    # state_size = len(env.observation_space.spaces)
    # epochs = 5000
    # batch_size = 32
    # gamma = 0.99
    # epsilon = 1
    # epsilon_min = 0.1
    # epsilon_decay = 0.9995
    #
    # visualizing_epsilon_decay(epochs, epsilon, epsilon_min, epsilon_decay)
    #
    # trained_actor_network = trainer(env, q_table, epochs, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, state_size)
    # values, policy = extract_values_policy(env, trained_actor_network, state_size)
    # visualisation_value_RM(values, env.T, env.C)
    # visualize_policy_RM(policy, env.T, env.C)
    #
    # alpha, alpha_min, alpha_decay, gamma = 0.8, 0, 0.99999, 1
    # nb_episodes = 10000
    # epsilon, epsilon_decay = 1, 0.9995
    # lr = 0.001
    # tau = 0.125
    # visualizing_epsilon_decay(nb_episodes, epsilon, 0, epsilon_decay)
    # policy = train_actor_critic(env, nb_episodes, epsilon, epsilon_decay, gamma, tau, lr)
    # visualize_policy_RM(policy, env.T, env.C)

    plt.plot([nb_episodes_list[0], nb_episodes_list[-1]], [average_n_episodes(env, P_DP, 10000, 0.01)] * 2,
             label="DP revenue", c='red', lw=2)
    plt.plot([nb_episodes_list[0], nb_episodes_list[-1]], [average_n_episodes(env, P_VI, 10000, 0.01)] * 2,
             label="VI revenue", c='orange', lw=2)
    plt.plot(nb_episodes_list, revenues, label="Q-Learning revenue", c='blue', lw=2)
    plt.legend()
    plt.ylabel("Revenue")
    plt.xlabel("Number of episodes")
    plt.title("Average revenue")
    plt.show()
