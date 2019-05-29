import numpy as np
import gym
from dynamic_programming import dynamic_programming
from dynamic_programming_env import dynamic_programming_env
from q_learning import q_learning, q_to_v
from visualization_and_metrics import visualisation_value_RM, v_to_q, visualize_policy_RM, average_n_episodes, \
    visualizing_epsilon_decay, extract_policy_RM, plot_evolution_difference_between_policies
import matplotlib.pyplot as plt
#from actor_critic_keras import trainer, extract_values_policy
from deep_q_learning_tf_RM import dql
from value_iteration import value_iteration
from policy_iteration import policy_iteration
from actor_critic_to_solve_RM_game import train_actor_critic

if __name__ == '__main__':
    micro_times = 50
    capacity = 10
    actions = tuple(k for k in range(50, 81, 10))
    alpha = 0.7
    lamb = 0.8

    env = gym.make('gym_RMDCP:RMDCP-v0', micro_times=micro_times, capacity=capacity, actions=actions, alpha=alpha, lamb=lamb)
    print(env.P)

    # V, P_ref = dynamic_programming(env.T, env.C, env.alpha, env.lamb, env.A)
    # visualisation_value_RM(V, env.T, env.C)
    # visualize_policy_RM(P_ref, env.T, env.C)
    # P_ref = P_ref.reshape(env.T * env.C)
    # print("Average reward over 1000 episodes : " + str(average_n_episodes(env, P_ref, 1000)))

    V, P_ref = dynamic_programming_env(env)
    V = V.reshape(env.T*env.C)
    visualisation_value_RM(V, env.T, env.C)
    visualize_policy_RM(P_ref, env.T, env.C)
    P_ref = P_ref.reshape(env.T * env.C)
    print("Average reward over 1000 episodes : " + str(average_n_episodes(env, P_ref, 1000)))
    # q_model = v_to_q(env, V, 1)
    # v = q_to_v(env, q_model)
    # p = extract_policy_RM(env, v, 1)
    # print(q_model[0])
    # visualize_policy_RM(p, env.T, env.C)

    # gamma = 1
    # alpha = 0.05
    #
    # nb_episodes, nb_steps = 2000, 10000
    # epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.999
    #
    # visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    # q_table, rList = dql(env, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay, T, C)
    # print(q_table)
    # v = q_to_v(env, q_table)
    # visualisation_value_RM(v, T, C)
    # policy = extract_policy_RM(env, v, gamma)
    # visualize_policy_RM(policy, T, C)
    # #print("Average reward over 1000 episodes : " + str(average_n_episodes(env, policy, 1000)))

    # alpha, alpha_min, alpha_decay, gamma = 0.8, 0, 0.99999, 1
    # nb_episodes = 500000
    # epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.999995
    # temp = 100
    #
    # visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    # visualizing_epsilon_decay(nb_episodes, alpha, alpha_min, alpha_decay)
    # q_table, nb_episodes_list, diff_with_policy_opt_list, M = q_learning(env, alpha, alpha_min, alpha_decay, gamma,
    #                                                                      nb_episodes, epsilon,
    #                                                                      epsilon_min, epsilon_decay, P_ref, temp)
    # v = q_to_v(env, q_table)
    # visualisation_value_RM(v, env.T, env.C)
    # policy = extract_policy_RM(env, v, gamma)
    # visualize_policy_RM(policy, env.T, env.C)
    # print("Average reward over 1000 episodes : " + str(average_n_episodes(env, policy, 1000)))
    # plot_evolution_difference_between_policies(nb_episodes_list, diff_with_policy_opt_list)
    #
    # M = M.reshape(env.nS, env.action_space.n)
    # plt.imshow(M, aspect='auto')
    # plt.colorbar()
    # plt.show()

    # max_iter = 100000
    # epsilon = 1e-20
    # gamma = 0.99
    #
    # v = value_iteration(env, max_iter, epsilon)
    # visualisation_value_RM(v, env.T, env.C)
    # policy = extract_policy_RM(env, v, gamma)
    # visualize_policy_RM(policy, env.T, env.C)
    # print("Average reward over 1000 episodes : " + str(average_n_episodes(env, policy, 1000)))
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
