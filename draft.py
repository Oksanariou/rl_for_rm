import numpy as np
import gym
from dynamic_programming import *
from q_learning import q_learning, q_to_v
from value_iteration import value_iteration

if __name__ == '__main__':
    micro_times = 50
    capacity = 10
    actions = tuple(k for k in range(50, 231, 20))
    alpha = 0.6
    lamb = 0.4

    env = gym.make('gym_RM:RM-v0', micro_times=micro_times, capacity=capacity, actions=actions, alpha=alpha, lamb=lamb)

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

    alpha, gamma = 0.07, 0.99
    nb_episodes, nb_steps = 50000, 10000
    epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.99995

    visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    q_table, rList = q_learning(env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay)
    v = q_to_v(env, q_table)
    visualisation_value_RM(v, env.T, env.C)
    policy = extract_policy_RM(env, v, gamma)
    visualize_policy_RM(policy, env.T, env.C)
    print("Average reward over 1000 episodes : " + str(average_n_episodes(env, policy, 1000)))

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
    # V, P = dynamic_programming(env.T, env.C, env.alpha, env.lamb, env.A)
    # visualisation_value_RM(V, env.T, env.C)
    # visualize_policy_RM(P, env.T, env.C)
    # P = P.reshape(1, env.T * env.C)
    # P = P[0]
    # print("Average reward over 1000 episodes : " + str(average_n_episodes(env, P, 1000)))
