import numpy as np
import gym
import matplotlib.pyplot as plt
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from visualization_and_metrics import *
from value_iteration import *
from policy_iteration import *
from q_learning import *
from deep_q_learning_tf import *
from dynamic_programming import *


if __name__ == '__main__':

    env = gym.make('gym_RM:RM-v0')
    #env = gym.make('FrozenLake-v0')
    T, C = 50, 10

    gamma = 1
    alpha = 0.05
    nb_episodes, nb_steps = 10000, 10000
    epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.9995
    
    visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    q_table, rList = dql(env, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay)
    v = q_to_v(env, q_table)
    print(v)
    visualisation_value_RM(v, T, C)
    policy = extract_policy_RM(env, v, gamma)
    visualize_policy_RM(policy, T, C)
    #print("Average reward over 1000 episodes : " + str(average_n_episodes(env, policy, 1000)))

    """
    alpha, gamma = 0.07, 0.99
    nb_episodes, nb_steps = 10000, 10000
    epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.9995

    visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    q_table, rList = q_learning(env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay)
    v = q_to_v(env, q_table)
    visualisation_value_RM(v, T, C)
    policy = extract_policy_RM(env, v, gamma)
    visualize_policy_RM(policy, T, C)
    #print("Average reward over 100 episodes : " + str(average_n_episodes(env, policy, 100)))
    """
    """
    max_iter = 100000
    epsilon = 1e-20
    gamma = 0.99

    v = value_iteration(env, gamma, max_iter, epsilon)
    visualisation_value_RM(v, T, C)
    policy = extract_policy_RM(env, v, gamma)
    visualize_policy_RM(policy, T, C)
    print("Average reward over 1000 episodes : " + str(average_n_episodes(env, policy, 1000)))
    """

    prices = [k for k in range(50, 231, 20)]
    alpha = 0.4
    lamb = 0.2
    V, P = dynamic_programming(T, C, alpha, lamb, prices)
    visualisation_value_RM(V, T, C)
    visualize_policy_RM(P, T, C)
    P = P.reshape(1, T*C)
    P = P[0]
    #print("Average reward over 1000 episodes : " + str(average_n_episodes(env, P, 1000)))
