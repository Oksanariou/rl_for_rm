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
from deep_q_learning_keras_RM import *


if __name__ == '__main__':
    env = gym.make('gym_RM:RM-v0')
    gamma = 0.99
    alpha = 0.05
    nb_episodes, nb_steps = 10000, 1000
    epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.99995
    T, C = 500, 50


    visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    q_table, rList = dql(env, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay, C)
    policy = q_to_policy_RM(q_table)
    visualize_policy_RM(policy, T, C)
    plt.plot(rList)
    plt.show()
