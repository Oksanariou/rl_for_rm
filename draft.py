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


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped
    gamma = 0.99
    alpha = 0.05
    nb_episodes, nb_steps = 3000, 100
    epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.9995

    visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    q_table, rList = dql(env, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay)
    policy = q_to_policy_FL(q_table)
    visualize_policy_FL(policy)
    print("Average reward over 100 episodes : " + str(average_n_episodes(env, policy, 100)))
    plt.plot(rList)
    plt.show()