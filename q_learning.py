import numpy as np
import random
from scipy.special import softmax
import matplotlib.pyplot as plt

from visualization_and_metrics import visualisation_value_RM, extract_policy_RM, visualize_policy_RM, \
    average_n_episodes, difference_between_policies, q_to_policy_RM


def softmax_(env, Q, state, temp):
    sum = 0
    proba_actions = []
    for k in range(env.action_space.n):
        sum += np.exp(Q[state][k] / temp)
    for k in range(env.action_space.n):
        proba_actions.append(np.exp(Q[state][k] / temp) / sum)
    return proba_actions


def q_learning(env, alpha, alpha_min, alpha_decay, gamma, nb_episodes, epsilon, epsilon_min, epsilon_decay, P_ref, temp):
    # Initialize the Q-table with zeros
    Q = np.ones([env.nS, env.action_space.n])
    M = np.zeros([env.T, env.C, env.action_space.n])
    trajectories = np.zeros([env.T, env.C])
    Q[:] = 0
    diff_with_policy_opt_list = []
    nb_episodes_list = []

    for episode in range(nb_episodes):
        state = env.set_random_state()

        state_idx = env.to_idx(*state)
        done = False
        while not done:
            # action_idx = np.random.choice(env.action_space.n, 1, p=proba_actions)[0]

            # Epsilon_greedy
            if random.random() < 1 - epsilon:
                action_idx = np.argmax(Q[state_idx])
            else:
                #proba_actions = softmax(np.array(Q[state_idx] / temp))
                #action_idx = np.random.choice(env.action_space.n, 1, p=proba_actions)[0]
                action_idx = np.random.randint(env.action_space.n)

            action = env.A[action_idx]

            # We get our transition <s, a, r, s'>
            next_state, r, done, _ = env.step(action)
            next_state_idx = env.to_idx(*next_state)

            # We update the Q-table with using new knowledge
            old_value = Q[state_idx, action_idx]
            new_value = (r + gamma * np.max(Q[next_state_idx]))
            Q[state_idx, action_idx] = alpha * new_value + (1 - alpha) * old_value
            t, x = env.to_coordinate(state_idx)
            M[t, x, action_idx] += 1
            trajectories[t][x] += 1

            state_idx = next_state_idx

        if episode % int(nb_episodes / 10) == 0:
            v = q_to_v(env, Q)
            # visualisation_value_RM(v, env.T, env.C)
            policy = q_to_policy_RM(env, Q)
            # visualize_policy_RM(policy, env.T, env.C)

            N = 1000
            revenue = average_n_episodes(env, policy, N)
            print("Average reward over {} episodes after {} episodes : {}".format(N, episode, revenue))
            difference_with_optimal_policy = difference_between_policies(policy, P_ref)
            print("Difference with the optimal policy after {} episodes : {}".format(episode, difference_with_optimal_policy))
            diff_with_policy_opt_list.append(difference_with_optimal_policy)
            nb_episodes_list.append(episode)

            # proba_1 = softmax(np.array(Q[50]/temp))
            # proba_2 = softmax(np.array(Q[450] / temp))
            # plt.plot(proba_1), plt.show()
            # plt.plot(proba_2), plt.show()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(alpha_min, alpha * alpha_decay)

    return Q, nb_episodes_list, diff_with_policy_opt_list, M, trajectories


def q_to_v(env, Q_table):
    V = []
    for q in Q_table:
        V.append(max(q))
    return np.array(V)
