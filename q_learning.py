import numpy as np
import random

from visualization_and_metrics import visualisation_value_RM, extract_policy_RM, visualize_policy_RM, \
    average_n_episodes, difference_between_policies


def q_learning(env, alpha, gamma, nb_episodes, epsilon, epsilon_min, epsilon_decay, P_ref):
    # Initialize the Q-table with zeros
    Q = np.zeros([env.nS, env.action_space.n])
    diff_with_policy_opt_list = []
    nb_episodes_list = []

    for episode in range(nb_episodes):
        state = env.set_random_state()

        state_idx = env.to_idx(*state)

        done = False
        while not done:
            # The action associated to s is the one that provides the best Q-value
            # with a proba 1-epsilon and is random with a proba epsilon
            if random.random() < 1 - epsilon:
                action_idx = np.argmax(Q[state_idx])
            else:
                action_idx = np.random.randint(env.action_space.n)
            action = env.A[action_idx]

            # We get our transition <s, a, r, s'>
            next_state, r, done, _ = env.step(action)
            next_state_idx = env.to_idx(*next_state)

            # We update the Q-table with using new knowledge
            old_value = Q[state_idx, action_idx]
            new_value = (r + gamma * np.max(Q[next_state_idx]))
            Q[state_idx, action_idx] = alpha * new_value + (1 - alpha) * old_value

            state_idx = next_state_idx

        if episode % int(nb_episodes / 10) == 0:
            v = q_to_v(env, Q)
            # visualisation_value_RM(v, env.T, env.C)
            policy = extract_policy_RM(env, v, gamma)
            # visualize_policy_RM(policy, env.T, env.C)

            N = 1000
            revenue = average_n_episodes(env, policy, N)
            print("Average reward over {} episodes after {} episodes : {}".format(N, episode, revenue))
            difference_with_optimal_policy = difference_between_policies(policy, P_ref)
            print("Difference with the optimal policy after {} episodes : {}".format(episode,
                                                                                     difference_with_optimal_policy))
            diff_with_policy_opt_list.append(difference_with_optimal_policy)
            nb_episodes_list.append(episode)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, nb_episodes_list, diff_with_policy_opt_list


def q_to_v(env, Q_table):
    V = []
    for q in Q_table:
        V.append(max(q))
    return np.array(V)
