import numpy as np
import random

from visualization_and_metrics import average_n_episodes_collaboration_individual_3D_policies, \
    average_n_episodes_collaboration_individual_2D_policies, average_n_episodes_collaboration_global_policy


def q_learning_global(global_env, individual2D_env, alpha, alpha_min, alpha_decay, gamma,
                               nb_episodes, epsilon,
                               epsilon_min, epsilon_decay):
    # Initialize the Q-table with zeros
    Q = np.zeros([global_env.nS, global_env.action_space.n])
    revenue1, revenue2, bookings1, bookings2, episodes = [], [], [], [], []

    for episode in range(nb_episodes):
        state = global_env.reset()
        done1, done2 = False, False

        while not (done1 and done2):

            # Epsilon_greedy
            if random.random() < 1 - epsilon:
                action_idx = np.argmax(Q[state])
            else:
                action_idx = np.random.randint(global_env.action_space.n)

            # We get our transition <s, a, r, s'>
            next_state, r, done, _ = global_env.step(action_idx)
            done1, done2 = done[0], done[1]

            # We update the Q-table with using new knowledge
            old_value = Q[state, action_idx]
            new_value = r[0] + r[1] + gamma * np.max(Q[next_state])
            Q[state, action_idx] = alpha * new_value + (
                        1 - alpha) * old_value

            state = next_state

        if episode % int(nb_episodes / 10) == 0:
            policy = q_to_policy_3D(global_env, Q)

            revenues, bookings = average_n_episodes_collaboration_global_policy(global_env, policy, individual2D_env, 10000)

            revenue1.append(revenues[0])
            revenue2.append(revenues[1])
            bookings1.append(bookings[0])
            bookings2.append(bookings[1])
            episodes.append(episode)
            print("Average reward after {} episodes : {}".format(episode, revenues))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(alpha_min, alpha * alpha_decay)

    return Q, [revenue1, revenue2], [bookings1, bookings2], episodes


def q_learning_collaboration3D(global_env, individual_env1, individual_env2, alpha, alpha_min, alpha_decay, beta,
                               beta_min, beta_decay, gamma,
                               nb_episodes, epsilon,
                               epsilon_min, epsilon_decay, fully_collaborative):
    # Initialize the Q-table with zeros
    Q1 = np.zeros([global_env.nS, individual_env1.action_space.n])
    Q2 = np.zeros([global_env.nS, individual_env2.action_space.n])
    revenue1, revenue2, bookings1, bookings2, episodes = [], [], [], [], []

    for episode in range(nb_episodes):
        state = global_env.reset()
        done1, done2 = False, False

        while not (done1 and done2):

            # Epsilon_greedy
            if random.random() < 1 - epsilon:
                action_idx1 = np.argmax(Q1[state])
                action_idx2 = np.argmax(Q2[state])
            else:
                action_idx1 = np.random.randint(individual_env1.action_space.n)
                action_idx2 = np.random.randint(individual_env2.action_space.n)

            action1 = individual_env1.A[action_idx1]
            action2 = individual_env2.A[action_idx2]
            action = global_env.A.index((action1, action2))

            # We get our transition <s, a, r, s'>
            next_state, r, done, _ = global_env.step(action)
            reward1 = r[0] + r[1] if fully_collaborative else r[0]
            reward2 = r[0] + r[1] if fully_collaborative else r[1]
            done1, done2 = done[0], done[1]

            # We update the Q-table with using new knowledge
            old_value1 = Q1[state, action_idx1]
            new_value1 = reward1 + gamma * np.max(Q1[next_state])
            Q1[state, action_idx1] = alpha * new_value1 + (
                        1 - alpha) * old_value1 if new_value1 > old_value1 else beta * new_value1 + (
                        1 - beta) * old_value1

            old_value2 = Q2[state, action_idx2]
            new_value2 = (reward2 + gamma * np.max(Q2[next_state]))
            Q2[state, action_idx2] = alpha * new_value2 + (
                        1 - alpha) * old_value2 if new_value2 > old_value2 else beta * new_value2 + (
                        1 - beta) * old_value2

            state = next_state

        if episode % int(nb_episodes / 10) == 0:
            policy1 = q_to_policy_3D(individual_env1, Q1)
            policy2 = q_to_policy_3D(individual_env2, Q2)

            revenues, bookings = average_n_episodes_collaboration_individual_3D_policies(global_env, policy1, policy2,
                                                                                         individual_env1, 10000)

            revenue1.append(revenues[0])
            revenue2.append(revenues[1])
            bookings1.append(bookings[0])
            bookings2.append(bookings[1])
            episodes.append(episode)
            print("Average reward after {} episodes : {}".format(episode, revenues))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(alpha_min, alpha * alpha_decay)
        beta = max(beta_min, beta * beta_decay)

    return [Q1, Q2], [revenue1, revenue2], [bookings1, bookings2], episodes


def q_learning_collaboration2D(global_env, individual_env1, individual_env2, alpha, alpha_min, alpha_decay, beta,
                               beta_min, beta_decay, gamma,
                               nb_episodes, epsilon,
                               epsilon_min, epsilon_decay, fully_collaborative):
    # Initialize the Q-table with zeros
    Q1 = np.zeros([individual_env1.nS, individual_env1.action_space.n])
    Q2 = np.zeros([individual_env1.nS, individual_env2.action_space.n])
    revenue1, revenue2, bookings1, bookings2, episodes = [], [], [], [], []

    for episode in range(nb_episodes):
        state = global_env.reset()
        t, x1, x2 = global_env.to_coordinate(state)
        state1, state2 = individual_env1.to_idx(t, x1), individual_env2.to_idx(t, x2)

        done1, done2 = False, False

        while not (done1 and done2):

            # Epsilon_greedy
            if random.random() < 1 - epsilon:
                action_idx1 = np.argmax(Q1[state1])
                action_idx2 = np.argmax(Q2[state2])
            else:
                action_idx1 = np.random.randint(individual_env1.action_space.n)
                action_idx2 = np.random.randint(individual_env2.action_space.n)

            action1 = individual_env1.A[action_idx1]
            action2 = individual_env2.A[action_idx2]
            action = global_env.A.index((action1, action2))

            # We get our transition <s, a, r, s'>
            next_state, r, done, _ = global_env.step(action)
            new_t, new_x1, new_x2 = global_env.to_coordinate(next_state)
            next_state1, next_state2 = individual_env1.to_idx(new_t, new_x1), individual_env2.to_idx(new_t, new_x2)

            reward1 = r[0] + r[1] if fully_collaborative else r[0]
            reward2 = r[0] + r[1] if fully_collaborative else r[1]
            done1, done2 = done[0], done[1]

            # We update the Q-table with using new knowledge

            old_value1 = Q1[state1, action_idx1]
            new_value1 = reward1 + gamma * np.max(Q1[next_state1])
            Q1[state1, action_idx1] = alpha * new_value1 + (
                    1 - alpha) * old_value1 if new_value1 > old_value1 else beta * new_value1 + (
                    1 - beta) * old_value1

            old_value2 = Q2[state2, action_idx2]
            new_value2 = (reward2 + gamma * np.max(Q2[next_state2]))
            Q2[state2, action_idx2] = alpha * new_value2 + (
                    1 - alpha) * old_value2 if new_value2 > old_value2 else beta * new_value2 + (
                    1 - beta) * old_value2

            state1 = next_state1
            state2 = next_state2

        if episode % int(nb_episodes / 10) == 0:
            policy1 = q_to_policy_2D(individual_env1, Q1)
            policy2 = q_to_policy_2D(individual_env2, Q2)

            revenues, bookings = average_n_episodes_collaboration_individual_2D_policies(global_env, individual_env1,
                                                                                         policy1, policy2, 10000)

            revenue1.append(revenues[0])
            revenue2.append(revenues[1])
            bookings1.append(bookings[0])
            bookings2.append(bookings[1])
            episodes.append(episode)
            print("Average reward after {} episodes : {}".format(episode, revenues))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(alpha_min, alpha * alpha_decay)
        beta = max(beta_min, beta * beta_decay)

    return [Q1, Q2], [revenue1, revenue2], [bookings1, bookings2], episodes

def q_learning_collaboration2D_shared_Q_table(global_env, individual_env1, individual_env2, alpha, alpha_min, alpha_decay, beta,
                               beta_min, beta_decay, gamma,
                               nb_episodes, epsilon,
                               epsilon_min, epsilon_decay, fully_collaborative):
    # Initialize the Q-table with zeros
    Q = np.zeros([individual_env1.nS, individual_env1.action_space.n])
    revenue1, revenue2, bookings1, bookings2, episodes = [], [], [], [], []

    for episode in range(nb_episodes):
        state = global_env.reset()
        t, x1, x2 = global_env.to_coordinate(state)
        state1, state2 = individual_env1.to_idx(t, x1), individual_env2.to_idx(t, x2)

        done1, done2 = False, False

        while not (done1 and done2):

            # Epsilon_greedy
            if random.random() < 1 - epsilon:
                action_idx1 = np.argmax(Q[state1])
                action_idx2 = np.argmax(Q[state2])
            else:
                action_idx1 = np.random.randint(individual_env1.action_space.n)
                action_idx2 = np.random.randint(individual_env2.action_space.n)

            action1 = individual_env1.A[action_idx1]
            action2 = individual_env2.A[action_idx2]
            action = global_env.A.index((action1, action2))

            # We get our transition <s, a, r, s'>
            next_state, r, done, _ = global_env.step(action)
            new_t, new_x1, new_x2 = global_env.to_coordinate(next_state)
            next_state1, next_state2 = individual_env1.to_idx(new_t, new_x1), individual_env2.to_idx(new_t, new_x2)

            reward1 = r[0] + r[1] if fully_collaborative else r[0]
            reward2 = r[0] + r[1] if fully_collaborative else r[1]
            done1, done2 = done[0], done[1]

            # We update the Q-table with using new knowledge

            old_value1 = Q[state1, action_idx1]
            new_value1 = reward1 + gamma * np.max(Q[next_state1])
            Q[state1, action_idx1] = alpha * new_value1 + (
                    1 - alpha) * old_value1 if new_value1 > old_value1 else beta * new_value1 + (
                    1 - beta) * old_value1

            old_value2 = Q[state2, action_idx2]
            new_value2 = (reward2 + gamma * np.max(Q[next_state2]))
            Q[state2, action_idx2] = alpha * new_value2 + (
                    1 - alpha) * old_value2 if new_value2 > old_value2 else beta * new_value2 + (
                    1 - beta) * old_value2

            state1 = next_state1
            state2 = next_state2

        if episode % int(nb_episodes / 10) == 0:
            policy1 = q_to_policy_2D(individual_env1, Q)
            policy2 = q_to_policy_2D(individual_env2, Q)

            revenues, bookings = average_n_episodes_collaboration_individual_2D_policies(global_env, individual_env1,
                                                                                         policy1, policy2, 10000)

            revenue1.append(revenues[0])
            revenue2.append(revenues[1])
            bookings1.append(bookings[0])
            bookings2.append(bookings[1])
            episodes.append(episode)
            print("Average reward after {} episodes : {}".format(episode, revenues))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        alpha = max(alpha_min, alpha * alpha_decay)
        beta = max(beta_min, beta * beta_decay)

    return [Q], [revenue1, revenue2], [bookings1, bookings2], episodes


def q_to_v(Q_table):
    V = []
    for q in Q_table:
        V.append(max(q))
    return np.array(V)


def q_to_policy_3D(individual_env, Q):
    policy = []
    for l in Q:
        idx_action = np.argmax(l)
        policy.append(idx_action)
    policy = np.array(policy).reshape(individual_env.T, individual_env.C1, individual_env.C2)
    policy[:, -1][:, -1] = 0.
    policy[-1] = 0.
    return np.array(policy.reshape(individual_env.nS))


def q_to_policy_2D(individual_env, Q):
    policy = []
    for l in Q:
        idx_action = np.argmax(l)
        policy.append(idx_action)
    policy = np.array(policy).reshape(individual_env.T, individual_env.C)
    policy[:, -1] = 0.
    policy[-1] = 0.
    return np.array(policy.reshape(individual_env.nS))
