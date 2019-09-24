import numpy as np
import matplotlib.pyplot as plt
import random
import gym
from tqdm import tqdm
from scipy import stats


def visualize_policy_FL(policy):
    visu = ''
    for k in range(len(policy)):
        if k > 0 and k % 4 == 0:
            visu += '\n'
        if k == 5 or k == 7 or k == 11 or k == 12 or k == 15:
            visu += 'H'
        elif int(policy[k]) == 0:
            visu += 'L'
        elif int(policy[k]) == 1:
            visu += 'D'
        elif int(policy[k]) == 2:
            visu += 'R'
        elif int(policy[k]) == 3:
            visu += 'U'
    print(visu)


def run_episode_FL(env, policy):
    """ Runs an episode and returns the total reward """
    state = env.reset()
    total_reward = 0
    while True:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def run_episode(env, policy, agent=None, epsilon=0.0):
    """ Runs an episode and returns the total reward """
    state = env.reset()
    total_reward = 0
    bookings = np.zeros(env.nA)
    while True:
        # t, x = env.to_coordinate(state)
        state_idx = env.to_idx(*state) if type(env.observation_space) == gym.spaces.tuple.Tuple else state #env.to_idx(state[0], state[1], state[2])
        if np.random.rand() <= epsilon:
            action_idx = random.randrange(env.action_space.n)
            action = env.A[action_idx] if type(env.observation_space) == gym.spaces.tuple.Tuple else action_idx

        else:
            if agent is not None:
                action_idx, _ = agent.predict(state)
                action = action_idx
            else:
                action_idx = policy[state_idx]
                action = action_idx
        state, reward, done, _ = env.step(action_idx)
        if reward != 0:
            bookings[action_idx] += 1
        total_reward += reward
        if done:
            break
    return total_reward, bookings


def run_episode_collaboration(env, policy):
    state_idx = env.reset()
    total_reward = 0
    while True:
        action = policy[state_idx]
        state_idx, reward, done, _ = env.step(action)
        total_reward += reward[0] + reward[1]
        if done[0] and done[1]:
            break
    return total_reward


def run_episode_collaboration_global_policy(global_env, global_policy, individual2D_env):
    state_idx = global_env.reset()
    total_reward = 0
    reward_1, reward_2 = 0, 0
    bookings_flight1 = np.zeros(individual2D_env.nA)
    bookings_flight2 = np.zeros(individual2D_env.nA)
    while True:
        action = global_policy[state_idx]
        state_idx, reward, done, _ = global_env.step(action)
        action1, action2 = global_env.A[int(action)][0], global_env.A[int(action)][1]

        if reward[0] != 0:
            bookings_flight1[individual2D_env.A.index(int(action1))] += 1
        if reward[1] != 0:
            bookings_flight2[individual2D_env.A.index(int(action2))] += 1

        total_reward += reward[0] + reward[1]
        reward_1 += reward[0]
        reward_2 += reward[1]
        if done[0] and done[1]:
            break
    return [reward_1, reward_2], [bookings_flight1, bookings_flight2]


def run_episode_collaboration_individual_2D_VS_3D_policies(global_env, policy_2D, policy_3D, individual2D_env):
    state_idx = global_env.reset()
    total_reward = 0
    reward_1, reward_2 = 0, 0
    bookings_flight1 = np.zeros(individual2D_env.nA)
    bookings_flight2 = np.zeros(individual2D_env.nA)
    while True:
        t, x1, x2 = global_env.to_coordinate(state_idx)
        state_idx1 = individual2D_env.to_idx(t, x1)
        action1 = policy_2D[state_idx1]
        action2 = policy_3D[state_idx]
        action_tuple = (int(action1), int(action2))
        action = global_env.A.index(action_tuple)

        state_idx, reward, done, _ = global_env.step(action)

        if reward[0] != 0:
            bookings_flight1[individual2D_env.A.index(int(action1))] += 1
        if reward[1] != 0:
            bookings_flight2[individual2D_env.A.index(int(action2))] += 1

        total_reward += reward[0] + reward[1]
        reward_1 += reward[0]
        reward_2 += reward[1]
        if done[0] and done[1]:
            break
    return [reward_1, reward_2], [bookings_flight1, bookings_flight2]


def run_episode_collaboration_individual_3D_policies(global_env, policy1, policy2, individual3D_env):
    state_idx = global_env.reset()
    total_reward = 0
    reward_1, reward_2 = 0, 0
    bookings_flight1 = np.zeros(individual3D_env.nA)
    bookings_flight2 = np.zeros(individual3D_env.nA)
    while True:
        action1 = policy1[state_idx]
        action2 = policy2[state_idx]
        action_tuple = (individual3D_env.A[int(action1)], individual3D_env.A[int(action2)])
        action = global_env.A.index(action_tuple)

        state_idx, reward, done, _ = global_env.step(action)

        if reward[0] != 0:
            bookings_flight1[action1] += 1
        if reward[1] != 0:
            bookings_flight2[action2] += 1

        total_reward += reward[0] + reward[1]
        reward_1 += reward[0]
        reward_2 += reward[1]
        if done[0] and done[1]:
            break
    return [reward_1, reward_2], [bookings_flight1, bookings_flight2]


def run_episode_collaboration_individual_policy_on_n_flights_env(n_flights_env, individual2D_env, individual_policy):
    state_idx = n_flights_env.reset()
    individual_rewards = np.zeros((n_flights_env.number_of_flights))
    individual_bookings = np.zeros((n_flights_env.number_of_flights, individual2D_env.nA))
    while True:
        t, x = n_flights_env.to_coordinate(state_idx)
        actions = []
        for k in range(n_flights_env.number_of_flights):
            state_idx = individual2D_env.to_idx(t, x[k])
            action = individual_policy[state_idx]
            actions.append(action)
        actions_tuple = tuple(int(a) for a in actions)
        actions_idx = n_flights_env.A.index(actions_tuple)

        state_idx, rewards, dones, _ = n_flights_env.step(actions_idx)

        for k in range(len(rewards)):
            individual_rewards[k] += rewards[k]
            if rewards[k] != 0:
                individual_bookings[k][individual2D_env.A.index(int(actions[k]))] += 1

        if np.all(dones):
            break
    return individual_rewards, individual_bookings


def run_episode_collaboration_individual_2D_policies(global_env, individual2D_env, policy1, policy2):
    state_idx = global_env.reset()
    total_reward = 0
    reward_1, reward_2 = 0, 0
    bookings_flight1 = np.zeros(individual2D_env.nA)
    bookings_flight2 = np.zeros(individual2D_env.nA)
    while True:
        t, x1, x2 = global_env.to_coordinate(state_idx)
        state_idx1 = individual2D_env.to_idx(t, x1)
        state_idx2 = individual2D_env.to_idx(t, x2)
        action1 = policy1[state_idx1]
        action2 = policy2[state_idx2]
        action_tuple = (individual2D_env.A[int(action1)], individual2D_env.A[int(action2)])
        action = global_env.A.index(action_tuple)

        state_idx, reward, done, _ = global_env.step(action)

        if reward[0] != 0:
            bookings_flight1[action1] += 1
        if reward[1] != 0:
            bookings_flight2[action2] += 1

        total_reward += reward[0] + reward[1]
        reward_1 += reward[0]
        reward_2 += reward[1]
        if done[0] and done[1]:
            break
    return [reward_1, reward_2], [bookings_flight1, bookings_flight2]


def average_n_episodes_FL(env, policy, n_eval):
    """ Runs n episodes and returns the average of the n total rewards"""
    scores = [run_episode_FL(env, policy) for _ in range(n_eval)]
    return np.mean(scores)


def average_n_episodes(env, policy, n_eval, agent=None, epsilon=0.0):
    """ Runs n episodes and returns the average of the n total rewards"""
    scores = [run_episode(env, policy, agent, epsilon) for _ in range(n_eval)]
    scores = np.array(scores)
    revenue = np.mean(scores[:, 0])
    bookings = np.mean(scores[:, 1], axis=0)
    return revenue, bookings


def average_n_episodes_collaboration_global_policy(global_env, global_policy, individual2D_env, n_eval):
    revenues_1, revenues_2 = [], []
    bookings_1, bookings_2 = [], []
    for k in range(n_eval):
        revenues, bookings = run_episode_collaboration_global_policy(global_env, global_policy, individual2D_env)
        revenues_1.append(revenues[0])
        revenues_2.append(revenues[1])
        bookings_1.append(bookings[0])
        bookings_2.append(bookings[1])
    return [np.mean(revenues_1), np.mean(revenues_2)], [np.mean(bookings_1, axis=0), np.mean(bookings_2, axis=0)]


def average_n_episodes_collaboration_individual_2D_VS_3D_policies(global_env, policy_3D, individual_2D_env, policy_2D,
                                                                  n_eval):
    revenues_1, revenues_2 = [], []
    bookings_1, bookings_2 = [], []
    for k in range(n_eval):
        revenues, bookings = run_episode_collaboration_individual_2D_VS_3D_policies(global_env, policy_3D,
                                                                                    individual_2D_env, policy_2D)
        revenues_1.append(revenues[0])
        revenues_2.append(revenues[1])
        bookings_1.append(bookings[0])
        bookings_2.append(bookings[1])
    return [np.mean(revenues_1), np.mean(revenues_2)], [np.mean(bookings_1, axis=0), np.mean(bookings_2, axis=0)]


def average_n_episodes_collaboration_individual_3D_policies(global_env, policy1, policy2, individual3D_env, n_eval):
    revenues_1, revenues_2 = [], []
    bookings_1, bookings_2 = [], []
    for k in range(n_eval):
        revenues, bookings = run_episode_collaboration_individual_3D_policies(global_env, policy1, policy2,
                                                                              individual3D_env)
        revenues_1.append(revenues[0])
        revenues_2.append(revenues[1])
        bookings_1.append(bookings[0])
        bookings_2.append(bookings[1])
    return [np.mean(revenues_1), np.mean(revenues_2)], [np.mean(bookings_1, axis=0), np.mean(bookings_2, axis=0)]


def average_n_episodes_collaboration_individual_2D_policies(global_env, individual2D_env, policy1, policy2, n_eval):
    revenues_1, revenues_2 = [], []
    bookings_1, bookings_2 = [], []
    for k in range(n_eval):
        revenues, bookings = run_episode_collaboration_individual_2D_policies(global_env, individual2D_env, policy1,
                                                                              policy2)
        revenues_1.append(revenues[0])
        revenues_2.append(revenues[1])
        bookings_1.append(bookings[0])
        bookings_2.append(bookings[1])
    return [np.mean(revenues_1), np.mean(revenues_2)], [np.mean(bookings_1, axis=0), np.mean(bookings_2, axis=0)]


def average_n_episodes_collaboration_individual_policy_on_n_flights_env(n_flights_env, individual2D_env,
                                                                        individual_policy, n_eval):
    all_revenues = [[] for k in range(n_flights_env.number_of_flights)]
    all_bookings = [[] for k in range(n_flights_env.number_of_flights)]
    for k in range(n_eval):
        revenues, bookings = run_episode_collaboration_individual_policy_on_n_flights_env(n_flights_env,
                                                                                          individual2D_env,
                                                                                          individual_policy)
        for i in range(n_flights_env.number_of_flights):
            all_revenues[i].append(revenues)
            all_bookings[i].append(bookings)
    return np.array([np.mean(revenue) for revenue in all_revenues]), np.array([np.mean(booking, axis=0) for booking in all_bookings])


def q_to_policy_FL(Q):
    policy = []
    for l in Q:
        if l[0] == l[1] == l[2] == l[3] == 0.0:
            policy.append(0)
        else:
            for k in range(0, len(l)):
                if l[k] == max(l):
                    policy.append(k)
                    break
    return policy


def visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay):
    X = [k for k in range(nb_episodes)]
    Y = []
    for k in range(nb_episodes):

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        Y.append(epsilon)
    plt.plot(X, Y, 'b')
    plt.title("Decaying alpha over the number of episodes")
    plt.xlabel("Number of episodes")
    plt.ylabel("Alpha")
    plt.grid()
    return plt.show()


def visualisation_value_RM(V, T, C):
    V = V.reshape(T, C)
    plt.title("Values of the states")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.imshow(V, aspect='auto')
    plt.colorbar()
    return plt.show()


def extract_policy_RM(env, U, gamma):
    policy = np.zeros(env.nS)
    for state_idx in range(env.nS):
        state = env.to_coordinate(state_idx)
        # if U[state_idx] == 0.00000000e+00:
        # policy[state_idx] == 0
        # else:
        list_sum = np.zeros(env.nA)
        for idx_a in range(env.nA):
            action = env.A[idx_a]
            for p, s_prime, r, _ in env.P[state][action]:
                list_sum[idx_a] += p * (r + gamma * U[env.to_idx(*s_prime)])
        idx_best_a = np.argmax(list_sum)
        policy[state_idx] = env.A[idx_best_a]
    return policy


def extract_policy_RM_discrete(env, U, gamma, P={}):
    if P == {}:
        P = env.P
    policy = np.zeros(env.nS)
    for state_idx in range(env.nS):
        list_sum = np.zeros(env.nA)
        for idx_a in range(env.nA):
            for p, s_prime, r, _ in env.P[state_idx][idx_a]:
                list_sum[idx_a] += p * (r + gamma * U[s_prime])
        idx_best_a = np.argmax(list_sum)
        policy[state_idx] = env.A[idx_best_a]
    return policy


def extract_policy_RM_discrete_collaboration(env, U, gamma, P={}):
    if P == {}:
        P = env.P
    policy = np.zeros(env.nS)
    for state_idx in range(env.nS):
        list_sum = np.zeros(env.nA)
        for idx_a in range(env.nA):
            for p, s_prime, r, _ in env.P[state_idx][idx_a]:
                list_sum[idx_a] += p * (r[0] + r[1] + gamma * U[s_prime])
        idx_best_a = np.argmax(list_sum)
        policy[state_idx] = idx_best_a
    return policy


def visualize_policy_RM(P, T, C):
    P = np.reshape(P, (T, C))
    P = P[:T - 1, :C - 1]
    plt.imshow(P, aspect='auto')
    plt.title("Prices coming from the optimal policy")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.colorbar()
    return plt.show()


def q_to_policy_RM(env, Q):
    policy = []
    for l in Q:
        idx_action = np.argmax(l)
        policy.append(idx_action)
    policy = np.array(policy).reshape(env.T, env.C)
    policy[:, -1] = 0.
    policy[-1] = 0.
    return np.array(policy.reshape(env.T * env.C))


def difference_between_policies(p1, p2):
    Error = 0
    for k in range(len(p1)):
        Error += abs(p1[k] - p2[k])
    return Error


def plot_evolution_difference_between_policies(X, P):
    plt.plot(X, P)
    plt.title("Evolution of the difference with the optimal policy")
    plt.xlabel("Number of episodes")
    plt.ylabel("Difference with the optimal policy")
    plt.grid()
    return plt.show()


def v_to_q(env, V, gamma):
    Q = np.zeros([env.nS, env.action_space.n])
    for state_idx in range(env.nS):
        for action_idx in range(env.nA):
            state = env.to_coordinate(state_idx)
            action = env.A[action_idx]
            for p, s_, r, _ in env.P[state][action]:
                next_state_idx = env.to_idx(*s_)
                Q[state_idx][action_idx] += p * (r + gamma * V[next_state_idx])
    return Q


def reshape_matrix_of_visits(M, env):
    X = []
    Y = []
    Z = []
    values = []
    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            for z in range(M.shape[2]):
                if M[x][y][z] != 0:
                    X.append(x)
                    Y.append(y)
                    Z.append(env.A[z])
                    values.append(M[x][y][z])
    return X, Y, Z, values


def from_microtimes_to_DCP(policy_microtimes, env_microtimes, env_DCP, way):
    policy_microtimes = policy_microtimes.reshape(env_microtimes.T, env_microtimes.C)
    policy_DCP = np.zeros((env_DCP.T, env_DCP.C), int)
    for t in range(env_DCP.T):
        for x in range(env_DCP.C):
            if way == "median":
                policy_DCP[t, x] = np.median(policy_microtimes[t * env_DCP.M:t * env_DCP.M + env_DCP.M, x:x + 1])
            elif way == "mean":
                policy_DCP[t, x] = np.mean(policy_microtimes[t * env_DCP.M:t * env_DCP.M + env_DCP.M, x:x + 1])
            else:
                print("Specify median or mean")
    return policy_DCP


def average_and_std_deviation_n_episodes(env, policy, nb_runs, epsilon=0.):
    scores = [run_episode(env, policy, epsilon) for _ in range(nb_runs)]
    return np.mean(scores), np.sqrt(np.var(scores))


def plot_average_and_std_deviatione(list_of_nb_episodes, env, policy, epsilon=0.):
    means, std_deviations = [], []

    for n in tqdm(list_of_nb_episodes):
        mean, std_deviation = average_and_std_deviation_n_episodes(env, policy, n, epsilon)
        means.append(mean)
        std_deviations.append(std_deviation)

    plt.figure(1)
    plt.plot(list_of_nb_episodes, means)
    plt.title("Evolution of the average revenue")
    plt.xlabel("Nb of runs used to compute the average revenue")
    plt.ylabel("Average revenue")
    plt.show()

    plt.figure(2)
    plt.plot(list_of_nb_episodes, std_deviations)
    plt.title("Evolution of the standard deviation")
    plt.xlabel("Nb of runs used to compute the standard deviation")
    plt.ylabel("Standard deviation")
    plt.show()
