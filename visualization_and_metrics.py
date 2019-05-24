import numpy as np
import matplotlib.pyplot as plt


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

def run_episode(env, policy):
    """ Runs an episode and returns the total reward """
    state = env.reset()
    total_reward = 0
    while True:
        state_idx = env.to_idx(*state)
        action = policy[state_idx]
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def average_n_episodes_FL(env, policy, n_eval):
    """ Runs n episodes and returns the average of the n total rewards"""
    scores = [run_episode_FL(env, policy) for _ in range(n_eval)]
    return np.mean(scores)

def average_n_episodes(env, policy, n_eval):
    """ Runs n episodes and returns the average of the n total rewards"""
    scores = [run_episode(env, policy) for _ in range(n_eval)]
    return np.mean(scores)


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
    plt.title("Decaying epsilon over the number of episodes")
    plt.xlabel("Number of episodes")
    plt.ylabel("Epsilon")
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


def visualize_policy_RM(P, T, C):
    P = np.reshape(P, (T, C))
    P = P[:T - 1, :C - 1]
    plt.imshow(P, aspect='auto')
    plt.title("Prices coming from the optimal policy")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.colorbar()
    return plt.show()


def q_to_policy_RM(Q):
    policy = []
    for l in Q:
        # if l[0] == l[1] == l[2] == l[3] == l[4] == l[5] == l[6] == l[7] == l[8] == l[9] == 0.0:
        # policy.append(10)
        # else:
        for k in range(0, len(l)):
            if l[k] == max(l):
                policy.append(k)
                break
    # for s in range(len(policy)):
    # policy[s] = 50 + 20*policy[s]

    return np.array(policy)


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
