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


def run_episode(env, policy):
    """ Runs an episode and returns the total reward """
    obs = env.reset()
    total_reward = 0
    while True:
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += reward
        if done:
            break
    return total_reward


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
    plt.imshow(V, aspect = 'auto')
    plt.colorbar()
    return plt.show()

def extract_policy_RM(env, U, gamma):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        if U[s] == 0.00000000e+00:
            policy[s] == 0
        else:
            list_sum = np.zeros(env.nA)
            for a in range(env.nA):
                for p, s_prime, r, _ in env.P[s][a]:
                    list_sum[a] += p*(r+gamma*U[s_prime])
            policy[s] = np.argmax(list_sum)
            #policy[s] = 50 + 20*policy[s]
    return policy

def visualize_policy_RM(P, T, C):
    P = P.reshape(T, C)
    plt.imshow(P, aspect='auto')
    plt.title("Prices coming from the optimal policy")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.colorbar()
    return plt.show()

def q_to_policy_RM(Q):
    policy = []
    for l in Q:
        #if l[0] == l[1] == l[2] == l[3] == l[4] == l[5] == l[6] == l[7] == l[8] == l[9] == 0.0:
            #policy.append(10)
        #else:
            for k in range(0, len(l)):
                if l[k] == max(l):
                    policy.append(k)
                    break
    #for s in range(len(policy)):
        #policy[s] = 50 + 20*policy[s]

    return np.array(policy)