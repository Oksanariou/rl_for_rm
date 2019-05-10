import gym
import numpy as np
import random
import matplotlib.pyplot as plt

def q_learning(env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay):
    # Initialize the Q-table with zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(nb_episodes):
        s = env.reset()  # Initial observation
        #print("Episode {}/{}, epsilon: {}".format(i, nb_episodes, epsilon))
        for j in range(nb_steps):
            # The action associated to s is the one that provides the best Q-value with a proba 1-epsilon and is random with a proba epsilon
            if random.random() < 1 - epsilon:
                a = np.argmax(Q[s, :])
            else:
                a = np.random.randint(env.action_space.n)
            # We get our transition <s, a, r, s'>
            s_prime, r, d, _ = env.step(a)
            # We update the Q-tqble with using new knowledge
            Q[s, a] = alpha * (r + gamma * np.max(Q[s_prime, :])) + (1 - alpha) * Q[s, a]
            s = s_prime
            if d == True:
                break
            if (epsilon > epsilon_min):
                epsilon *= epsilon_decay
    return Q

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

def q_to_policy(Q):
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

def evaluate_policy(env, policy, n_eval):
    """ Runs n episodes and returns the average of the n total rewards"""
    scores = [run_episode(env, policy) for _ in range(n_eval)]
    return np.mean(scores)

def running_q_learning_n_times(n, n_eval, env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay):
    scores = []
    for k in range(n):
        print(k)
        q_table = q_learning(env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay)
        policy = q_to_policy(q_table)
        score = evaluate_policy(env, policy, n_eval)
        scores.append(score)
    return np.mean(scores)

def visualize_policy(policy):
    visu = ''
    for k in range(len(policy)):
        if k > 0 and k%4 == 0:
            visu += '\n'
        if k == 5 or k == 7 or k == 11 or k == 12 or k == 15:
            visu+='H'
        elif int(policy[k]) == 0:
            visu += 'L'
        elif int(policy[k]) == 1:
            visu += 'D'
        elif int(policy[k]) == 2:
            visu += 'R'
        elif int(policy[k]) == 3:
            visu += 'U'
    print(visu)

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    n, n_eval = 100, 10000
    alpha, gamma = 0.05, 0.99
    nb_episodes, nb_steps = 5000, 100
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.97

    q_table = q_learning(env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay)
    #print(running_q_learning_n_times(n, n_eval, env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay))
    print(q_table)
    policy = q_to_policy(q_table)
    visualize_policy(policy)
    visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    print(running_q_learning_n_times(n, n_eval, env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min,epsilon_decay))
    #q_learning(env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay)