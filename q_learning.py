import numpy as np
import random

def q_learning(env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay):
    # Initialize the Q-table with zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    rList = []
    for i in range(nb_episodes):
        print(i)
        rAll = 0
        s = env.reset()  # Initial observation
        for j in range(nb_steps):
            # The action associated to s is the one that provides the best Q-value with a proba 1-epsilon and is random with a proba epsilon
            if random.random() < 1 - epsilon:
                a = np.argmax(Q[s, :])
            else:
                a = np.random.randint(env.action_space.n)
            # We get our transition <s, a, r, s'>
            s_prime, r, d, _ = env.step(a)
            rAll += r
            # We update the Q-table with using new knowledge
            Q[s, a] = alpha * (r + gamma * np.max(Q[s_prime, :])) + (1 - alpha) * Q[s, a]
            s = s_prime
            if d == True:
                break
        if (epsilon > epsilon_min):
            epsilon *= epsilon_decay
        rList.append(rAll)
    return Q, rList

def q_to_v(env, Q_table):
    V = []
    for q in Q_table:
        V.append(max(q))
    return np.array(V)
