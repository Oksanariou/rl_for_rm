
"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
import matplotlib.pyplot as plt
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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

def extract_policy(U, gamma):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        list_sum = np.zeros(env.nA)
        for a in range(env.nA):
            for p, s_prime, r, _ in env.P[s][a]:
                list_sum[a] += p*(r+gamma*U[s_prime])
        policy[s] = np.argmax(list_sum)
        #policy[s] = 50 + 20 * policy[s]
    return policy

def evaluate_policy(env, policy, gamma, epsilon):
    U = np.zeros(env.nS)
    while True:
        prev_U = np.copy(U)
        for s in range(env.nS):
            a = policy[s]
            U[s] = sum([p * (r + gamma * prev_U[s_]) for p, s_, r, _ in env.P[s][a]])
            #for p, s_prime, r, _ in env.P[s][a]:
                #U[s] += p*(r + gamma*prev_U[s_prime])
        if (np.sum(np.fabs(prev_U - U)) <= epsilon):
            break
    return U

def policy_iteration(env, gamma, max_iter, epsilon):
    policy = np.random.choice(env.nA, env.nS)
    for i in range(max_iter):
        print(i)
        U = evaluate_policy(env, policy, gamma, epsilon)
        new_policy = extract_policy(U, gamma)
        if (np.all(policy == new_policy)):
            print("Converged at " + str(i))
            break
        policy = new_policy
    return policy

def value_iteration(env, gamma, max_iter, epsilon):
    U = np.zeros(env.nS)
    for i in range(max_iter):
        prev_U = np.copy(U)
        for s in range(env.nS):
            list_sum = np.zeros(env.nA)
            for a in range(env.nA):
                for p, s_prime, r, _ in env.P[s][a]:
                    list_sum[a] += p*(r + gamma*prev_U[s_prime])
            U[s] = max(list_sum)
        if (np.sum(np.fabs(prev_U - U)) <= epsilon):
            print("Converged at "+str(i))
            break
    return U

def plot_rev(X, alpha, C):
    R = []
    for p in X:
        R.append(p*C* np.exp(-alpha * ((p / X[0]) - 1)))
    plt.plot(X, R, 'r+')
    plt.title("Revenue as a function of price, alpha = "+str(alpha))
    plt.xlabel("Prices")
    plt.grid()
    return plt.show()

def visualisation_value(V, T, C):
    V = V.reshape(T, C)
    plt.title("Values of the states")
    plt.xlabel('Capacity')
    plt.ylabel('Number of micro-times')
    plt.imshow(V, aspect='auto')
    plt.colorbar()

    return plt.show()

def visualize_policy_RM(P, T, C):
    P = P.reshape(T, C)
    plt.imshow(P, aspect='auto')
    plt.title("Prices coming from the optimal policy")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.colorbar()

    return plt.show()

def visualize_policy_FL(policy):
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


def q_learning(env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay):
    # Initialize the Q-table with zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(nb_episodes):
        s = env.reset()  # Initial observation
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

def q_to_policy_RM(Q):
    policy = []
    for l in Q:
        if l[0] == l[1] == l[2] == l[3] == l[4] == l[5] == l[6] == l[7] == l[8] == l[9] == 0.0:
            policy.append(10)
        else:
            for k in range(0, len(l)):
                if l[k] == max(l):
                    policy.append(k)
                    break
    return np.array(policy)

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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.6
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def compute_q_table(env, network):
    q_table = []
    for s in range(env.observation_space.n):
        state = np.zeros(1, int)
        state[0] = s
        state = np.reshape(state, [1, state_size])
        q_table.append(network.predict(state)[0])
    return q_table

if __name__ == '__main__':
    gamma = 0.99
    #env = gym.make('gym_RM:RM-v0')
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped
    T, C = 500, 50

    v = value_iteration(env, gamma, 100000, 1e-20)

    policy = extract_policy(v, gamma)
    print(policy)
    #visualisation_value(v, T, C)
    #policy = policy_iteration(env, gamma, 100000, 1e-20)

    #visualize_policy_RM(policy, T, C)
    visualize_policy_FL(policy)
    average_n_episodes(env, policy, 100)
    print(average_n_episodes(env, policy, 100))

    #X = [k for k in range(50, 231, 20)]
    #plot_rev(X, 0.66, C)

    #Q-LEARNING - Optimal parameters for the original frozen lake environment

    alpha = 0.05
    nb_episodes, nb_steps = 5000, 500
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.99995
    visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay)
    q_table = q_learning(env, alpha, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay)
    policy = q_to_policy_FL(q_table)
    visualize_policy_FL(policy)
    print(average_n_episodes(env, policy, 100))

    #DQL

    state_size = 1
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    EPISODES = 100

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(100):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    trained_network = agent.model
    Q_table = compute_q_table(env, trained_network)
    policy = q_to_policy_FL(Q_table)
    visualize_policy_FL(policy)
    print(average_n_episodes(env, policy, 100))
