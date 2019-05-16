# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation='relu'))
        model.add(Dense(10, activation='relu'))
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

def q_to_policy(Q):
    policy = []
    for l in Q:
        if l[0] == l[1] == l[2] == 0.0:
            policy.append(5)
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

def visualisation_policy(P, T, C):
    P = P.reshape(T, C)
    plt.imshow(P)
    plt.title("Prices coming from the optimal policy")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.colorbar()

    return plt.show()

if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    #state_size = env.observation_space.shape[0]
    #env = gym.make('FrozenLake-v0')
    env = gym.make('gym_RM:RM-v0')
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
    policy = np.array(q_to_policy(Q_table))
    T, C = 150, 50
    visualisation_policy(policy, T, C)
