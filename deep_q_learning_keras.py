# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from visualization_and_metrics import visualize_policy_RM


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99

        self.loss = 0.
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(10, input_shape=(self.state_size,), activation='relu', name='state'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='linear', name='action'))
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

        state_batch, act_values_batch = [], []

        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            act_values = self.model.predict(state)
            act_values[0][action] = target

            state_batch.append(state[0])
            act_values_batch.append(act_values[0])

        history = self.model.fit(np.array(state_batch), np.array(act_values_batch), batch_size, epochs=1, verbose=0)
        self.loss = history.history['loss'][0]

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def compute_q_table(env, network):
    state_size = len(env.observation_space)
    shape = [space.n for space in env.observation_space]

    states = [np.asarray((t, x)) for t in range(shape[0]) for x in range(shape[1])]

    return network.predict(np.array(states))


def q_to_policy(env, Q):
    return np.argmax(Q, axis=1)


def visualize_V_RM(Q, T, C):
    P = np.reshape(np.max(Q, axis=1), (T, C))
    plt.imshow(P, aspect='auto')
    plt.title("Expected reward")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.colorbar()
    return plt.show()


if __name__ == "__main__":

    micro_times = 50
    capacity = 5
    actions = tuple(k for k in range(50, 231, 20))
    alpha = 0.6
    lamb = 0.4

    env = gym.make('gym_RM:RM-v0', micro_times=micro_times, capacity=capacity, actions=actions, alpha=alpha, lamb=lamb)

    state_size = len(env.observation_space.spaces)
    action_size = env.action_space.n

    batch_size = 32
    EPISODES = 100

    agent = DQNAgent(state_size, action_size)

    revenues = []

    for e in range(EPISODES):

        Q_table = compute_q_table(env, agent.model)
        visualize_V_RM(Q_table, env.T, env.C)

        policy = q_to_policy(env, Q_table)
        visualize_policy_RM(policy, env.T, env.C)

        state = env.set_random_state()
        state = np.reshape(state, [1, state_size])

        done = False
        revenue = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            revenue += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print("episode: {}/{}, revenue=: {}, loss: {:.2}, e: {:.2}".format(e, EPISODES, revenue, agent.loss,
                                                                           agent.epsilon))

        revenues.append(revenue)
