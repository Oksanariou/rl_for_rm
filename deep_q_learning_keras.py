# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam

from dynamic_programming import dynamic_programming
from visualization_and_metrics import visualize_policy_RM, average_n_episodes


class DQNAgent:
    def __init__(self, input_size, action_size, target_model_update=10):
        self.input_size = input_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99

        self.replay_count = 0
        self.target_model_update = target_model_update
        self.loss = 0.
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(50, input_shape=(self.input_size,), activation='relu', name='state'))
        model.add(BatchNormalization())
        # model.add(Dropout(rate=0.2))
        model.add(Dense(50, activation='relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(rate=0.2))
        model.add(Dense(self.action_size, activation='linear', name='action'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def set_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_target_q_value(self, next_state):

        next_q_values = self.model.predict(next_state)
        action = np.argmax(next_q_values[0])

        q_value = self.target_model.predict(next_state)[0][action]

        return reward + self.gamma * q_value

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        state_batch, q_values_batch = [], []

        for state, action, reward, next_state, done in minibatch:

            q_value = self.get_target_q_value(next_state)

            q_values = self.model.predict(state)
            q_values[0][action] = reward if done else q_value

            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        history = self.model.fit(np.array(state_batch), np.array(q_values_batch), batch_size, epochs=1, verbose=0)
        self.loss = history.history['loss'][0]

        self.update_epsilon()

        self.replay_count += 1
        if self.replay_count % self.target_model_update == 0:
            self.set_target()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def init(self, X, Y, epochs):
        history = self.model.fit(X, Y, batch_size, epochs=epochs, verbose=0)
        self.loss = history.history['loss'][0]


def compute_q_table(env, model):
    state_size = len(env.observation_space)
    shape = [space.n for space in env.observation_space]

    states = [np.asarray((t, x)) for t in range(shape[0]) for x in range(shape[1])]

    return model.predict(np.array(states))


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


def init_with_V(agent, env, gamma):
    shape = [space.n for space in env.observation_space]
    states = [(t, x) for t in range(shape[0]) for x in range(shape[1])]

    true_V, true_policy = dynamic_programming(env.T, env.C, env.alpha, env.lamb, env.A)

    true_Q_table = []
    for state in states:
        q_values = []
        for action in env.A:
            expected_discounted_reward = 0.
            for proba, next_state, reward, done in env.P[state][action]:
                expected_discounted_reward += proba * (reward + gamma * true_V[next_state[0], next_state[1]])
            q_values.append(expected_discounted_reward)
        true_Q_table.append(q_values)

    true_Q_table = np.asarray(true_Q_table)
    visualize_V_RM(true_Q_table, env.T, env.C)

    N = 1000
    revenue = average_n_episodes(env, true_policy.flatten(), N)
    print("Average reward over {} episodes  : {}".format(N, revenue))

    error = float("inf")
    training_errors = []
    total_epochs = 0

    tol = 100
    epochs = 10
    while error > tol and total_epochs <= 2000:
        Q_table = compute_q_table(env, agent.model)
        agent.init(np.asarray(states), np.asarray(true_Q_table), epochs)
        error = np.sqrt(np.square(true_Q_table - Q_table).sum())

        total_epochs += epochs
        training_errors.append(error)
        print("After {} epochs , error:{:.2}".format(total_epochs, error))

    Q_table = compute_q_table(env, agent.model)
    visualize_V_RM(Q_table, env.T, env.C)

    plt.plot(range(0, total_epochs, epochs), training_errors, '-o')
    plt.show()

    N = 1000
    revenue = average_n_episodes(env, true_policy.flatten(), N)
    print("Average reward over {} episodes  : {}".format(N, revenue))


if __name__ == "__main__":

    micro_times = 50
    capacity = 10
    actions = tuple(k for k in range(50, 231, 20))
    alpha = 0.4
    lamb = 0.2

    env = gym.make('gym_RM:RM-v0', micro_times=micro_times, capacity=capacity, actions=actions, alpha=alpha, lamb=lamb)

    state_size = len(env.observation_space.spaces)
    action_size = env.action_space.n

    batch_size = 64
    nb_episodes = 100

    agent = DQNAgent(state_size, action_size)

    revenues = []

    for episodes in range(nb_episodes):

        Q_table = compute_q_table(env, agent.model)
        visualize_V_RM(Q_table, env.T, env.C)

        policy = q_to_policy(env, Q_table)
        visualize_policy_RM(policy, env.T, env.C)

        state = env.set_random_state()
        state = np.reshape(state, [1, state_size])

        done = False
        revenue = 0

        while not done:
            action_idx = agent.act(state)
            next_state, reward, done, _ = env.step(env.A[action_idx])
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            revenue += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print("episode: {}/{}, revenue=: {}, loss: {:.2}, e: {:.2}".format(episodes, nb_episodes, revenue, agent.loss,
                                                                           agent.epsilon))

        revenues.append(revenue)
