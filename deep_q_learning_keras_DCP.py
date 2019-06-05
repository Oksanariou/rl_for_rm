# -*- coding: utf-8 -*-
import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from q_learning import q_to_v
from visualization_and_metrics import visualize_policy_RM, average_n_episodes, visualisation_value_RM


class DQNAgent:
    def __init__(self, input_size, action_size, target_model_update=10):
        self.input_size = input_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

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
        model.add(Dense(self.action_size, activation='relu', name='action'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def set_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_discounted_max_q_value(self, next_state):
        next_q_values = self.model.predict(next_state)
        action_idx = np.argmax(next_q_values[0])

        max_target_value = self.target_model.predict(next_state)[0][action_idx]

        return self.gamma * max_target_value

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

        for state, action_idx, reward, next_state, done in minibatch:
            q_value = reward + self.get_discounted_max_q_value(next_state)
            # q_value = reward + self.gamma * np.max(self.target_model.predict(next_state))
            # q_value = np.max(self.target_model.predict(state))

            q_values = self.model.predict(state)
            q_values[0][action_idx] = reward if done else q_value

            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        history = self.model.fit(np.array(state_batch), np.array(q_values_batch), batch_size, epochs=1, verbose=0)
        self.loss = history.history['loss'][0]

        self.update_epsilon()

        self.replay_count += 1
        if self.replay_count % self.target_model_update == 0:
            print("Updating target with current model")
            self.set_target()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def init(self, X, Y, epochs):
        self.model.fit(X, Y, batch_size, epochs=epochs, verbose=0)
        self.set_target()


def compute_q_table(env, model):
    state_size = len(env.observation_space)
    shape = [space.n for space in env.observation_space]

    states = [np.asarray((t, x)) for t in range(shape[0]) for x in range(shape[1])]

    return model.predict(np.array(states))


def q_to_policy(env, Q):
    return [env.A[action_idx] for action_idx in np.argmax(Q_table, axis=1)]
    # return np.argmax(Q, axis=1)


def visualize_V_RM(Q, T, C):
    P = np.reshape(np.max(Q, axis=1), (T, C))
    plt.figure()
    plt.imshow(P, aspect='auto')
    plt.title("Expected reward")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.colorbar()
    return plt.show()


def get_true_Q_table(env, gamma):
    shape = [space.n for space in env.observation_space]
    states = [(t, x) for t in range(shape[0]) for x in range(shape[1])]

    true_V, true_policy = dynamic_programming_env_DCP(env)

    true_Q_table = []
    for state in states:
        q_values = []
        for action in env.A:
            expected_discounted_reward = 0.
            for proba, next_state, reward, done in env.P[state][action]:
                expected_discounted_reward += proba * (reward + gamma * true_V[next_state[0], next_state[1]])
            q_values.append(expected_discounted_reward)
        true_Q_table.append(q_values)

    return np.asarray(true_Q_table), true_policy


def init_with_V(agent, env):
    shape = [space.n for space in env.observation_space]
    states = [(t, x) for t in range(shape[0]) for x in range(shape[1])]

    true_Q_table, true_policy = get_true_Q_table(env, agent.gamma)
    true_V = q_to_v(env, true_Q_table)
    visualisation_value_RM(true_V, env.T, env.C)

    N = 10000
    revenue = average_n_episodes(env, true_policy.flatten(), N)
    print("Average reward over {} episodes  : {}".format(N, revenue))

    error = float("inf")
    training_errors = []
    total_epochs = 0

    tol = 100
    epochs = 10
    while error > tol and total_epochs <= 2000:
        agent.init(np.asarray(states), np.asarray(true_Q_table), epochs)
        Q_table = compute_q_table(env, agent.model)
        error = np.sqrt(np.square(true_Q_table - Q_table).sum())

        total_epochs += epochs
        training_errors.append(error)
        print("After {} epochs , error:{:.2}".format(total_epochs, error))

    Q_table = compute_q_table(env, agent.model)
    V = q_to_v(env, Q_table)
    visualisation_value_RM(V, env.T, env.C)

    plt.figure()
    plt.plot(range(0, total_epochs, epochs), training_errors, '-o')
    plt.show()

    N = 1000
    revenue = average_n_episodes(env, true_policy.flatten(), N)
    print("Average reward over {} episodes  : {}".format(N, revenue))


if __name__ == "__main__":

    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 20))
    alpha = 0.4
    lamb = 0.2

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    state_size = len(env.observation_space.spaces)
    action_size = env.action_space.n

    batch_size = 64
    nb_episodes = 100

    agent = DQNAgent(state_size, action_size)

    revenues = []

    for episode in range(nb_episodes):

        if episode % int(nb_episodes / 10) == 0:
            Q_table = compute_q_table(env, agent.model)
            V = q_to_v(env, Q_table)
            visualisation_value_RM(V, env.T, env.C)

            policy = q_to_policy(env, Q_table)
            # visualize_policy_RM(policy, env.T, env.C)

            N = 1000
            revenue = average_n_episodes(env, policy, N)
            print("Average reward over {} episodes after {} episodes : {}".format(N, episode, revenue))
            revenues.append(revenue)

        state = env.set_random_state()
        # state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False

        while not done:
            action_idx = agent.act(state)
            next_state, reward, done, _ = env.step(env.A[action_idx])
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print("episode: {}/{}, loss: {:.2}, e: {:.2}".format(episode, nb_episodes, agent.loss, agent.epsilon))
