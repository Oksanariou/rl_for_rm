import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
import numpy as np
import gym
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import random
from collections import deque
from tqdm import tqdm
from visualization_and_metrics import visualizing_epsilon_decay, visualize_policy_RM, visualisation_value_RM


class CriticNetwork:
    def __init__(self, state_size):
        self.state_size = state_size
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = deque(maxlen=100)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, value):
        self.memory.append((state, value))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        for m_state, m_value in minibatch:
            y = np.empty([1])
            y[0] = m_value
            X_train.append(m_state[0])
            y_train.append(y.reshape((1,)))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.model.fit(X_train, y_train, batch_size, nb_epoch=1, verbose=0)


class ActorNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = deque(maxlen=100)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, delta):
        self.memory.append((state, action, delta))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        for m_state, m_action, m_delta in minibatch:
            old_qval = self.model.predict(m_state)[0]
            y = np.zeros((1, self.action_size))
            y[:] = old_qval[:]
            y[0][m_action] = m_delta
            X_train.append(m_state[0])
            y_train.append(y.reshape((self.action_size,)))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.model.fit(X_train, y_train, batch_size, nb_epoch=1, verbose=0)


def to_onehot(size, value):
    my_onehot = np.zeros((size))
    my_onehot[value] = 1.0
    return my_onehot


def trainer(env, epochs=1000, batch_size=40, gamma=0.975, epsilon=1, epsilon_min=0.1, epsilon_decay=0.99, state_size=2):
    critic = CriticNetwork(state_size)
    actor = ActorNetwork(state_size, env.nA)

    for i in tqdm(range(epochs)):
        state = env.set_random_state()
        state = np.reshape(state, [1, state_size])
        done = False

        while (not done):
            value_state = critic.model.predict(state)[0]

            if (random.random() < epsilon):
                action_idx = np.random.randint(0, env.nA)
            else:
                qval = actor.model.predict(state)[0]
                action_idx = (np.argmax(qval))
            action = env.A[action_idx]

            next_state, r, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            value_next_state = critic.model.predict(next_state)[0]

            if not done:  # Non-terminal state.
                target = r + (gamma * value_next_state)
            else:
                target = r
            error = target - value_state

            critic.remember(state, target)

            old_qval = actor.model.predict(state)[0]
            y = np.zeros((1, env.nA))
            y[:] = old_qval[:]
            y[0][action_idx] = error
            actor.model.fit(state, y, epochs=1, verbose=0)

            if (len(critic.memory) >= batch_size):
                critic.replay(batch_size)

            state = next_state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if i % int(epochs / 10) == 0:
            values, policy = extract_values_policy(env, actor.model)
            visualisation_value_RM(values, env.T, env.C)
            visualize_policy_RM(policy, env.T, env.C)
    return actor.model


def extract_values_policy(env, network):
    policy, values = [], []
    for state_idx in range(env.nS):
        state = env.to_coordinate(state_idx)
        state = np.reshape(state, [1, state_size])
        qval = network.predict(state)[0]
        idx_policy = np.argmax(qval)
        action = env.A[idx_policy]
        policy.append((action))
        values.append(np.max(qval))
    policy, values = np.array(policy), np.array(values)
    return values, policy


if __name__ == '__main__':
    # MY_ENV_NAME = 'FrozenLake-v0'
    # env = gym.make(MY_ENV_NAME)

    micro_times = 50
    capacity = 10
    actions = tuple(k for k in range(50, 231, 20))
    alpha = 0.4
    lamb = 0.2

    env = gym.make('gym_RM:RM-v0', micro_times=micro_times, capacity=capacity, actions=actions, alpha=alpha, lamb=lamb)
    state_size = len(env.observation_space.spaces)
    epochs = 5000
    batch_size = 32
    gamma = 0.99
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 0.9995

    visualizing_epsilon_decay(epochs, epsilon, epsilon_min, epsilon_decay)

    trained_actor_network = trainer(env, epochs, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, state_size)
    values, policy = extract_values_policy(env, trained_actor_network)
    visualisation_value_RM(values, env.T, env.C)
    visualize_policy_RM(policy, env.T, env.C)
    # print(average_n_episodes_FL(env, policy, 100))
