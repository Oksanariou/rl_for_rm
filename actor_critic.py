import matplotlib.pyplot as plt
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
import numpy as np
import gym
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import random
from visualization_and_metrics import visualize_policy_FL, average_n_episodes_FL
from collections import deque
from tqdm import tqdm


class CriticNetwork:
    def __init__(self, state_size):
        self.state_size = state_size
        self.learning_rate = 0.1
        self.model = self._build_model()
        self.memory = deque(maxlen=80)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(164, init='lecun_uniform', input_shape=(self.state_size,)))
        model.add(Activation('relu'))
        model.add(Dense(150, init='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(1, init='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True))
        return model

    def remember(self, state, value):
        self.memory.append((state, value))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        for m_state, m_value in minibatch:
            y = np.empty([1])
            y[0] = m_value
            X_train.append(m_state.reshape((self.state_size,)))
            y_train.append(y.reshape((1,)))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)


class ActorNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.1
        self.model = self._build_model()
        self.memory = deque(maxlen=80)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(164, init='lecun_uniform', input_shape=(self.state_size,)))
        model.add(Activation('relu'))
        model.add(Dense(150, init='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size, init='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True))
        return model

    def remember(self, state, action, delta):
        self.memory.append((state, action, delta))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        for m_state, m_action, m_delta in minibatch:
            old_qval = self.model.predict(m_state.reshape(1, self.state_size, ))
            y = np.zeros((1, self.action_size))
            y[:] = old_qval[:]
            y[0][m_action] = m_delta
            X_train.append(m_state.reshape((self.state_size,)))
            y_train.append(y.reshape((self.action_size,)))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)


def to_onehot(size, value):
    my_onehot = np.zeros((size))
    my_onehot[value] = 1.0
    return my_onehot


def trainer(env, epochs=1000, batch_size=40, gamma=0.975, epsilon=1, min_epsilon=0.1):
    OBSERVATION_SPACE = env.observation_space.n
    ACTION_SPACE = env.action_space.n
    critic = CriticNetwork(OBSERVATION_SPACE)
    actor = ActorNetwork(OBSERVATION_SPACE, ACTION_SPACE)

    for i in tqdm(range(epochs)):
        observation = env.reset()
        done = False
        reward = 0

        while (not done):
            # Get original state, original reward, and critic's value for this state.
            orig_state = to_onehot(OBSERVATION_SPACE, observation)
            orig_reward = reward
            orig_val = critic.model.predict(orig_state.reshape(1, OBSERVATION_SPACE))

            if (random.random() < epsilon):
                action = np.random.randint(0, ACTION_SPACE)
            else:
                qval = actor.model.predict(orig_state.reshape(1, OBSERVATION_SPACE))
                action = (np.argmax(qval))

            # Take action, observe new state S'
            new_observation, new_reward, done, info = env.step(action)
            new_state = to_onehot(OBSERVATION_SPACE, new_observation)
            # Critic's value for this new state.
            new_val = critic.model.predict(new_state.reshape(1, OBSERVATION_SPACE))

            if not done:  # Non-terminal state.
                target = orig_reward + (gamma * new_val)
            else:
                # In terminal states, the environment tells us
                # the value directly.
                target = orig_reward + (gamma * new_reward)

            best_val = max((orig_val * gamma), target)
            critic.remember(orig_state, best_val)
            # If we are in a terminal state, append a replay for it also.
            if done:
                critic.remember(new_state, float(new_reward))
            actor_delta = new_val - orig_val
            actor.remember(orig_state, action, actor_delta)

            if (len(critic.memory) >= batch_size):
                critic.replay(batch_size)
            if (len(actor.memory) >= batch_size):
                actor.replay(batch_size)

            observation = new_observation
            reward = new_reward

        if epsilon > min_epsilon:
            epsilon -= (1 / epochs)
    return actor.model


def extract_policy(env, network):
    policy = []
    for s in range(env.nS):
        state_one_hot = to_onehot(env.nS, s)
        qval = network.predict(state_one_hot.reshape(1, env.nS))
        policy.append((np.argmax(qval)))
    return policy

if __name__ == '__main__':
    MY_ENV_NAME = 'FrozenLake-v0'

    env = gym.make(MY_ENV_NAME)
    epochs = 1000
    batch_size = 40
    gamma = 0.975
    epsilon = 1
    min_epsilon = 0.1

    trained_actor_network = trainer(env, epochs, batch_size, gamma, epsilon, min_epsilon)
    policy = extract_policy(env, trained_actor_network)
    visualize_policy_FL(policy)
    print(average_n_episodes_FL(env, policy, 100))
