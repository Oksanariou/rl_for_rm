# -*- coding: utf-8 -*-
import random
from collections import deque

import sys
import matplotlib.pyplot as plt
import numpy as np

from keras import Input
from keras.layers import Dense, BatchNormalization, Lambda, GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error, logcosh

from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from SumTree import SumTree

import tensorflow as tf
from keras import backend as K


def DQNAgent_builder(env, parameters_dict):
    return DQNAgent(env, gamma=parameters_dict["gamma"], epsilon=parameters_dict["epsilon"],
                    epsilon_min=parameters_dict["epsilon_min"], epsilon_decay=parameters_dict["epsilon_decay"],
                    replay_method=parameters_dict["replay_method"],
                    target_model_update=parameters_dict["target_model_update"],
                    batch_size=parameters_dict["batch_size"], state_scaler=parameters_dict["state_scaler"],
                    value_scaler=parameters_dict["value_scaler"], learning_rate=parameters_dict["learning_rate"],
                    dueling=parameters_dict["dueling"], hidden_layer_size=parameters_dict["hidden_layer_size"],
                    prioritized_experience_replay=parameters_dict["prioritized_experience_replay"],
                    memory_size=parameters_dict["memory_size"], mini_batch_size=parameters_dict["mini_batch_size"],
                    loss=parameters_dict["loss"], use_weights=parameters_dict["use_weights"],
                    use_optimal_policy=parameters_dict["use_optimal_policy"])


class DQNAgent:
    def __init__(self, env, gamma=0.9,
                 epsilon=1., epsilon_min=0.2, epsilon_decay=0.9999,
                 replay_method="DDQL", target_model_update=10, batch_size=32,
                 state_scaler=None, value_scaler=None,
                 learning_rate=0.001, dueling=False, hidden_layer_size=50,
                 prioritized_experience_replay=False, memory_size=500,
                 mini_batch_size=64,
                 loss=mean_squared_error,
                 use_weights=False,
                 use_optimal_policy=False):

        self.env = env
        self.input_size = len(self.env.observation_space.spaces)
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=memory_size)
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.replay_method = replay_method
        self.target_model_update = target_model_update

        self.episode = 0
        self.replay_count = 0
        self.loss_value = 0.
        self.last_visited = []

        self.state_scaler = state_scaler
        self.value_scaler = value_scaler

        self.hidden_layer_size = hidden_layer_size
        self.dueling = dueling
        self.loss = loss
        self.learning_rate = learning_rate
        self.use_weights = use_weights
        self.state_weights = self.compute_state_weights()
        # self.state_weights = self.compute_state_weights() if state_weights else state_weights

        self.model = self._build_model()
        self.target_model = self._build_model()

        self.prioritized_experience_replay = prioritized_experience_replay
        self.priority_capacity = 2000
        self.tree = SumTree(self.priority_capacity)
        self.priority_e = 0.01
        self.priority_a = 0.7
        self.priority_b = 0.5
        self.priority_b_increase = 0.9999

        self.name = "agent"

        self.use_optimal_policy = use_optimal_policy
        self.optimal_policy = self.compute_optimal_policy()

    def compute_optimal_policy(self):
        V, P_ref = dynamic_programming_env_DCP(self.env)
        return P_ref.reshape(self.env.T * self.env.C)

    def _build_model(self):
        model_builder = self._build_dueling_model if self.dueling else self._build_simple_model
        return model_builder()

    def _build_simple_model(self):
        # with K.tf.device('/gpu:0'):
        #     config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 1})
        #     session = tf.Session(config=config)
        #     K.set_session(session)
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.hidden_layer_size, input_shape=(self.input_size,), activation='relu', name='state'))
        model.add(BatchNormalization())
        # model.add(Dropout(rate=0.2))
        model.add(Dense(self.hidden_layer_size, activation='relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(rate=0.2))
        model.add(Dense(self.action_size, activation='relu', name='action'))
        # model.add(GaussianNoise(0.01))
        model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))

        return model

    def _build_dueling_model(self):
        # Neural Net for Dueling Deep-Q learning Model
        # We need the Keras functional API here

        state_layer = Input(shape=(self.input_size,))

        action_value_layer = Dense(self.hidden_layer_size, activation='relu')(state_layer)
        action_value_layer = BatchNormalization()(action_value_layer)
        action_value_layer = Dense(self.hidden_layer_size, activation='relu')(action_value_layer)
        action_value_layer = BatchNormalization()(action_value_layer)
        action_value_layer = Dense(self.action_size, activation='relu')(action_value_layer)

        state_value_layer = Dense(self.hidden_layer_size, activation='relu')(state_layer)
        state_value_layer = BatchNormalization()(state_value_layer)
        state_value_layer = Dense(self.hidden_layer_size, activation='relu')(state_value_layer)
        state_value_layer = BatchNormalization()(state_value_layer)
        state_value_layer = Dense(1, activation='relu')(state_value_layer)

        merge_layer = Lambda(lambda x: x[0] + x[1] - K.mean(x[1], axis=1, keepdims=True),
                             output_shape=(self.action_size,))

        q_value_layer = merge_layer([state_value_layer, action_value_layer])

        model = Model(inputs=[state_layer], outputs=[q_value_layer])

        model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))

        return model

    def set_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def set_model(self, model):
        self.model.set_weights(model.get_weights())

    def get_discounted_max_q_value(self, next_state_batch):
        next_q_values = self.model.predict(next_state_batch)
        # action_idx = np.argmax(next_q_values[0])
        action_idx = next_q_values.argmax(axis=1)

        target_values = self.target_model.predict(next_state_batch)
        max_target_values = np.array([target_values[k][action_idx[k]] for k in range(len(next_q_values))])

        return self.gamma * max_target_values

    def remember(self, state, action_idx, reward, next_state, done):
        sample_weight = self.state_weights[state] if self.use_weights else 1.

        state = self.normalize_state(state)
        state = np.reshape(state, [1, self.input_size])

        next_state = self.normalize_state(next_state)
        next_state = np.reshape(next_state, [1, self.input_size])

        reward = self.normalize_value(reward)

        if self.prioritized_experience_replay:
            self.memory.append((state, action_idx, reward, next_state, done, sample_weight))
            self.tree.add(reward + self.priority_e, (state, action_idx, reward, next_state, done, sample_weight))
        else:
            self.memory.append((state, action_idx, reward, next_state, done, sample_weight))

    def normalize_states(self, states):
        if self.state_scaler is None:
            return np.asarray(states)
        return np.asarray([self.normalize_state(state) for state in states])

    def normalize_state(self, state):
        if self.state_scaler is None:
            return state
        return self.state_scaler.scale(state)

    def denormalize_states(self, states):
        if self.state_scaler is None:
            return np.asarray(states)
        return np.asarray([self.denormalize_state(state) for state in states])

    def denormalize_state(self, state):
        if self.state_scaler is None:
            return state
        return self.state_scaler.unscale(state)

    def normalize_values(self, values):
        if self.value_scaler is None:
            return np.asarray(values)
        return np.asarray([self.normalize_value(value) for value in values])

    def normalize_value(self, value):
        if self.value_scaler is None:
            return value
        return self.value_scaler.scale(value)

    def denormalize_values(self, values):
        if self.value_scaler is None:
            return np.asarray(values)
        return np.asarray([self.denormalize_value(value) for value in values])

    def denormalize_value(self, value):
        if self.value_scaler is None:
            return value
        return self.value_scaler.unscale(value)

    def prioritized_sample(self, batch_size):
        minibatch = []
        segment = (self.tree.total()) / batch_size
        for i in range(0, batch_size):
            a = segment * i + self.priority_e
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, priority, data) = self.tree.get(s)
            minibatch.append((idx, data))
        return minibatch

    def compute_priority(self, error):
        return (error + self.priority_e) ** self.priority_a

    def prioritized_update(self, idx_batch, error_bacth):
        for k in range(len(idx_batch)):
            priority = self.compute_priority(error_bacth[k])
            self.tree.update(idx_batch[k], priority)

    def update_priority_b(self):
        self.priority_b = min(1., self.priority_b / self.priority_b_increase)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def compute_state_weights(self):
        shape = [space.n for space in self.env.observation_space]
        compute_weight = lambda x: 1 + max(1. * x[0] / self.env.T, 1. * x[1] / self.env.C)
        state_weights = [((t, x), compute_weight((t, x))) for t in range(shape[0]) for x in range(shape[1])]

        return dict(state_weights)

    def compute_sample_weight(self, states):
        if self.state_weights is not None:
            return np.asarray([self.state_weights[(t, x)] for t, x in states])
        else:
            return None

    def compute_q_table(self, target=False):
        shape = [space.n for space in self.env.observation_space]

        states = [np.asarray((t, x)) for t in range(shape[0]) for x in range(shape[1])]

        model = self.target_model if target else self.model

        Q_table = model.predict(self.normalize_states(states))

        Q_table = self.denormalize_values(Q_table.flatten()).reshape(self.env.T, self.env.C, self.env.action_space.n)

        Q_table[:, -1] = 0.  # Setting the Q values of the states (x,t) such that x = C to zero
        Q_table[-1] = 0.  # Setting the Q values of the states (x,t) such that t = T to zero

        return Q_table.reshape(self.env.T * self.env.C, self.env.action_space.n)

    def get_true_Q_table(self):
        shape = [space.n for space in self.env.observation_space]
        states = [(t, x) for t in range(shape[0]) for x in range(shape[1])]

        true_V, true_policy = dynamic_programming_env_DCP(self.env)

        true_Q_table = []
        for state in states:
            q_values = []
            for action in self.env.A:
                expected_discounted_reward = 0.
                for proba, next_state, reward, done in self.env.P[state][action]:
                    expected_discounted_reward += proba * (reward + self.gamma * true_V[next_state[0], next_state[1]])
                q_values.append(expected_discounted_reward)
            true_Q_table.append(q_values)

        return np.asarray(true_Q_table), true_policy

    def init(self, X, Y, epochs):
        X = self.normalize_states(X)
        Y = self.normalize_values(Y)

        self.model.fit(X, Y, epochs=epochs, verbose=0, batch_size=self.batch_size)
        self.set_target()

    def init_with_V(self):
        shape = [space.n for space in self.env.observation_space]
        states = [(t, x) for t in range(shape[0]) for x in range(shape[1])]

        true_Q_table, true_policy = self.get_true_Q_table()

        error = float("inf")
        training_errors = []
        total_epochs = 0

        tol = 10
        epochs = 10
        while error > tol and total_epochs <= 3000:
            self.init(states, true_Q_table, epochs)
            Q_table = self.compute_q_table()
            error = np.sqrt(np.square(true_Q_table - Q_table).sum())

            total_epochs += epochs
            training_errors.append(error)
            # print("After {} epochs , error:{:.2}".format(total_epochs, error))

        # Q_table = self.compute_q_table(env, self)
        # V = q_to_v(env, Q_table)
        # visualisation_value_RM(V, env.T, env.C)

        plt.figure()
        plt.plot(range(0, total_epochs, epochs), training_errors, '-o')
        plt.xlabel("Epochs")
        plt.ylabel("Error between the true Q-table and the agent's Q-table")
        plt.show()

    def init_network_with_true_Q_table(self):
        self.init_with_V()

    def init_target_network_with_true_Q_table(self):
        self.init_network_with_true_Q_table()
        # Reset main model
        self.model = self._build_model()
        # And make sure the target model is never updated
        self.target_model_update = sys.maxsize

    def act(self, state):
        state_idx = self.env.to_idx(state[0], state[1])

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = self.normalize_state(state)
        state = np.reshape(state, [1, self.input_size])
        q_values = self.model.predict(state)
        action = self.env.A.index(self.optimal_policy[state_idx]) if self.use_optimal_policy else np.argmax(q_values[0])

        return action  # returns action

    def replay(self, episode):
        self.episode = episode

        if len(self.memory) < self.mini_batch_size:
            return

        if self.prioritized_experience_replay:
            minibatch_with_idx = self.prioritized_sample(self.mini_batch_size)
            idx_batch, minibatch = zip(*minibatch_with_idx)
            idx_batch = np.array(idx_batch)
        else:
            minibatch = random.sample(self.memory, self.mini_batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, sample_weights = zip(*minibatch)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, sample_weights = np.array(
            state_batch).reshape(self.mini_batch_size, self.input_size), np.array(action_batch), np.array(
            reward_batch), np.array(next_state_batch).reshape(self.mini_batch_size, self.input_size), np.array(
            done_batch), np.array(sample_weights)

        if self.replay_method == "TARGET_ONLY":
            q_values_target = np.array(
                [self.target_model.predict(state_batch)[k][action_batch[k]] for k in range(self.mini_batch_size)])
        elif self.replay_method == "DQL":
            q_values_target = reward_batch + self.gamma * self.model.predict(next_state_batch).max(axis=1)
        elif self.replay_method == "DDQL":
            q_values_target = reward_batch + self.get_discounted_max_q_value(next_state_batch)

        q_values_state = self.model.predict(state_batch)

        if self.prioritized_experience_replay:
            error_batch = np.array(
                [abs(q_values_state[k][action_batch[k]] - q_values_target[k]) for k in range(self.mini_batch_size)])
            self.prioritized_update(idx_batch, error_batch)

        for k in range(self.mini_batch_size):
            q_values_state[k][action_batch[k]] = reward_batch[k] if done_batch[k] else q_values_target[k]

        history = self.model.fit(np.array(state_batch), np.array(q_values_state), epochs=1, verbose=0,
                                 sample_weight=np.array(sample_weights), batch_size=self.batch_size)
        self.loss_value = history.history['loss'][0]

        self.update_priority_b()

        self.last_visited = zip(state_batch, action_batch)

        self.replay_count += 1
        if self.replay_count % self.target_model_update == 0:
            print("Updating target with current model")
            self.set_target()

    def train(self, nb_episodes, callbacks):
        for episode in range(nb_episodes):

            # state = self.env.set_random_state()
            state = self.env.reset()

            done = False

            while not done:
                action_idx = self.act(state)
                next_state, reward, done, _ = self.env.step(self.env.A[action_idx])

                self.remember(state, action_idx, reward, next_state, done)

                state = next_state

            self.replay(episode)

            self.update_epsilon()

            for callback in callbacks:
                callback.run(episode)
