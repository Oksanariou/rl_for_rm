# -*- coding: utf-8 -*-
import random
from collections import deque

import sys
import matplotlib.pyplot as plt
import numpy as np

from keras import Input
from keras.layers import Dense, BatchNormalization, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error, logcosh

from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from SumTree import SumTree

import tensorflow as tf
from keras import backend as K

class DQNAgent:
    def __init__(self, env, gamma=0.9,
                 epsilon=1., epsilon_min=0.2, epsilon_decay=0.9999,
                 replay_method="DDQL", target_model_update=10, batch_size=32,
                 state_scaler=None, value_scaler=None,
                 learning_rate=0.001, dueling=False, hidden_layer_size=50,
                 prioritized_experience_replay=False, memory_size=500,
                 mini_batch_size=64,
                 loss=mean_squared_error,
                 state_weights=None):

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
        self.state_weights = self.compute_state_weights() if state_weights else state_weights

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

    def get_discounted_max_q_value(self, next_state):
        next_q_values = self.model.predict(next_state)
        action_idx = np.argmax(next_q_values[0])

        max_target_value = self.target_model.predict(next_state)[0][action_idx]

        return self.gamma * max_target_value

    def remember(self, state, action_idx, reward, next_state, done):
        sample_weight = self.state_weights[state] if self.state_weights is not None else 1.

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

    def prioritized_update(self, idx, error):
        priority = self.compute_priority(error)
        self.tree.update(idx, priority)

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
        while error > tol and total_epochs <= 3_000:
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
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = self.normalize_state(state)
        state = np.reshape(state, [1, self.input_size])
        q_values = self.model.predict(state)

        return np.argmax(q_values[0])  # returns action

    def replay(self, episode):
        self.episode = episode

        if len(self.memory) < self.mini_batch_size:
            return

        minibatch = self.prioritized_sample(self.mini_batch_size) if self.prioritized_experience_replay else random.sample(
            self.memory, self.mini_batch_size)

        state_batch, q_values_batch, action_batch, sample_weights = [], [], [], []
        for i in range(len(minibatch)):
            if self.prioritized_experience_replay:
                idx, (state, action_idx, reward, next_state, done, sample_weight) = minibatch[i][0], minibatch[i][1]
            else:
                state, action_idx, reward, next_state, done, sample_weight = minibatch[i]

            if self.replay_method == "TARGET_ONLY":
                # To learn the target model Q values directly without accounting for an instant reward
                q_value = self.target_model.predict(state)[0][action_idx]
            elif self.replay_method == "DQL":
                # To learn the instant reward and model V table
                q_value = reward + self.gamma * np.max(self.model.predict(next_state))
            elif self.replay_method == "DDQL":
                # To learn the instant reward, the model optimal action and target model V table
                q_value = reward + self.get_discounted_max_q_value(next_state)

            q_values = self.model.predict(state)
            q_values[0][action_idx] = reward if done else q_value

            state_batch.append(state[0])
            q_values_batch.append(q_values[0])
            action_batch.append(action_idx)

            if self.prioritized_experience_replay:
                error = abs(q_values[0][action_idx] - q_value)
                self.prioritized_update(idx, error)

            sample_weights.append(sample_weight)

        history = self.model.fit(np.array(state_batch), np.array(q_values_batch), epochs=1, verbose=0,
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

