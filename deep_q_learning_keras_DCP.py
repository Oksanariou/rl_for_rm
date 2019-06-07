# -*- coding: utf-8 -*-
import random
from collections import deque

import gym
import sys
import matplotlib.pyplot as plt
import numpy as np
from keras import Input
from keras.layers import Dense, BatchNormalization, Lambda, K
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error, logcosh

from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from q_learning import q_to_v
from visualization_and_metrics import visualize_policy_RM, average_n_episodes, visualisation_value_RM, q_to_policy_RM, \
    reshape_matrix_of_visits
from mpl_toolkits.mplot3d import Axes3D
from SumTree import SumTree


class DQNAgent:
    def __init__(self, input_size, action_size, gamma=0.9,
                 epsilon=1., epsilon_min=1., epsilon_decay=0.9999,
                 target_model_update=10,
                 learning_rate=0.001, dueling=False, prioritized_experience_replay=False, hidden_layer_size=50,
                 loss=mean_squared_error):

        self.input_size = input_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.replay_count = 0
        self.target_model_update = target_model_update
        self.loss_value = 0.

        self.hidden_layer_size = hidden_layer_size
        self.dueling = dueling
        self.loss = loss
        self.learning_rate = learning_rate

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.M = np.zeros([env.T, env.C, env.action_space.n])

        self.prioritized_experience_replay = prioritized_experience_replay
        self.priority_capacity = 5000
        self.tree = SumTree(self.priority_capacity)
        self.priority_e = 0.01
        self.priority_a = 0.6
        self.priority_b = 0.01
        self.priority_b_increase = 0.999

    def _build_model(self):
        model_builder = self._build_dueling_model if self.dueling else self._build_simple_model
        return model_builder()

    def _build_simple_model(self):
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

        merge_layer = Lambda(lambda x: x[0] + K.mean(x[1], axis=1, keepdims=True) - x[1],
                             output_shape=(self.action_size,))

        q_value_layer = merge_layer([state_value_layer, action_value_layer])

        model = Model(inputs=[state_layer], outputs=[q_value_layer])

        model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))

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

        #q_values = self.model.predict(state)
        q_values = self.target_model.predict(state)
        return np.argmax(q_values[0])  # returns action

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

    def prioritized_update(self, idx, error):
        priority = ((error + self.priority_e) ** self.priority_a) * (
                1 / ((error + self.priority_e) ** self.priority_a)) ** self.priority_b
        self.tree.update(idx, priority)

    def replay(self, batch_size, method):
        minibatch = self.prioritized_sample(batch_size) if self.prioritized_experience_replay else random.sample(
            self.memory, batch_size)

        state_batch, q_values_batch = [], []
        for i in range(len(minibatch)):
            if self.prioritized_experience_replay:
                idx, (state, action_idx, reward, next_state, done) = minibatch[i][0], minibatch[i][1]
            else:
                state, action_idx, reward, next_state, done = minibatch[i]

            t, x = state[0][0], state[0][1]
            self.M[t, x, action_idx] += 1
            if method == 0:
                q_value = self.target_model.predict(state)[0][action_idx]
            elif method == 1:
                q_value = reward + self.gamma * np.max(self.target_model.predict(next_state))
            elif method == 2:
                q_value = reward + self.get_discounted_max_q_value(next_state)

            # q_values = self.target_model.predict(state)
            q_values = self.model.predict(state)
            error = abs(q_values[0][action_idx] - q_value)
            q_values[0][action_idx] = reward if done else q_value

            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

            if self.prioritized_experience_replay:
                self.prioritized_update(idx, error)

        history = self.model.fit(np.array(state_batch), np.array(q_values_batch), batch_size, epochs=1, verbose=0)
        self.loss_value = history.history['loss'][0]

        self.update_epsilon()
        self.update_priority_b()

        self.replay_count += 1
        if self.replay_count % self.target_model_update == 0:
            print("Updating target with current model")
            self.set_target()

    def update_priority_b(self):
        self.priority_b = min(1., self.priority_b / self.priority_b_increase)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def init(self, X, Y, epochs, batch_size):
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


def init_with_V(agent, env, batch_size):
    shape = [space.n for space in env.observation_space]
    states = [(t, x) for t in range(shape[0]) for x in range(shape[1])]

    true_Q_table, true_policy = get_true_Q_table(env, agent.gamma)
    true_V = q_to_v(env, true_Q_table)
    # visualisation_value_RM(true_V, env.T, env.C)

    N = 10000
    revenue = average_n_episodes(env, true_policy.flatten(), N)
    print("Average reward of the true policy over {} episodes  : {}".format(N, revenue))

    error = float("inf")
    training_errors = []
    total_epochs = 0

    tol = 10
    epochs = 10
    while error > tol and total_epochs <= 2000:
        agent.init(np.asarray(states), np.asarray(true_Q_table), epochs, batch_size)
        Q_table = compute_q_table(env, agent.model)
        error = np.sqrt(np.square(true_Q_table - Q_table).sum())

        total_epochs += epochs
        training_errors.append(error)
        # print("After {} epochs , error:{:.2}".format(total_epochs, error))

    Q_table = compute_q_table(env, agent.model)
    V = q_to_v(env, Q_table)
    # visualisation_value_RM(V, env.T, env.C)

    plt.figure()
    plt.plot(range(0, total_epochs, epochs), training_errors, '-o')
    plt.xlabel("Epochs")
    plt.ylabel("Error between the true Q-table and the agent's Q-table")
    plt.show()


def print_diff(env, agent):
    true_Q_table, true_policy = get_true_Q_table(env, agent.gamma)
    true_V = q_to_v(env, true_Q_table)

    Q_table = compute_q_table(env, agent.model)
    V = q_to_v(env, Q_table)
    policy = q_to_policy_RM(env, Q_table)

    print("Visited states")
    print(agent.M)
    print("Difference with the true Q-table")
    print(
        abs(true_Q_table.reshape(env.T, env.C, env.action_space.n) - Q_table.reshape(env.T, env.C, env.action_space.n)))
    print("Difference with the true V-table")
    print(abs(true_V.reshape(env.T, env.C) - V.reshape(env.T, env.C)))
    print("Difference with the true Policy")
    print(abs(true_policy.reshape(env.T, env.C) - policy.reshape(env.T, env.C)))


def train(agent, nb_episodes, batch_size, method, a, absc,
          errors_Q_table, errors_V_table, errors_policy):
    true_Q_table, true_policy = get_true_Q_table(env, agent.gamma)
    true_V = q_to_v(env, true_Q_table)
    revenues = []
    for episode in range(nb_episodes):

        if episode % int(nb_episodes / 10) == 0:
            Q_table = compute_q_table(env, agent.model)
            policy = q_to_policy_RM(env, Q_table)
            V = q_to_v(env, Q_table)
            visualisation_value_RM(V, env.T, env.C)
            error_Q_table = np.sqrt(
                np.square(true_Q_table.reshape(4, 4, 4)[:-1, :-1] - Q_table.reshape(4, 4, 4)[:-1, :-1]).sum())
            errors_Q_table.append(error_Q_table)
            error_V_table = np.sqrt(np.square(true_V - V).sum())
            errors_V_table.append(error_V_table)
            error_policy = np.sqrt(np.square(true_policy[:-1, :-1] - policy.reshape(4, 4)[:-1, :-1]).sum())
            errors_policy.append(error_policy)
            a += int(nb_episodes / 10)
            absc.append(a)

            policy = q_to_policy_RM(env, Q_table)
            # visualize_policy_RM(policy, env.T, env.C)

            N = 1000
            revenue = average_n_episodes(env, policy, N)
            print("Average reward over {} episodes after {} episodes : {}".format(N, episode, revenue))
            revenues.append(revenue)
            print_diff(env, agent)

        state = env.set_random_state()
        # state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False

        while not done:
            action_idx = agent.act(state)
            next_state, reward, done, _ = env.step(env.A[action_idx])
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action_idx, reward, next_state, done)
            if agent.prioritized_experience_replay:
                agent.tree.add(reward + agent.priority_e, (state, action_idx, reward, next_state, done))

            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, method)
        print("episode: {}/{}, loss: {:.2}, e: {:.2}, b: {:.2}".format(episode, nb_episodes, agent.loss_value, agent.epsilon, agent.priority_b))
    plt.figure()
    plt.plot(absc, errors_Q_table, '-o')
    plt.xlabel("Epochs")
    plt.ylabel("Difference with the true Q-table")
    plt.show()
    plt.figure()
    plt.plot(absc, errors_V_table, '-o')
    plt.xlabel("Epochs")
    plt.ylabel("Difference with the true V-table")
    plt.show()
    plt.plot(absc, errors_policy, '-o')
    plt.xlabel("Epochs")
    plt.ylabel("Difference with the policy")
    plt.show()
    return agent, a


def init_target_network_with_true_Q_table(agent, env, batch_size):
    init_with_V(agent, env, batch_size)
    agent.model = agent._build_model()
    agent.target_model_update = sys.maxsize
    return agent


if __name__ == "__main__":
    data_collection_points = 4
    micro_times = 3
    capacity = 4
    actions = tuple(k for k in range(50, 231, 50))
    alpha = 0.8
    lamb = 0.7

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    state_size = len(env.observation_space.spaces)
    action_size = env.action_space.n

    batch_size = 30
    nb_episodes = 5000

    agent = DQNAgent(state_size, action_size)
    init_target_network_with_true_Q_table(agent, env, batch_size)
    method = 0

    errors_Q_table = []
    errors_V_table = []
    errors_policy = []
    absc = []
    a = 0

    true_Q_table, true_policy = get_true_Q_table(env, agent.gamma)
    true_V = q_to_v(env, true_Q_table)
    visualisation_value_RM(true_V, env.T, env.C)
    visualize_policy_RM(true_policy, env.T, env.C)

    agent, a = train(agent, nb_episodes, batch_size, method, a, absc, errors_Q_table, errors_V_table, errors_policy)

    Q_table = compute_q_table(env, agent.target_model)
    V = q_to_v(env, Q_table)
    visualisation_value_RM(V, env.T, env.C)

    X, Y, Z, values = reshape_matrix_of_visits(agent.M, env)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter3D(X, Y, Z, c=values, cmap='hot')
    fig.colorbar(p, ax=ax)
    plt.show()
