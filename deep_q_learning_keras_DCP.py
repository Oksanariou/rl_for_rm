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
                 epsilon=1., epsilon_min=0.2, epsilon_decay=0.9999,
                 replay_method="DDQL", target_model_update=10, batch_size=32,
                 state_scaler=None, value_scaler=None,
                 learning_rate=0.001, dueling=False, hidden_layer_size=50,
                 prioritized_experience_replay=False, memory_size=500,
                 loss=mean_squared_error,
                 state_weights=None):

        self.input_size = input_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
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
        self.state_weights = state_weights

        self.model = self._build_model()
        self.target_model = self._build_model()

        self.prioritized_experience_replay = prioritized_experience_replay
        self.priority_capacity = 5000
        self.tree = SumTree(self.priority_capacity)
        self.priority_e = 0.01
        self.priority_a = 0.7
        self.priority_b = 0.5
        self.priority_b_increase = 0.9999

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

        merge_layer = Lambda(lambda x: x[0] + x[1] - K.mean(x[1], axis=1, keepdims=True),
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

    def remember(self, state, action_idx, reward, next_state, done):
        sample_weight = self.state_weights[state] if self.state_weights is not None else 1.

        state = self.normalize_state(state)
        state = np.reshape(state, [1, self.input_size])

        next_state = self.normalize_state(next_state)
        next_state = np.reshape(next_state, [1, self.input_size])

        reward = self.normalize_value(reward)

        if self.prioritized_experience_replay:
            self.tree.add(reward + agent.priority_e, (state, action_idx, reward, next_state, done, sample_weight))
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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = self.normalize_state(state)
        state = np.reshape(state, [1, self.input_size])
        q_values = self.model.predict(state)

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
        # priority = ((error + self.priority_e) ** self.priority_a) * (
        #         1 / ((error + self.priority_e) ** self.priority_a)) ** self.priority_b
        priority = ((error + self.priority_e) ** self.priority_a)
        self.tree.update(idx, priority)

    def replay(self, episode):
        self.episode = episode

        if len(self.memory) < self.batch_size:
            return

        minibatch = self.prioritized_sample(self.batch_size) if self.prioritized_experience_replay else random.sample(
            self.memory, self.batch_size)

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
                                 sample_weight=np.array(sample_weights))
        self.loss_value = history.history['loss'][0]

        self.update_priority_b()

        self.last_visited = zip(state_batch, action_batch)

        self.replay_count += 1
        if self.replay_count % self.target_model_update == 0:
            print("Updating target with current model")
            self.set_target()

    def update_priority_b(self):
        self.priority_b = min(1., self.priority_b / self.priority_b_increase)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def init(self, X, Y, epochs):
        X = self.normalize_states(X)
        Y = self.normalize_values(Y)

        self.model.fit(X, Y, epochs=epochs, verbose=0)
        self.set_target()

    def compute_sample_weight(self, states):
        if self.state_weights is not None:
            return np.asarray([self.state_weights[(t, x)] for t, x in states])
        else:
            return None


def compute_q_table(env, agent, target=False):
    state_size = len(env.observation_space)
    shape = [space.n for space in env.observation_space]

    states = [np.asarray((t, x)) for t in range(shape[0]) for x in range(shape[1])]

    model = agent.target_model if target else agent.model

    Q_table = model.predict(agent.normalize_states(states))

    Q_table = agent.denormalize_values(Q_table.flatten()).reshape(env.T, env.C, env.action_space.n)

    Q_table[:, -1] = 0.  # Setting the Q values of the states (x,t) such that x = C to zero
    Q_table[-1] = 0.  # Setting the Q values of the states (x,t) such that t = T to zero

    return Q_table.reshape(env.T * env.C, env.action_space.n)


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

    error = float("inf")
    training_errors = []
    total_epochs = 0

    tol = 10
    epochs = 10
    while error > tol and total_epochs <= 2000:
        agent.init(states, true_Q_table, epochs)
        Q_table = compute_q_table(env, agent)
        error = np.sqrt(np.square(true_Q_table - Q_table).sum())

        total_epochs += epochs
        training_errors.append(error)
        # print("After {} epochs , error:{:.2}".format(total_epochs, error))

    Q_table = compute_q_table(env, agent)
    V = q_to_v(env, Q_table)
    visualisation_value_RM(V, env.T, env.C)

    plt.figure()
    plt.plot(range(0, total_epochs, epochs), training_errors, '-o')
    plt.xlabel("Epochs")
    plt.ylabel("Error between the true Q-table and the agent's Q-table")
    plt.show()


class Callback(object):
    def __init__(self, condition, agent, env):
        super(Callback).__init__()
        self.condition = condition
        self.agent = agent
        self.env = env

    def run(self, episode):
        if not self.condition(episode):
            return
        self._run()

    def _run(self):
        raise NotImplementedError


class AgentMonitor(Callback):

    def _run(self):
        print("episode: {}, replay count: {}, loss: {:.2}, e: {:.2}, b: {:.2}".format(
            self.agent.episode, self.agent.replay_count, self.agent.loss_value, self.agent.epsilon,
            self.agent.priority_b))


class TrueCompute(Callback):

    def __init__(self, condition, agent, env):
        super().__init__(condition, agent, env)
        self.Q_table = None
        self.policy = None
        self.V_table = None

    def _run(self):
        true_Q_table, true_policy = get_true_Q_table(self.env, self.agent.gamma)
        true_V = q_to_v(self.env, true_Q_table)

        self.Q_table = true_Q_table
        self.policy = true_policy
        self.V_table = true_V


class QCompute(Callback):

    def __init__(self, condition, agent, env):
        super().__init__(condition, agent, env)
        self.Q_table = None
        self.policy = None
        self.V_table = None

    def _run(self):
        Q_table = compute_q_table(self.env, self.agent)
        policy = q_to_policy_RM(self.env, Q_table)
        V = q_to_v(self.env, Q_table)

        self.Q_table = Q_table
        self.policy = policy
        self.V_table = V


class QErrorMonitor(Callback):

    def __init__(self, condition, agent, env, true_compute, q_compute):
        super().__init__(condition, agent, env)
        self.replays = []
        self.errors_Q_table = []
        self.errors_V_table = []
        self.errors_policy = []

        self.true = true_compute
        self.q = q_compute

        borders = np.reshape(np.arange(env.T * env.C * env.nA), (env.T, env.C, env.nA))
        borders = ((borders // env.nA + 1) % 4 == 0) | (borders >= (env.T - 1) * env.C * env.nA)
        self.mask_borders = borders.reshape((env.T * env.C, env.nA))
        self.mask_not_borders = np.logical_not(self.mask_borders)

        self.errors_borders = []
        self.errors_not_borders = []

    def _run(self):

        true_Q_table = self.true.Q_table
        true_V_table = self.true.V_table
        true_policy = self.true.policy
        if true_Q_table is None:
            return

        Q_table = self.q.Q_table
        V_table = self.q.V_table
        policy = self.q.policy
        if Q_table is None:
            return

        T, C, nA = self.env.T, self.env.C, self.env.action_space.n

        self.replays.append(self.agent.replay_count)
        self.errors_Q_table.append(self._mse(true_Q_table, Q_table))
        self.errors_V_table.append(self._mse(true_V_table, V_table))
        self.errors_policy.append(self._mse(true_policy.reshape(T, C), policy.reshape(T, C)))

        self.errors_borders.append(self._mse(true_Q_table[self.mask_borders], Q_table[self.mask_borders]))
        self.errors_not_borders.append(self._mse(true_Q_table[self.mask_not_borders], Q_table[self.mask_not_borders]))

        print("Difference with the true Q-table")
        print(abs(true_Q_table.reshape(T, C, nA) - Q_table.reshape(T, C, nA)))

        print("Difference with the true V-table")
        print(abs(true_V_table.reshape(T, C) - V_table.reshape(T, C)))

        print("Difference with the true Policy")
        print(abs(true_policy.reshape(T, C) - policy.reshape(T, C)))

    def _mse(self, A, B):
        return np.sqrt(np.square(A.flatten() - B.flatten()).sum())


class QErrorDisplay(Callback):

    def __init__(self, condition, agent, env, q_error):
        super().__init__(condition, agent, env)
        self.q_error = q_error

    def _run(self):
        plt.figure()
        plt.plot(self.q_error.replays, self.q_error.errors_Q_table, '-o')
        plt.xlabel("Epochs")
        plt.ylabel("Difference with the true Q-table")
        plt.show()

        plt.figure()
        plt.plot(self.q_error.replays, self.q_error.errors_V_table, '-o')
        plt.xlabel("Epochs")
        plt.ylabel("Difference with the true V-table")
        plt.show()

        plt.plot(self.q_error.replays, self.q_error.errors_policy, '-o')
        plt.xlabel("Epochs")
        plt.ylabel("Difference with the policy")
        plt.show()

        plt.figure()
        plt.plot(self.q_error.replays, self.q_error.errors_borders, self.q_error.replays,
                 self.q_error.errors_not_borders)
        plt.legend(["Error at the borders", "Error not at the borders"])
        plt.xlabel("Epochs")
        plt.ylabel("Difference of each coefficient of the Q-table with the true Q-table")
        plt.show()


class RevenueMonitor(Callback):

    def __init__(self, condition, agent, env, q_compute, N):
        super().__init__(condition, agent, env)
        self.replays = []
        self.revenues = []
        self.q = q_compute
        self.N = N

    def _run(self):
        policy = self.q.policy
        if policy is None:
            return

        revenue = average_n_episodes(self.env, policy.flatten(), self.N)
        self.replays.append(self.agent.replay_count)
        self.revenues.append(revenue)

        print("Average reward over {} episodes after {} replay : {}".format(self.N, self.agent.replay_count, revenue))


class RevenueDisplay(Callback):

    def __init__(self, condition, agent, env, revenue_compute, reference=None):
        super().__init__(condition, agent, env)
        self.revenue = revenue_compute
        self.reference = reference

    def _run(self):
        plt.figure()
        plt.plot(self.revenue.replays, self.revenue.revenues, '-o')
        plt.xlabel("Epochs")
        plt.ylabel("Average revenue")
        plt.title("Average revenue over {} episodes".format(self.revenue.N))
        if self.reference is not None:
            X = self.revenue.replays[0], self.revenue.replays[-1]
            Y = [self.reference.revenues[0]] * 2
            plt.plot(X, Y, c='red', lw=3)
        plt.show()


class VDisplay(Callback):

    def __init__(self, condition, agent, env, q_compute):
        super().__init__(condition, agent, env)
        self.q = q_compute

    def _run(self):
        V_table = self.q.V_table
        if V_table is not None:
            visualisation_value_RM(V_table, self.env.T, self.env.C)


class PolicyDisplay(Callback):

    def __init__(self, condition, agent, env, q_compute):
        super().__init__(condition, agent, env)
        self.q = q_compute

    def _run(self):
        policy = self.q.policy
        if policy is not None:
            visualize_policy_RM(policy, self.env.T, self.env.C)


class MemoryMonitor(Callback):

    def __init__(self, condition, agent, env):
        super().__init__(condition, agent, env)
        self.replays = []
        self.hist2d = []

    def _run(self):
        self.replays.append(self.agent.replay_count)
        states = []
        actions = []
        for k in range(len(agent.memory)):
            states.append(env.to_idx(agent.memory[k][0][0][0], agent.memory[k][0][0][1]))
            actions.append(agent.memory[k][1])
        h, xedges, yedges = np.histogram2d(states, actions, bins=[max(states) + 1, env.action_space.n])
        self.hist2d.append((h, xedges, yedges))


class MemoryDisplay(Callback):

    def __init__(self, condition, agent, env, memory_monitor):
        super().__init__(condition, agent, env)
        self.memory_monitor = memory_monitor

    def _run(self):
        fig, ax = plt.subplots(tight_layout=True)
        h, xedges, yedges = self.memory_monitor.hist2d[-1]
        im = plt.pcolormesh(xedges, yedges, h.T)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])
        plt.xlabel("State index")
        plt.ylabel("Action index")
        plt.title("Transitions present in the agent's memory")
        plt.colorbar(im, ax=ax)
        plt.show()

class BatchMonitor(Callback):

    def __init__(self, condition, agent, env):
        super().__init__(condition, agent, env)
        self.replays = []
        self.hist2d = []

    def _run(self):
        self.replays.append(self.agent.replay_count)
        states = []
        actions = []
        for k in range(len(list(agent.last_visited))):
            states.append(env.to_idx(agent.memory[k][0][0][0], agent.memory[k][0][0][1]))
            actions.append(agent.memory[k][1])
        h, xedges, yedges = np.histogram2d(states, actions, bins=[max(states) + 1, env.action_space.n])
        self.hist2d.append((h, xedges, yedges))


class BatchDisplay(Callback):

    def __init__(self, condition, agent, env, batch_monitor):
        super().__init__(condition, agent, env)
        self.batch_monitor = batch_monitor

    def _run(self):
        fig, ax = plt.subplots(tight_layout=True)
        h, xedges, yedges = self.batch_monitor.hist2d[-1]
        im = plt.pcolormesh(xedges, yedges, h.T)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])
        plt.xlabel("State index")
        plt.ylabel("Action index")
        plt.title("Transitions present in the agent's minibatch")
        plt.colorbar(im, ax=ax)
        plt.show()


def train(agent, nb_episodes, callbacks):
    for episode in range(nb_episodes):

        state = env.set_random_state()
        # state = env.reset()

        done = False

        while not done:
            action_idx = agent.act(state)
            next_state, reward, done, _ = env.step(env.A[action_idx])

            agent.remember(state, action_idx, reward, next_state, done)

            state = next_state

            agent.replay(episode)

        agent.update_epsilon()

        for callback in callbacks:
            callback.run(episode)


def init_target_network_with_true_Q_table(agent, env):
    init_network_with_true_Q_table(agent, env)
    # Reset main model
    agent.model = agent._build_model()
    # And make sure the target model is never updated
    agent.target_model_update = sys.maxsize


def init_network_with_true_Q_table(agent, env):
    init_with_V(agent, env)


def compute_state_weights(env):
    shape = [space.n for space in env.observation_space]
    compute_weight = lambda x: 1 + max(1. * x[0] / env.T, 1. * x[1] / env.C)
    state_weights = [((t, x), compute_weight((t, x))) for t in range(shape[0]) for x in range(shape[1])]

    return dict(state_weights)


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

    nb_episodes = 5000

    agent = DQNAgent(state_size, action_size,
                     # state_scaler=env.get_state_scaler(), value_scaler=env.get_value_scaler(),
                     replay_method="DDQL", batch_size=30, memory_size=5000,
                     prioritized_experience_replay=False,
                     hidden_layer_size=50, dueling=False, loss=mean_squared_error, learning_rate=0.001,
                     epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9995,
                     state_weights=compute_state_weights(env))
    # init_target_network_with_true_Q_table(agent, env)
    # init_network_with_true_Q_table(agent, env)

    before_train = lambda episode: episode == 0
    every_episode = lambda episode: True
    while_training = lambda episode: episode % (nb_episodes / 20) == 0
    after_train = lambda episode: episode == nb_episodes - 1
    while_training_after_replay_has_started = lambda episode: len(agent.memory) > agent.batch_size and episode % (nb_episodes / 10) == 0

    true_compute = TrueCompute(before_train, agent, env)
    true_v_display = VDisplay(before_train, agent, env, true_compute)
    true_revenue = RevenueMonitor(before_train, agent, env, true_compute, 10_000)

    agent_monitor = AgentMonitor(every_episode, agent, env)

    q_compute = QCompute(while_training, agent, env)
    v_display = VDisplay(after_train, agent, env, q_compute)
    policy_display = PolicyDisplay(after_train, agent, env, q_compute)

    q_error = QErrorMonitor(while_training, agent, env, true_compute, q_compute)
    q_error_display = QErrorDisplay(after_train, agent, env, q_error)

    revenue_compute = RevenueMonitor(while_training, agent, env, q_compute, 10_000)
    revenue_display = RevenueDisplay(after_train, agent, env, revenue_compute, true_revenue)

    memory_monitor = MemoryMonitor(while_training, agent, env)
    memory_display = MemoryDisplay(after_train, agent, env, memory_monitor)

    batch_monitor = BatchMonitor(while_training_after_replay_has_started, agent, env)
    batch_display = BatchDisplay(after_train, agent, env, batch_monitor)

    callbacks = [true_compute, true_v_display, true_revenue,
                 agent_monitor,
                 q_compute, v_display, policy_display,
                 q_error, q_error_display,
                 revenue_compute, revenue_display,
                 memory_monitor, memory_display,
                 batch_monitor, batch_display]

    train(agent, nb_episodes, callbacks)

    # X, Y, Z, values = reshape_matrix_of_visits(agent.M, env)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # p = ax.scatter3D(X, Y, Z, c=values, cmap='hot')
    # fig.colorbar(p, ax=ax)
    # plt.show()
