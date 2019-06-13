import gym
import numpy as np
import random
from gym import spaces
from gym.utils import seeding
import scipy.special
import matplotlib.pyplot as plt

default_data_collection_points = 500
default_capacity = 50
default_actions = tuple(k for k in range(50, 231, 20))
default_alpha = 0.66
default_lambda = 0.2
default_micro_times = 20


class RMDCPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_collection_points=default_data_collection_points, capacity=default_capacity,
                 micro_times=default_micro_times, actions=default_actions,
                 alpha=default_alpha, lamb=default_lambda):

        super(RMDCPEnv, self).__init__()

        self.T = data_collection_points
        self.C = capacity
        self.M = micro_times
        self.nS = data_collection_points * capacity  # number of states

        self.A = actions
        self.nA = len(self.A)  # number of actions

        self.alpha = alpha
        self.lamb = lamb

        self.observation_space = spaces.Tuple((spaces.Discrete(self.T), spaces.Discrete(self.C)))
        self.action_space = spaces.Discrete(self.nA)
        self.reward_range = [min(self.A), max(self.A)]

        self.seed()

        self.s = (0, 0)

        self.P = self.init_transitions(self.T, self.C, self.A)

    def init_transitions(self, T, C, A):

        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        P = {(t, x): {a: [] for a in A} for t in range(T) for x in range(C)}
        for t in range(T):
            for x in range(C):
                s = (t, x)
                for a in A:
                    P[s][a] = self.transitions(s, a)
        return P

    def transitions(self, state, action):
        list_transitions = []
        t, x = state[0], state[1]
        done = False
        if t == self.T - 1 or x == self.C - 1:
            list_transitions.append((1, state, 0, True))
        else:
            k = 0
            while k <= self.M:
                proba_buy, reward = self.proba_buy(action)
                proba_next_state = ((1 - proba_buy) ** (self.M - k)) * (proba_buy ** k) * scipy.special.binom(
                    self.M, k)
                total_reward = k * reward
                new_t, new_x = t + 1, x + k
                new_state = (new_t, new_x)
                if new_t == self.T - 1 or new_x == self.C - 1:
                    done = True

                # When we reach maximum capacity before the end of the DCP
                # The remaining probabilities go to the last valid transition
                if new_x == self.C - 1:
                    remaining_proba = 0.
                    while k + 1 <= self.M:
                        k = k + 1
                        remaining_proba += ((1 - proba_buy) ** (self.M - k)) * (proba_buy ** k) * scipy.special.binom(
                            self.M, k)

                    proba_next_state += remaining_proba

                list_transitions.append((proba_next_state, new_state, total_reward, done))
                k = k + 1

        return list_transitions

    def to_coordinate(self, state_idx):
        t = int(int(state_idx) / self.C)
        x = int(state_idx - t * self.C)
        return t, x

    def to_idx(self, t, x):
        return t * self.C + x

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = (0, 0)

        return self.s

    def set_random_state(self):
        self.s = self.observation_space.sample()
        while self.s[0] == self.T - 1 or self.s[1] == self.C - 1:
            self.s = self.observation_space.sample()

        return self.s

    def step(self, a):
        transitions = self.transitions(self.s, a)
        transition_idx = self.categorical_sample([t[0] for t in transitions])
        p, s, r, d = transitions[transition_idx]
        self.s = s
        return s, r, d, {"prob": p}

    def proba_buy(self, a):
        """Returns:
            - the probability that a person will buy the ticket at the price p
            - the reward that the agent gets if the person buys the ticket"""
        proba = self.lamb * np.exp(-self.alpha * ((a / self.A[0]) - 1))
        reward = a
        return proba, reward

    def proba_not_buy(self, a):
        """Returns:
            - the probability that a person will not buy the ticket at the price p
            - the reward that the agent gets if the person does not buy the ticket"""
        reward = 0
        p, r = self.proba_buy(a)
        return 1 - p, reward

    def categorical_sample(self, prob_n):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > self.np_random.rand()).argmax()

    def get_state_scaler(self):

        return StateScaler(self.T, self.C)

    def get_value_scaler(self):

        return ValueScaler(self.A, self.C)


class StateScaler(object):
    def __init__(self, T, C):
        super(StateScaler, self).__init__()
        self.scale_time = 2. / (T - 1)
        self.scale_capacity = 2. / (C - 1)

    def scale(self, state):
        dcp, capacity = state

        return self.scale_time * dcp - 1, self.scale_capacity * capacity - 1

    def unscale(self, state):
        dcp, capacity = state

        return int((dcp + 1) / self.scale_time), int((capacity + 1) / self.scale_capacity)


class ValueScaler(object):
    def __init__(self, A, C):
        super(ValueScaler, self).__init__()
        self.scale_value = 2. / C * max(A)

    def scale(self, value):
        return self.scale_value * value - 1

    def unscale(self, value):
        return (value + 1) / self.scale_value