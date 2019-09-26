import gym
import numpy as np
import random
from gym import spaces
from gym.utils import seeding
import scipy.special
from gym.envs.toy_text import discrete
from scipy.stats import sem, t
import glob
import matplotlib.pyplot as plt

default_data_collection_points = 500
default_capacity = 50
default_actions = tuple(k for k in range(50, 231, 20))
default_alpha = 0.66
default_lambda = 0.2
default_micro_times = 20


class RMDCPDiscreteEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_collection_points=default_data_collection_points, capacity=default_capacity,
                 micro_times=default_micro_times, actions=default_actions,
                 alpha=default_alpha, lamb=default_lambda):

        self.T = data_collection_points
        self.C = capacity
        self.M = micro_times
        self.nS = data_collection_points * capacity  # number of states
        self.states = [k for k in range(self.nS)]

        self.A = actions
        self.nA = len(self.A)  # number of actions

        self.alpha = alpha
        self.lamb = lamb

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.reward_range = [min(self.A), max(self.A)]

        self.seed()

        self.s = (0, 0)

        self.P = self.init_transitions()

        self.isd = np.zeros(self.nS, float)
        self.isd[0] = 1.

        super(RMDCPDiscreteEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def init_transitions(self):

        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for t in range(self.T):
            for x in range(self.C):
                s = self.to_idx(t,x)
                for a in range(self.nA):
                    P[s][a] = self.transitions(s, a)
        return P

    def transitions(self, state, action):
        list_transitions = []
        t, x = self.to_coordinate(state)
        done = False
        if t == self.T - 1 or x == self.C - 1:
            list_transitions.append((1, state, 0, True))
        else:
            k = 0
            while k <= self.M:
                proba_buy, reward = self.proba_buy(self.A[action])
                proba_next_state = ((1 - proba_buy) ** (self.M - k)) * (proba_buy ** k) * scipy.special.binom(
                    self.M, k)
                total_reward = k * reward
                new_t, new_x = t + 1, x + k
                new_state = self.to_idx(new_t, new_x)
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


    def set_random_state(self):
        self.s = self.observation_space.sample()
        t, x = self.to_coordinate(self.s)
        while t == self.T - 1 or x == self.C - 1:
            self.s = self.observation_space.sample()
            t, x = self.to_coordinate(self.s)

        return self.s


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

    def get_state_scaler(self):

        return StateScaler(self.T, self.C)

    def get_value_scaler(self):

        return ValueScaler(self.A, self.C)

    def compute_weights(self):
        compute_weight = lambda x: 1 + max(1. * x[0] / self.T, 1. * x[1] / self.C)
        weights = [(s, compute_weight((self.to_coordinate(s)[0], self.to_coordinate(s)[1]))) for s in
                   range(self.T * self.C)]
        return weights

    def run_episode(self, policy):
        state = self.reset()
        total_reward = 0
        bookings = np.zeros(self.nA)
        while True:
            action_idx = policy[state]
            state, reward, done, _ = self.step(action_idx)
            if reward != 0:
                bookings[action_idx] += 1
            total_reward += reward
            if done:
                break
        return total_reward, bookings

    def average_n_episodes(self, policy, n_eval):
        """ Runs n episodes and returns the average of the n total rewards"""
        scores = [self.run_episode(policy) for _ in range(n_eval)]
        scores = np.array(scores)
        revenue = np.mean(scores[:, 0])
        bookings = np.mean(scores[:, 1], axis=0)
        return revenue, bookings

    def collect_list_of_mean_revenues_and_bookings(self, experience_name):
        list_of_rewards = []
        for np_name in glob.glob(str(experience_name) + '/*.np[yz]'):
            list_of_rewards.append(list(np.load(np_name, allow_pickle=True)))

        nb_collection_points = len(list_of_rewards[0])

        all_rewards_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]
        all_bookings_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]

        for k in range(len(list_of_rewards)):
            rewards = list_of_rewards[k]
            for i in range(nb_collection_points):
                all_rewards_combined_at_each_collection_point[i].append(rewards[i][0])
                all_bookings_combined_at_each_collection_point[i].append(rewards[i][1])

        mean_revenues = [np.mean(list) for list in all_rewards_combined_at_each_collection_point]
        mean_bookings = [np.mean(list, axis=0) for list in all_bookings_combined_at_each_collection_point]
        std_revenues = [sem(list) for list in all_rewards_combined_at_each_collection_point]
        confidence_revenues = [std_revenues[k] * t.ppf((1 + 0.95) / 2, nb_collection_points - 1) for k in
                               range(nb_collection_points)]
        min_revenues = [mean_revenues[k] - confidence_revenues[k] for k in range(nb_collection_points)]
        max_revenues = [mean_revenues[k] + confidence_revenues[k] for k in range(nb_collection_points)]

        return mean_revenues, min_revenues, max_revenues, mean_bookings



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



