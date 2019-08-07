import gym
import numpy as np
import random
from gym import spaces
from gym.envs.toy_text import discrete

default_capacity = 50
default_actions = tuple(k for k in range(50, 231, 20))
default_alpha = 0.66
default_lambda = 0.2
default_micro_times = 20


class RMDiscreteEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, capacity=default_capacity,
                 micro_times=default_micro_times, actions=default_actions,
                 alpha=default_alpha, lamb=default_lambda):

        self.T = micro_times
        self.C = capacity
        self.nS = micro_times * capacity  # number of states

        self.A = actions
        self.nA = len(self.A)  # number of actions

        self.alpha = alpha
        self.lamb = lamb

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.reward_range = [min(self.A), max(self.A)]

        self.seed()

        self.s = 0

        self.P = self.init_transitions()

        self.isd = np.zeros(self.nS, float)
        self.isd[0] = 1.

        super(RMDiscreteEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def init_transitions(self):

        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for t in range(self.T):
            for x in range(self.C):
                s = self.to_idx(t,x)
                for a in range(self.nA):
                    li = P[s][a]
                    if t == self.T - 1 or x == self.C - 1:  # Terminal states, the game ends
                        li.append((1.0, s, 0, True))
                    else:
                        for b in range(
                                2):  # If the agent is in a state s with the action a then there are two possible states where he might end in
                            if b == 0:  # The person buys the ticket
                                new_t, new_x = self.inc_buy(t, x)
                                new_state = self.to_idx(new_t, new_x)
                                p, r = self.proba_buy(self.A[a])
                                done = False
                                if t == self.T - 2 or x == self.C - 2:
                                    done = True
                                li.append((p, new_state, r, done))
                            else:  # The person does not buy the ticket
                                new_t, new_x = self.inc_not_buy(t, x)
                                new_state = self.to_idx(new_t, new_x)
                                p, r = self.proba_not_buy(self.A[a])
                                done = False
                                if t == self.T - 2:
                                    done = True
                                li.append((p, new_state, r, done))

        return P

    def to_coordinate(self, state_idx):
        t = int(int(state_idx) / self.C)
        x = int(state_idx - t * self.C)
        return t, x

    def to_idx(self, t, x):
        return t * self.C + x

    def inc_buy(self, t, x):
        """Returns the next state when the person buys the ticket"""
        # t = min(t + 1, self.T - 1)
        # x = min(x + 1, self.C - 1)
        t = t + 1
        x = x + 1
        return t, x

    def inc_not_buy(self, t, x):
        """Returns the next state when the person does not buys the ticket"""
        # t = min(t + 1, self.T - 1)
        t = t + 1
        return t, x

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

    def compute_weights(self):
        compute_weight = lambda x: 1 + max(1. * x[0] / self.T, 1. * x[1] / self.C)
        weights = [(s, compute_weight((self.to_coordinate(s)[0], self.to_coordinate(s)[1]))) for s in
                   range(self.T * self.C)]
        return weights


