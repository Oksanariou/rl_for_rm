import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete
import scipy.special

default_micro_times = 500
default_capacity_airline_1 = 50
default_capacity_airline_2 = 20
default_actions = tuple((k, m) for k in range(50, 231, 20) for m in range(50, 231, 20))
default_beta = 0.001
default_k_airline1 = 1.5
default_k_airline2 = 1.5


class CompetitionEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, micro_times=default_micro_times, capacity_airline_1=default_capacity_airline_1,
                 capacity_airline_2=default_capacity_airline_2, actions=default_actions,
                 beta=default_beta, k_airline1=default_k_airline1, k_airline2=default_k_airline2):

        self.T = micro_times
        self.C1 = capacity_airline_1
        self.C2 = capacity_airline_2
        self.nS = micro_times * capacity_airline_1 * capacity_airline_2  # number of states

        self.A = actions
        self.nA = len(self.A)  # number of actions

        self.beta = beta
        self.k_airline1 = k_airline1
        self.k_airline2 = k_airline2

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        # self.reward_range = [min(self.A), max(self.A)]

        self.seed()

        self.s = 0

        self.P = self.init_transitions()

        self.isd = np.zeros(self.nS, float)
        self.isd[0] = 1.

        super(CompetitionEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def init_transitions(self):

        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for t in range(self.T):
            for x1 in range(self.C1):
                for x2 in range(self.C2):
                    s = self.to_idx(t, x1, x2)
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
        t = int(int(state_idx) / self.C1 * self.C2)
        x2 = int(int(state_idx - self.C1 * self.C2 * t) / self.C1)
        x1 = int(state_idx - self.C1 * x2 - self.C1 * self.C2 * t)
        return t, x1, x2

    def to_idx(self, t, x1, x2):
        return x1 + x2 * self.C1 + self.C1 * self.C2 * t

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
