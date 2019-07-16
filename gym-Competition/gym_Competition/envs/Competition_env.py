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
                    state_idx = self.to_idx(t, x1, x2)
                    for action_idx in range(self.nA):
                        P[state_idx][action_idx] = self.transitions(state_idx, action_idx)
        return P

    def transitions(self, state_idx, action_idx):
        list_transitions = []
        t, x1, x2 = self.to_coordinate(state_idx)
        action = self.A[action_idx]
        a1, a2 = action[0], action[1]
        done = False
        if t == self.T - 1 or (x1 == self.C1 - 1 and x2 == self.C2 - 1):
            list_transitions.append((1, state_idx, 0, True))
        else:
            Utilities = [0, self.k_airline1 - self.beta * a1, self.k_airline2 - self.beta * a2]
            probas_logit = self.compute_probas_logit(Utilities)

            # Case no buy
            new_t, new_x1, new_x2 = t + 1, x1, x2
            reward1, reward2 = 0, 0
            proba_next_state = probas_logit[0]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            if new_t == self.T - 1 or (new_x1 == self.C1 - 1 and new_x2 == self.C2 - 1):
                done = True
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), done))

            # Case Airline1
            new_t, new_x1, new_x2 = t + 1, x1 + 1, x2
            reward1, reward2 = a1, 0
            proba_next_state = probas_logit[1]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            if new_t == self.T - 1 or (new_x1 == self.C1 - 1 and new_x2 == self.C2 - 1):
                done = True
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), done))

            # Case Airline2
            new_t, new_x1, new_x2 = t + 1, x1, x2 + 1
            reward1, reward2 = 0, a2
            proba_next_state = probas_logit[2]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            if new_t == self.T - 1 or (new_x1 == self.C1 - 1 and new_x2 == self.C2 - 1):
                done = True
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), done))

        return list_transitions

    def compute_probas_logit(self, representative_utilities):
        """
            Input: List of the representative utilities of the different alternatives
            Output: List of the probabilities to choose each of the alternatives
        """
        numerators = np.exp([k for k in representative_utilities])
        normalization = sum(numerators)

        return numerators / normalization

    def to_coordinate(self, state_idx):
        t = int(int(state_idx) / (self.C1 * self.C2))
        print(t)
        x2 = int(int(state_idx - self.C1 * self.C2 * t) / self.C1)
        x1 = int(state_idx - self.C1 * x2 - self.C1 * self.C2 * t)
        return t, x1, x2

    def to_idx(self, t, x1, x2):
        return x1 + x2 * self.C1 + self.C1 * self.C2 * t

    def set_random_state(self):
        self.s = self.observation_space.sample()
        t, x1, x2 = self.to_coordinate(self.s)
        while t == self.T - 1 or (x1 == self.C1 - 1 and x2 == self.C2 - 1):
            self.s = self.observation_space.sample()
            t, x1, x2 = self.to_coordinate(self.s)

        return self.s

