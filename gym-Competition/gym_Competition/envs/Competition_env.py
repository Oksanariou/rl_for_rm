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
default_lambda = 0.8


class CompetitionEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, micro_times=default_micro_times, capacity_airline_1=default_capacity_airline_1,
                 capacity_airline_2=default_capacity_airline_2, actions=default_actions,
                 beta=default_beta, k_airline1=default_k_airline1, k_airline2=default_k_airline2, lamb=default_lambda):

        self.T = micro_times
        self.C1 = capacity_airline_1
        self.C2 = capacity_airline_2
        self.nS = micro_times * capacity_airline_1 * capacity_airline_2  # number of states

        self.A = actions
        self.nA = len(self.A)  # number of actions

        self.beta = beta
        self.k_airline1 = k_airline1
        self.k_airline2 = k_airline2

        self.lamb = lamb

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
        done1, done2 = False, False

        if t == self.T - 1 or (x1 == self.C1 - 1 and x2 == self.C2 - 1):
            list_transitions.append((1, state_idx, (0,0), (True, True)))

        elif x1 == self.C1 - 1: #Airline1 has sold all its tickets but Airline2 has not
            done1, reward1, new_x1 = True, 0, x1
            if t + 1 == self.T - 1:
                done2 = True
            Utilities = [0, self.k_airline2 - self.beta * a2]
            probas_logit = self.compute_probas_logit(Utilities)
            probas_buy = self.compute_proba_buy(probas_logit)

            # Case no buy
            new_t, new_x2 = t + 1, x2
            reward2 = 0
            proba_next_state = probas_buy[0]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), (done1, done2)))

            # Case buys Airline2
            new_t, new_x2 = t + 1, x2 + 1
            reward2 = a2
            proba_next_state = probas_buy[1]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            if new_x2 == self.C2 - 1:
                done2 = True
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), (done1, done2)))

        elif x2 == self.C2 - 1: #Airline2 has sold all its tickets but Airline1 has not
            done2, reward2, new_x2 = True, 0, x2
            if t + 1 == self.T - 1:
                done1 = True
            Utilities = [0, self.k_airline1 - self.beta * a1]
            probas_logit = self.compute_probas_logit(Utilities)
            probas_buy = self.compute_proba_buy(probas_logit)

            # Case no buy
            new_t, new_x1 = t + 1, x1
            reward1 = 0
            proba_next_state = probas_buy[0]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), (done1, done2)))

            # Case buys Airline1
            new_t, new_x1 = t + 1, x1 + 1
            reward1 = a1
            proba_next_state = probas_buy[1]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            if new_x1 == self.C1 - 1:
                done1 = True
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), (done1, done2)))

        else:
            if t + 1 == self.T - 1:
                done1, done2 = True, True
            Utilities = [0, self.k_airline1 - self.beta * a1, self.k_airline2 - self.beta * a2]
            probas_logit = self.compute_probas_logit(Utilities)
            probas_buy = self.compute_proba_buy(probas_logit)

            # Case no buy
            new_t, new_x1, new_x2 = t + 1, x1, x2
            reward1, reward2 = 0, 0
            proba_next_state = probas_buy[0]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), (done1, done2)))

            # Case buys Airline1
            new_t, new_x1, new_x2 = t + 1, x1 + 1, x2
            reward1, reward2 = a1, 0
            proba_next_state = probas_buy[1]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            new_done1 = True if new_x1 == self.C1 - 1 else done1
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), (new_done1, done2)))

            # Case buys Airline2
            new_t, new_x1, new_x2 = t + 1, x1, x2 + 1
            reward1, reward2 = 0, a2
            proba_next_state = probas_buy[2]
            new_state = self.to_idx(new_t, new_x1, new_x2)
            new_done2 = True if new_x2 == self.C2 - 1 else done2
            list_transitions.append((proba_next_state, new_state, (reward1, reward2), (done1, new_done2)))

        return list_transitions


    def compute_probas_logit(self, representative_utilities):
        """
            Input: List of the representative utilities of the different alternatives
            Output: List of the probabilities to choose each of the alternatives
        """
        numerators = np.exp([k for k in representative_utilities])
        normalization = sum(numerators)

        return numerators / normalization

    def compute_proba_buy(self, probas_logit):
        probas = [0]
        for k in range(1, len(probas_logit)):
            probas.append(self.lamb * probas_logit[k])
        proba_nogo = 1 - self.lamb * (sum(probas_logit[1:]))
        probas[0] = proba_nogo

        return probas


    def to_coordinate(self, state_idx):
        t = int(int(state_idx) / (self.C1 * self.C2))
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

