import gym
import numpy as np
import random
from gym import spaces
from gym.envs.toy_text import discrete

default_capacity = 50
default_actions = tuple(k for k in range(50, 231, 20))
default_lambda = 0.2
default_micro_times = 20
default_beta = 0.015
default_nested_lamb = 1.


class CompetitionIndividual2DEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, capacity=default_capacity,
                 micro_times=default_micro_times, actions=default_actions, lamb=default_lambda, beta=default_beta, k=0,
                 nested_lamb=default_nested_lamb,
                 competition_aware=False,
                 competitor_policy=None, competitor_distribution=None):

        self.T = micro_times
        self.C = capacity
        self.nS = self.T * self.C  # number of states

        self.A = actions
        self.nA = len(self.A)  # number of actions

        self.lamb = lamb
        self.beta = beta
        self.k = k

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.reward_range = [min(self.A), max(self.A)]

        self.seed()

        self.s = 0

        self.nested_lamb = nested_lamb
        self.nest1 = {"lambda": 1, "representative_utilities": [0]}

        self.competitor_policy = competitor_policy
        self.competitor_distribution = competitor_distribution
        self.competition_aware = competition_aware

        self.P = self.init_transitions()

        self.isd = np.zeros(self.nS, float)
        self.isd[0] = 1.

        super(CompetitionIndividual2DEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def init_transitions(self):

        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for t in range(self.T):
            for x in range(self.C):
                s = self.to_idx(t, x)
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
                                p = self.proba_buy_not_alone(self.A[a],
                                                             t) if self.competition_aware else self.proba_buy_alone(
                                    self.A[a])
                                r = self.A[a]
                                done = False
                                if t == self.T - 2 or x == self.C - 2:
                                    done = True
                                li.append((p, new_state, r, done))
                            else:  # The person does not buy the ticket
                                new_t, new_x = self.inc_not_buy(t, x)
                                new_state = self.to_idx(new_t, new_x)
                                p = 1 - self.proba_buy_not_alone(self.A[a],
                                                                 t) if self.competition_aware else 1 - self.proba_buy_alone(
                                    self.A[a])
                                r = 0
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
        t = t + 1
        x = x + 1
        return t, x

    def inc_not_buy(self, t, x):
        """Returns the next state when the person does not buys the ticket"""
        t = t + 1
        return t, x

    def proba_buy_alone(self, a):
        proba = self.lamb * np.exp(self.k - self.beta * a) / (1 + np.exp(self.k - self.beta * a))
        return proba

    def compute_probas_nested_logit(self, nests):
        """
            Input: List of nests. Each nest is a dictionary with two elements: a parameter lambda and a list of representative utilities.
            Output: List of the size of the number of nests made of lists containing the probabilities of the alternatives of the corresponding nest
        """
        probas_all = []
        sum_nests = []
        nb_nests = len(nests)
        for k in range(nb_nests):
            representative_utilities = nests[k]["representative_utilities"]
            lamb = nests[k]["lambda"]
            sum_nests.append(sum(np.exp([i / lamb for i in representative_utilities])))
        for k in range(nb_nests):
            representative_utilities = nests[k]["representative_utilities"]
            lamb = nests[k]["lambda"]
            for representative_utility in representative_utilities:
                numerator = np.exp(representative_utility / lamb) * (sum_nests[k] ** (lamb - 1))
                normalization = np.sum(sum_nests[k] ** nests[k]["lambda"] for k in range(nb_nests))
                probas_all.append(numerator / normalization)
            # probas_all.append(probas_nest)

        return probas_all

    def proba_buy_not_alone(self, a, t):
        competitor_policy = self.competitor_policy.reshape(self.T, self.C)
        competitor_actions = competitor_policy[t][:]
        competitor_actions_distribution = self.competitor_distribution[t][:]
        proba = 0
        for i in range(len(competitor_actions)):
            competitor_action = competitor_actions[i]
            competitor_action_probability = competitor_actions_distribution[i]
            Utilities = [self.k - self.beta * a, self.k - self.beta * competitor_action]

            nest_2 = {}
            nest_2["lambda"] = self.nested_lamb
            nest_2["representative_utilities"] = Utilities
            nests = [self.nest1, nest_2]
            probas = self.compute_probas_nested_logit(nests)
            proba += probas[1] * competitor_action_probability

        return self.lamb * proba

    def compute_weights(self):
        compute_weight = lambda x: 1 + max(1. * x[0] / self.T, 1. * x[1] / self.C)
        weights = [(s, compute_weight((self.to_coordinate(s)[0], self.to_coordinate(s)[1]))) for s in
                   range(self.T * self.C)]
        return weights
