import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import scipy.special

default_micro_times = 500
default_capacity = 50
default_actions = tuple(k for k in range(50, 231, 20))
default_alpha = 0.66
default_lambda = 0.2


class RMEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, micro_times=default_micro_times, capacity=default_capacity, actions=default_actions,
                 alpha=default_alpha, lamb=default_lambda):

        super(RMEnv, self).__init__()

        self.T = micro_times
        self.C = capacity
        self.nS = micro_times * capacity  # number of states

        self.A = actions
        self.nA = len(self.A)  # number of actions

        self.alpha = alpha
        self.lamb = lamb

        self.observation_space = spaces.Tuple((spaces.Discrete(self.T), spaces.Discrete(self.C)))
        self.action_space = spaces.Discrete(self.nA)
        self.reward_range = [min(self.A), max(self.A)]

        self.seed()

        self.s = (0, 0)

        self.P = self.init_transitions(self.T, self.C, self.A, self.nA, self.nS)

    def init_transitions(self, T, C, A, nA, nS):

        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        P = {(t, x): {a: [] for a in A} for t in range(T) for x in range(C)}

        # Filling the transitions dictionnary P
        for t in range(T):
            for x in range(C):
                s = (t, x)
                for a in A:
                    li = P[s][a]
                    if t == T - 1 or x == C - 1:  # Terminal states, the game ends
                        li.append((1.0, s, 0, True))
                    else:
                        for b in range(
                                2):  # If the agent is in a state s with the action a then there are two possible states where he might end in
                            if b == 0:  # The person buys the ticket
                                new_t, new_x = self.inc_buy(t, x)
                                new_state = (new_t, new_x)
                                p, r = self.proba_buy(a)
                                done = False
                                if t == T - 2 or x == C - 2:
                                    done = True
                                li.append((p, new_state, r, done))
                            else:  # The person does not buy the ticket
                                new_t, new_x = self.inc_not_buy(t, x)
                                new_state = (new_t, new_x)
                                p, r = self.proba_not_buy(a)
                                done = False
                                if t == T - 2:
                                    done = True
                                li.append((p, new_state, r, done))

        return P

    def transitions(self, state, action):
        list_transitions = []
        t, x = state[0], state[1]
        done = False
        if t == self.T - 1 or x == self.C - 1:
            list_transitions.append((1, state, 0, True))
        else:
            for k in range(2):
                proba_buy, _ = self.proba_buy(action)
                proba_next_state = ((1 - proba_buy) ** (1 - k)) * (proba_buy ** k) * scipy.special.binom(1, k)
                reward = k * action
                new_t, new_x = t + 1, x + k
                new_state = (new_t, new_x)
                if new_t == self.T - 1 or new_x == self.C - 1:
                    done = True
                if new_x > self.C - 1:
                    break
                list_transitions.append((proba_next_state, new_state, reward, done))

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
        # self.s = self.observation_space.sample()
        self.s = (0, 0)

        return self.s

    def set_random_state(self):
        self.s = self.observation_space.sample()

        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        transition_idx = self.categorical_sample([t[0] for t in transitions])
        p, s, r, d = transitions[transition_idx]
        self.s = s
        return s, r, d, {"prob": p}

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

    def categorical_sample(self, prob_n):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > self.np_random.rand()).argmax()
