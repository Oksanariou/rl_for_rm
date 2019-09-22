import gym
import numpy as np
from gym import spaces
from gym.envs.toy_text import discrete

default_capacity = 50
default_actions = tuple(k for k in range(50, 231, 20))
default_micro_times = 20


class CollaborationIndividual2DEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, capacity=default_capacity,
                 micro_times=default_micro_times, actions=default_actions):
        self.T = micro_times
        self.C = capacity
        self.nS = self.T * self.C  # number of states

        self.A = actions
        self.nA = len(self.A)  # number of actions

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.reward_range = [min(self.A), max(self.A)]

        self.seed()

        self.s = 0

        self.P = self.init_transitions()

        self.isd = np.zeros(self.nS, float)
        self.isd[0] = 1.

        super(CollaborationIndividual2DEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def init_transitions(self):
        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        return P

    def to_coordinate(self, state_idx):
        t = int(int(state_idx) / self.C)
        x = int(state_idx - t * self.C)
        return t, x

    def to_idx(self, t, x):
        return t * self.C + x
