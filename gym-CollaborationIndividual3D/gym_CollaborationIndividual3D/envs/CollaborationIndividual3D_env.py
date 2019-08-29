import gym
import numpy as np
from gym import spaces
from gym.envs.toy_text import discrete

default_capacity1 = 50
default_capacity2 = 50
default_actions = tuple(k for k in range(50, 231, 20))
default_micro_times = 20


class CollaborationIndividual3DEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, capacity1=default_capacity1, capacity2=default_capacity2,
                 micro_times=default_micro_times, actions=default_actions):
        self.T = micro_times
        self.C1 = capacity1
        self.C2 = capacity2
        self.nS = self.T * self.C1 * self.C2  # number of states

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

        super(CollaborationIndividual3DEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def init_transitions(self):
        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        return P

    def to_coordinate(self, state_idx):
        t = int(int(state_idx) / (self.C1 * self.C2))
        x1 = int(int(state_idx - self.C1 * self.C2 * t) / self.C1)
        x2 = int(state_idx - self.C1 * x1 - self.C1 * self.C2 * t)
        return t, x1, x2

    def to_idx(self, t, x1, x2):
        return x2 + x1 * self.C1 + self.C1 * self.C2 * t
