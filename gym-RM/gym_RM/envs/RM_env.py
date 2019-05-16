from gym.envs.toy_text import discrete
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class RMEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self, T = 500, C = 50):
        self.T = T #Number of micro-times
        self.C = C #Total capacity

        A = [k for k in range(50, 231, 20)] #10 different prices
        nA = len(A) #number of actions
        nS = T*C #number of states

        #Initial state distribution, used in the reset function
        isd = np.zeros(nS, float)
        isd[0] = 1.

        P = {s: {a: [] for a in range(nA)} for s in range(nS)} #Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]

        def to_s(t, x):
            """Returns a state number"""
            return t*C + x

        def inc_buy(t, x):
            """Returns the next state when the person buys the ticket"""
            t = min(t+1, T-1)
            x = min(x+1, C-1)
            return (t, x)

        def inc_not_buy(t, x):
            """Returns the next state when the person does not buys the ticket"""
            t = min(t + 1, T - 1)
            return(t, x)

        def proba_buy(a):
            """Returns:
                - the probability that a person will buy the ticket at the price p
                - the reward that the agent gets if the person buys the ticket"""
            alpha = 0.66
            lamb = 0.2
            proba = lamb*np.exp(-alpha*((a/A[0])-1))
            reward = a
            return proba, reward

        def proba_not_buy(a):
            """Returns:
                - the probability that a person will not buy the ticket at the price p
                - the reward that the agent gets if the person does not buy the ticket"""
            reward = 0
            p, r = proba_buy(a)
            return 1-p, reward

        #Filling the transitions dictionnary P
        for t in range(T):
            for x in range(C):
                s = to_s(t, x)
                for k in range(nA):
                    a = A[k]
                    li = P[s][k]
                    if t == T-1 or x == C-1: #Terminal states, the game ends
                        li.append((1.0, s, 0, True))
                    else:
                        for b in range(2): #If the agent is in a state s with the action a then there are two possible states where he might end in
                            if b == 0: #The person buys the ticket
                                new_t, new_x = inc_buy(t, x)
                                new_state = to_s(new_t, new_x)
                                p, r = proba_buy(a)
                                done = False
                                li.append((p, new_state, r, done))
                            else: #The person does not buy the ticket
                                new_t, new_x = inc_not_buy(t, x)
                                new_state = to_s(new_t, new_x)
                                p, r = proba_not_buy(a)
                                done = False
                                li.append((p, new_state, r, done))

        super(RMEnv, self).__init__(nS, nA, P, isd)