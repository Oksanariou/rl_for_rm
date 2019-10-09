import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from scipy.stats import sem, t
import glob
import itertools
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
default_nested_lambda = 1.
default_prices_flight1 = [k for k in range(50, 231, 20)]
default_prices_flight2 = [k for k in range(50, 231, 20)]


class CollaborationGlobal3DMultiDiscreteEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, micro_times=default_micro_times, capacity1=default_capacity_airline_1,
                 capacity2=default_capacity_airline_2, prices=[default_prices_flight1, default_prices_flight2],
                 beta=default_beta, k_airline1=default_k_airline1, k_airline2=default_k_airline2, lamb=default_lambda,
                 nested_lamb=default_nested_lambda):

        super(CollaborationGlobal3DMultiDiscreteEnv, self).__init__()

        self.T = micro_times
        self.C1 = capacity1
        self.C2 = capacity2
        self.C = self.C1 + self.C2
        self.nS = self.T * self.C1 * self.C2  # number of states
        self.states = [[t, x1, x2] for t in range(self.T) for x1 in range(self.C1) for x2 in range(self.C2)]
        self.states1 = [[t, x1] for t in range(self.T) for x1 in range(self.C1)]
        self.states2 = [[t, x2] for t in range(self.T) for x2 in range(self.C2)]

        self.prices = prices
        self.prices_flight1 = self.prices[0]
        self.prices_flight2 = self.prices[1]
        self.A = list(itertools.product(self.prices[0], self.prices[1]))
        self.nA = len(self.A)  # number of actions

        self.beta = beta
        self.k_airline1 = k_airline1
        self.k_airline2 = k_airline2

        self.lamb = lamb

        self.observation_space = spaces.MultiDiscrete([self.T, self.C1, self.C2])
        self.action_space = spaces.Discrete(self.nA)
        # self.reward_range = [min(self.A), max(self.A)]

        self.seed()

        self.s = [0, 0, 0]

        self.nested_lamb = nested_lamb
        self.nest1 = {"lambda": 1, "representative_utilities": [0]}

        self.probas_from_two_prices = self.build_probas_from_two_prices()
        self.probas_from_one_price_flight1 = self.build_probas_from_one_price(1)
        self.probas_from_one_price_flight2 = self.build_probas_from_one_price(2)

        self.P, self.proba_cumsum = self.init_transitions()

        self.isd = np.zeros(self.nS, float)
        self.isd[0] = 1.

    def build_probas_from_two_prices(self):
        probas_from_two_prices = []
        for action_couple in self.A:
            Utilities = [0, self.k_airline1 - self.beta * action_couple[0],
                         self.k_airline2 - self.beta * action_couple[1]]

            nest_2 = {}
            nest_2["lambda"] = self.nested_lamb
            nest_2["representative_utilities"] = Utilities[1:]
            nests = [self.nest1, nest_2]
            probas = self.compute_probas_nested_logit(nests)
            probas_from_two_prices.append(self.compute_proba_buy(probas))

        return probas_from_two_prices

    def build_probas_from_one_price(self, flight):
        probas_from_one_price = []
        k = self.k_airline2
        prices = self.prices_flight2
        if flight == 1:
            k = self.k_airline1
            prices = self.prices_flight1
        for price in prices:
            Utilities = [0, k - self.beta * price]

            nest_2 = {}
            nest_2["lambda"] = self.nested_lamb
            nest_2["representative_utilities"] = Utilities[1:]
            nests = [self.nest1, nest_2]
            probas = self.compute_probas_nested_logit(nests)
            probas_from_one_price.append(self.compute_proba_buy(probas))
        return probas_from_one_price

    def init_transitions(self):
        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        proba_cumsum = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for t in range(self.T):
            for x1 in range(self.C1):
                for x2 in range(self.C2):
                    state_idx = self.to_idx(t, x1, x2)
                    for action_idx in range(self.nA):
                        transitions = self.transitions(state_idx, action_idx)
                        P[state_idx][action_idx] = transitions
                        prob_n = np.asarray([t[0] for t in transitions])
                        csprob_n = np.cumsum(prob_n)
                        proba_cumsum[state_idx][action_idx] = csprob_n
        return P, proba_cumsum

    def transitions(self, state_idx, action_idx):
        list_transitions = []
        t, x1, x2 = self.to_coordinate(state_idx)
        action = self.A[action_idx]
        a1, a2 = action[0], action[1]

        done1, done2 = False, False

        if t == self.T - 1 or (x1 == self.C1 - 1 and x2 == self.C2 - 1):
            list_transitions.append((1, state_idx, (0, 0), (True, True)))

        elif x1 == self.C1 - 1:  # Airline1 has sold all its tickets but Airline2 has not
            done1, reward1, new_x1 = True, 0, x1
            if t + 1 == self.T - 1:
                done2 = True

            probas_buy = self.probas_from_one_price_flight2[self.prices_flight2.index(a2)]

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

        elif x2 == self.C2 - 1:  # Airline2 has sold all its tickets but Airline1 has not
            done2, reward2, new_x2 = True, 0, x2
            if t + 1 == self.T - 1:
                done1 = True

            probas_buy = self.probas_from_one_price_flight1[self.prices_flight1.index(a1)]

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

            probas_buy = self.probas_from_two_prices[action_idx]

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

    def compute_proba_buy(self, probas_logit):
        probas = [0]
        for k in range(1, len(probas_logit)):
            probas.append(self.lamb * probas_logit[k])
        proba_nogo = 1 - self.lamb * (sum(probas_logit[1:]))
        probas[0] = proba_nogo

        return probas

    def to_coordinate(self, state_idx):
        t = int(int(state_idx) / (self.C1 * self.C2))
        x1 = int(int(state_idx - self.C1 * self.C2 * t) / self.C1)
        x2 = int(state_idx - self.C1 * x1 - self.C1 * self.C2 * t)
        return t, x1, x2

    def to_idx(self, t, x1, x2):
        return x2 + x1 * self.C1 + self.C1 * self.C2 * t

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = [0, 0, 0]

        return self.s

    def categorical_sample(self, csprob_n):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        return (csprob_n > self.np_random.rand()).argmax()

    def step(self, a):
        # transitions = self.transitions(self.s, a)
        s_idx = self.to_idx(self.s[0], self.s[1], self.s[2])
        csprob_n = self.proba_cumsum[s_idx][a]
        transition_idx = self.categorical_sample(csprob_n)
        p, s, r, d = self.P[s_idx][a][transition_idx]
        self.s = list(self.to_coordinate(s))
        return self.s, r[0] + r[1], d[0] and d[1], {"prob": p, "individual_reward1": r[0], "individual_reward2": r[1]}

    def run_episode(self, policy):
        policy = np.asarray(policy, dtype=np.int16).flatten()
        state = self.reset()
        total_reward1 = 0
        total_reward2 = 0
        bookings = np.zeros(self.nA)
        bookings_flight1 = np.zeros(len(self.prices_flight1))
        bookings_flight2 = np.zeros(len(self.prices_flight2))
        prices_proposed_flight1 = np.zeros(len(self.prices_flight1))
        prices_proposed_flight2 = np.zeros(len(self.prices_flight2))
        while True:
            # t, x = env.to_coordinate(state)
            state_idx = self.to_idx(state[0], state[1], state[2])
            action_idx = policy[state_idx]
            state, reward, done, info = self.step(action_idx)
            rewards = (info["individual_reward1"], info["individual_reward2"])
            action1, action2 = self.A[action_idx][0], self.A[action_idx][1]
            prices_proposed_flight1[self.prices_flight1.index(int(action1))] += 1
            prices_proposed_flight2[self.prices_flight2.index(int(action2))] += 1
            if rewards[0] != 0:
                bookings_flight1[self.prices_flight1.index(int(action1))] += 1
                # self.A[action_idx][0]
                bookings[action_idx] += 1
                total_reward1 += rewards[0]
            if rewards[1] != 0:
                bookings_flight2[self.prices_flight2.index(int(action2))] += 1
                # self.A[action_idx][1]
                bookings[action_idx] += 1
                total_reward2 += rewards[1]
            if done:
                break
        return total_reward1, total_reward2, bookings, bookings_flight1, bookings_flight2, prices_proposed_flight1, prices_proposed_flight2

    def average_n_episodes(self, policy, n_eval):
        """ Runs n episodes and returns the average of the n total rewards"""
        scores = [self.run_episode(policy) for _ in range(n_eval)]
        scores = np.array(scores)
        revenue1 = np.mean(scores[:, 0])
        revenue2 = np.mean(scores[:, 1])
        bookings = np.mean(scores[:, 2], axis=0)
        bookings_flight1 = np.mean(scores[:, 3], axis=0)
        bookings_flight2 = np.mean(scores[:, 4], axis=0)
        prices_proposed_flight1 = np.mean(scores[:, 5], axis=0)
        prices_proposed_flight2 = np.mean(scores[:, 6], axis=0)
        return revenue1, revenue2, bookings, bookings_flight1, bookings_flight2, prices_proposed_flight1, prices_proposed_flight2

    def collect_list_of_mean_revenues_and_bookings(self, experience_name):
        list_of_rewards = []
        for np_name in glob.glob(str(experience_name) + '/*.np[yz]'):
            list_of_rewards.append(list(np.load(np_name, allow_pickle=True)))

        nb_collection_points = len(list_of_rewards[0])

        rewards1_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]
        rewards2_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]
        all_bookings_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]
        bookings1_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]
        bookings2_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]
        prices_proposed1_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]
        prices_proposed2_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]

        for k in range(len(list_of_rewards)):
            rewards = list_of_rewards[k]
            for i in range(nb_collection_points):
                rewards1_combined_at_each_collection_point[i].append(rewards[i][0])
                rewards2_combined_at_each_collection_point[i].append(rewards[i][1])
                all_bookings_combined_at_each_collection_point[i].append(rewards[i][2])
                bookings1_combined_at_each_collection_point[i].append(rewards[i][3])
                bookings2_combined_at_each_collection_point[i].append(rewards[i][4])
                prices_proposed1_combined_at_each_collection_point[i].append(rewards[i][5])
                prices_proposed2_combined_at_each_collection_point[i].append(rewards[i][6])

        mean_revenues1 = [np.mean(list) for list in rewards1_combined_at_each_collection_point]
        mean_revenues2 = [np.mean(list) for list in rewards2_combined_at_each_collection_point]
        mean_bookings = [np.mean(list, axis=0) for list in all_bookings_combined_at_each_collection_point]
        mean_bookings1 = [np.mean(list, axis=0) for list in bookings1_combined_at_each_collection_point]
        mean_bookings2 = [np.mean(list, axis=0) for list in bookings2_combined_at_each_collection_point]
        mean_prices_proposed1 = [np.mean(list, axis=0) for list in prices_proposed1_combined_at_each_collection_point]
        mean_prices_proposed2 = [np.mean(list, axis=0) for list in prices_proposed2_combined_at_each_collection_point]

        return list_of_rewards, mean_revenues1, mean_revenues2, mean_bookings, mean_bookings1, mean_bookings2, mean_prices_proposed1, mean_prices_proposed2
