import numpy as np
from gym import spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete
import itertools
import copy

number_of_flights = 2
default_micro_times = 500
default_individual_capacity = 50
default_individual_actions = tuple(k for k in range(50, 231, 20))
default_beta = 0.001
default_k = 1.5
default_arrival_rate = 0.8
default_nested_lambda = 1.


class CollaborationGlobalNFlightsEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, micro_times=default_micro_times, individual_capacity=default_individual_capacity,
                 individual_actions=default_individual_actions,
                 beta=default_beta, k=default_k, arrival_rate=default_arrival_rate,
                 nested_lamb=default_nested_lambda, number_of_flights=number_of_flights):

        self.number_of_flights = number_of_flights

        self.T = micro_times
        self.individual_capacity = individual_capacity
        self.C = individual_capacity ** self.number_of_flights
        self.nS = self.T * self.C  # number of states

        self.individual_actions = individual_actions
        self.A = list(itertools.product(self.individual_actions, repeat=self.number_of_flights))
        self.nA = len(self.A)  # number of actions

        self.beta = beta
        self.k = k

        self.arrival_rate = arrival_rate

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.seed()

        self.s = 0

        self.nested_lamb = nested_lamb
        self.nest1 = {"lambda": 1, "representative_utilities": [0]}

        self.probas_from_all_prices = self.compute_all_possible_probas()

        self.P = self.init_transitions()

        self.isd = np.zeros(self.nS, float)
        self.isd[0] = 1.

        super(CollaborationGlobalNFlightsEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def compute_all_possible_probas(self):
        all_possible_probas = []
        for action_idx in range(self.nA):
            action = self.A[action_idx]
            utilities = np.array([self.k - self.beta * action[i] for i in range(self.number_of_flights)])
            nest_2 = {}
            nest_2["lambda"] = self.nested_lamb
            nest_2["representative_utilities"] = utilities
            nests = [self.nest1, nest_2]
            probas_logit = self.compute_probas_nested_logit(nests)
            all_possible_probas.append(self.compute_proba_buy(probas_logit))
        return all_possible_probas

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
            probas.append(self.arrival_rate * probas_logit[k])
        proba_nogo = 1 - self.arrival_rate * (sum(probas_logit[1:]))
        probas[0] = proba_nogo

        return probas

    def init_transitions(self):
        # Transitions: P[s][a] = [(probability, nextstate, reward, done), ...]
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for state_idx in range(self.nS):
            for action_idx in range(self.nA):
                P[state_idx][action_idx] = self.transitions(state_idx, action_idx)
        return P

    def transitions(self, state_idx, action_idx):
        list_transitions = []
        time, bookings = self.to_coordinate(state_idx)

        rewards = np.array([0 for k in range(self.number_of_flights)])

        dones = np.array([(time >= self.T - 2) for k in range(self.number_of_flights)])

        if time == self.T - 1 or (bookings == [self.individual_capacity - 1 for k in range(self.number_of_flights)]):
            dones = np.array([True for k in range(self.number_of_flights)])
            list_transitions.append((1, state_idx, rewards, dones))

        elif np.all(np.array(bookings) < np.array([self.individual_capacity - 1 for k in range(self.number_of_flights)])):
            new_time = time + 1
            booleans = np.array([0 for k in range(self.number_of_flights)])
            for flight_idx in range(self.number_of_flights):
                if bookings[flight_idx] == (self.individual_capacity - 1):
                    booleans[flight_idx] = -1000000
                    dones[flight_idx] = True

            action = self.A[action_idx]
            probas = self.probas_from_all_prices[action_idx]

            list_transitions.append((probas[0], self.to_idx(time + 1, bookings), rewards, dones))

            for flight_idx in range(self.number_of_flights):
                new_flight_capacity = bookings[flight_idx] + 1
                new_bookings = copy.deepcopy(bookings)
                new_bookings[flight_idx] = new_flight_capacity

                new_state = self.to_idx(new_time, new_bookings)

                probability = probas[flight_idx + 1]
                new_rewards = copy.deepcopy(rewards)
                new_rewards[flight_idx] = action[flight_idx]

                new_dones = copy.deepcopy(dones)
                new_dones[flight_idx] = (new_flight_capacity == self.individual_capacity - 1) or (
                        bookings[flight_idx] == self.individual_capacity - 1) or (time >= self.T - 2)

                list_transitions.append((probability, new_state, new_rewards, new_dones))

        else:
            new_time = time + 1
            booleans = np.array([0 for k in range(self.number_of_flights)])
            for flight_idx in range(self.number_of_flights):
                if bookings[flight_idx] == (self.individual_capacity - 1):
                    booleans[flight_idx] = -1000000
                    dones[flight_idx] = True

            action = self.A[action_idx]

            utilities = np.array([self.k - self.beta * action[i] for i in range(self.number_of_flights)])
            utilities = utilities + booleans

            nest_2 = {}
            nest_2["lambda"] = self.nested_lamb
            nest_2["representative_utilities"] = utilities
            nests = [self.nest1, nest_2]
            probas_logit = self.compute_probas_nested_logit(nests)
            probas = self.compute_proba_buy(probas_logit)

            list_transitions.append((probas[0], self.to_idx(time + 1, bookings), rewards, dones))

            for flight_idx in range(self.number_of_flights):
                new_flight_capacity = bookings[flight_idx] + 1
                new_bookings = copy.deepcopy(bookings)
                new_bookings[flight_idx] = new_flight_capacity

                new_state = self.to_idx(new_time, new_bookings)

                probability = probas[flight_idx + 1]
                new_rewards = copy.deepcopy(rewards)
                new_rewards[flight_idx] = action[flight_idx]

                new_dones = copy.deepcopy(dones)
                new_dones[flight_idx] = (new_flight_capacity == self.individual_capacity - 1) or (
                        bookings[flight_idx] == self.individual_capacity - 1) or (time >= self.T - 2)

                list_transitions.append((probability, new_state, new_rewards, new_dones))
        return list_transitions

    def to_coordinate(self, state_idx):
        t = int(int(state_idx) / (self.individual_capacity ** self.number_of_flights))
        x = [t]
        for flight_idx in range(self.number_of_flights):
            numerator = state_idx
            for previous_x_idx in range(len(x)):
                numerator -= x[previous_x_idx] * self.individual_capacity ** (self.number_of_flights - previous_x_idx)
            numerator = int(numerator)
            denominator = self.individual_capacity ** (
                    self.number_of_flights - 1 - flight_idx)
            x.append(int(numerator / denominator))
        return t, x[1:]

    def to_idx(self, t, x):
        idx = t * self.individual_capacity ** self.number_of_flights
        for booking_idx in range(self.number_of_flights):
            idx += x[booking_idx] * self.individual_capacity ** (
                    self.number_of_flights - 1 - booking_idx)
        return idx
