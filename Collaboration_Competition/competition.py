import gym
from visualization_and_metrics import average_n_episodes, average_n_episodes_collaboration_global_policy, \
    average_n_episodes_collaboration_individual_3D_policies, \
    average_n_episodes_collaboration_individual_2D_policies
from value_iteration import value_iteration_discrete_collaboration, value_iteration_discrete
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dynamic_programming_env import dynamic_programming_collaboration, dynamic_programming_env


def build_competitor_state_distribution(competitor_env, competitor_policy):
    state_distribution = np.zeros((competitor_env.T, competitor_env.C))

    state_distribution[0][0] = 1
    for time in range(micro_times - 1):
        probability_to_buy = [competitor_env.proba_buy_not_alone(
            competitor_policy.reshape(competitor_env.T, competitor_env.C)[time][k],
            time) for k in range(len(competitor_policy.reshape(competitor_env.T, competitor_env.C)[time][
                                     :]))] if competitor_env.competition_aware else competitor_env.proba_buy_alone(
            competitor_policy.reshape(competitor_env.T, competitor_env.C)[time][:])
        probability_to_buy = np.array(probability_to_buy)
        transition_matrix = np.zeros((competitor_env.C, competitor_env.C))
        np.fill_diagonal(transition_matrix, 1 - probability_to_buy)
        for k in range(competitor_env.C - 1):
            transition_matrix[k + 1, k] = probability_to_buy[k]
        transition_matrix[competitor_env.C - 1][competitor_env.C - 1] = 1
        state_distribution[time + 1][:] = np.matmul(transition_matrix, np.transpose(state_distribution[time][:]))

    return state_distribution


def return_single_policies_from_collab_policy(collaboration_policy, collaboration_env):
    policy1, policy2 = [], []
    for action_tuple_idx in collaboration_policy:
        policy1.append(collaboration_env.A[int(action_tuple_idx)][0])
        policy2.append(collaboration_env.A[int(action_tuple_idx)][1])
    policy1 = np.array(policy1)
    policy2 = np.array(policy2)

    return policy1, policy2


def visualize_MNL_VS_nested(price_flight1, prices_flight2, values_of_lambda, global_env):
    plt.figure()
    capacity1 = global_env.C1
    capacity2 = global_env.C2
    micro_times = global_env.T
    actions = global_env.A
    beta = global_env.beta
    k_airline1, k_airline2 = global_env.k_airline1, global_env.k_airline2
    lamb, nested_lamb = global_env.lamb, global_env.nested_lamb
    for nested_lamb in values_of_lambda:
        collaboration_env = gym.make('gym_CollaborationGlobal3D:CollaborationGlobal3D-v0', micro_times=micro_times,
                                     capacity1=capacity1,
                                     capacity2=capacity2,
                                     actions=actions, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2,
                                     lamb=lamb,
                                     nested_lamb=nested_lamb)
        probas_buy_price_2 = []
        for price_2 in prices_flight2:
            action_tuple = (price_flight1, price_2)
            action_idx = collaboration_env.A.index(action_tuple)
            proba_buy_price_2 = collaboration_env.P[0][action_idx][2][0]
            probas_buy_price_2.append(proba_buy_price_2)
        plt.plot(prices_flight2, probas_buy_price_2, label="Mu = {}".format(nested_lamb))
    plt.axvline(x=price_flight1, color='g', label="Price of Flight 1", linestyle='--')
    plt.xticks(prices_flight2.A)
    plt.xlabel("Prices of Flight 2")
    plt.ylabel("Probability to buy ticket from Flight 2")
    plt.title("Evolution of the probability to buy the ticket from Flight 2 \n when the price of Flight 1 is fixed")
    plt.legend()
    plt.show()


def compute_revenues_and_bookings_from_competition(nb_iterations, initial_revenues, initial_bookings,
                                                   individual_2D_env, P_flight1, global_env):
    revenues_follow, revenues_adapt = [initial_revenues[0]], [initial_revenues[1]]
    bookings_follow, bookings_adapt = [initial_bookings[0]], [initial_bookings[1]]
    total_revenues = [initial_revenues[0] + initial_revenues[1]]

    capacity = individual_2D_env.C
    micro_times = individual_2D_env.T
    actions = individual_2D_env.A
    beta = individual_2D_env.beta
    k = individual_2D_env.k
    lamb, nested_lamb = individual_2D_env.lamb, individual_2D_env.nested_lamb

    for i in range(nb_iterations):
        competitor_state_distribution = build_competitor_state_distribution(individual_2D_env, P_flight1)

        individual_2D_env = gym.make('gym_CompetitionIndividual2D:CompetitionIndividual2D-v0', capacity=capacity,
                                     micro_times=micro_times, actions=actions, lamb=lamb, beta=beta, k=k,
                                     competition_aware=True, competitor_policy=P_flight1,
                                     competitor_distribution=competitor_state_distribution, nested_lamb=nested_lamb)

        V_flight2, P_flight2 = dynamic_programming_env(individual_2D_env)
        P_flight2[-1][:] = actions[0]
        P_flight2[:, -1] = actions[0]
        P_flight2 = P_flight2.reshape(micro_times * capacity)
        P_flight2_int = np.zeros((len(P_flight2)), int)
        P_flight1_int = np.zeros((len(P_flight1)), int)
        for m in range(len(P_flight1)):
            P_flight1_int[m] = int(individual_2D_env.A.index(P_flight1[m]))
            P_flight2_int[m] = int(individual_2D_env.A.index(P_flight2[m]))
        revenues, bookings = average_n_episodes_collaboration_individual_2D_policies(global_env, individual_2D_env,
                                                                                     P_flight1_int, P_flight2_int,
                                                                                     10000)
        revenues_follow.append(revenues[0]), revenues_adapt.append(revenues[1]), total_revenues.append(
            revenues[0] + revenues[1])
        bookings_follow.append(bookings[0]), bookings_adapt.append(bookings[1])
        P_flight1 = P_flight2

    revenues_1 = revenues_follow[:1]
    revenues_2 = revenues_adapt[:1]
    bookings_1 = bookings_follow[:1]
    bookings_2 = bookings_adapt[:1]

    for k in range(len(revenues_follow) // 2):
        revenues_1.append(revenues_follow[2 * (k) + 1])
        revenues_1.append(revenues_adapt[2 * (k + 1)])

        revenues_2.append(revenues_adapt[2 * (k) + 1])
        revenues_2.append(revenues_follow[2 * (k + 1)])

        bookings_1.append(bookings_follow[2 * (k) + 1])
        bookings_1.append(bookings_adapt[2 * (k + 1)])

        bookings_2.append(bookings_adapt[2 * (k) + 1])
        bookings_2.append(bookings_follow[2 * (k + 1)])

    return [revenues_1, revenues_2], [bookings_1, bookings_2]


def plot_comparison_global_policy_VS_individual_policies_coming_from_competition(nb_iterations, revenues_competition,
                                                                                 revenues_global):
    plt.figure()
    absc = [k for k in range(nb_iterations + 1)]
    plt.plot(absc, revenues_competition[1], label="Flight 2")
    plt.plot(absc, revenues_competition[0], label="Flight 1")
    plt.plot(absc, [np.sum(revenues_global)] * len(absc), "--", label="Global policy")
    plt.plot(absc, np.array(revenues_competition[0]) + np.array(revenues_competition[1]), label="Flight 1 + Flight 2")
    plt.plot(absc, [revenues_global[0]] * len(absc), '--', label="Global policy - Flight_1")
    plt.plot(absc, [revenues_global[1]] * len(absc), '--', label="Global policy - Flight_2")
    plt.legend(loc='upper center', bbox_to_anchor=(0.7, 0.45))

    plt.xlabel("Number of iterations")
    plt.ylabel("Average revenue on {} flights".format(10000))

    plt.show()


def plot_competition_bookings_histograms(nb_iterations, individual_2D_env, bookings_competition):
    for idx in range(nb_iterations):
        plt.figure()
        width = 5
        bookings_1 = bookings_competition[0]
        bookings_2 = bookings_competition[1]
        capacity = individual_2D_env.C
        plt.bar(individual_2D_env.A, bookings_2[idx], width,
                label="Flight 2, Load factor of {:.2}".format(np.sum(bookings_2[idx] / capacity)))
        plt.bar(individual_2D_env.A, bookings_1[idx], width,
                label="Flight 1, Load factor of {:.2}".format(np.sum(bookings_1[idx] / capacity)),
                bottom=bookings_2[idx])
        plt.title("Iteration {}, Load factor of the two flights combined: {:.2}".format(idx,
                                                                                        (np.sum(
                                                                                            bookings_1[idx]) + np.sum(
                                                                                            bookings_2[idx])) / (
                                                                                                capacity + capacity)))
        plt.xlabel("Prices")
        plt.ylabel("Average number of bookings")
        plt.legend()
        plt.show()


def plot_global_bookings_histograms(individual_2D_env, bookings_collab, title=None):
    plt.figure()
    width = 5
    capacity1, capacity2 = individual_2D_env.C, individual_2D_env.C
    lamb = individual_2D_env.lamb
    micro_times = individual_2D_env.T
    plt.bar(individual_2D_env.A, bookings_collab[1], width, color="blue", label="Flight 2")
    plt.bar(individual_2D_env.A, bookings_collab[0], width, color="orange", label="Flight 1", bottom=bookings_collab[1])
    plt.xlabel("Prices")
    plt.ylabel("Average number of bookings")
    plt.title("Overall load factor: {:.2}".format((np.sum(bookings_collab[0]) + np.sum(bookings_collab[1])) / (capacity1 + capacity2)))
    # if title is not None:
    #     plt.title(title + " - Global environment, demand ratio: {:.2}, load factor: {:.2}".format(
    #         (lamb * micro_times) / (capacity1 + capacity2),
    #         (np.sum(bookings_collab[0]) + np.sum(bookings_collab[1])) / (capacity1 + capacity2)))
    # else:
    #     plt.title("Global environment, demand ratio: {:.2}, load factor: {:.2}".format(
    #         (lamb * micro_times) / (capacity1 + capacity2),
    #         (np.sum(bookings_collab[0]) + np.sum(bookings_collab[1])) / (capacity1 + capacity2)))
    plt.legend()
    plt.xticks(individual_2D_env.A)
    plt.show()


def plot_comparison_Q_learning_VS_stabilized_competition(episodes, revenues_global, revenues_competition,
                                                         revenues_Q_learning):
    plt.figure()
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["g", "r", "m", "y", "c", "black"])
    plt.plot(episodes, [np.sum(revenues_global)] * len(episodes), '--', label="P_Global")
    plt.plot(episodes, [revenues_competition[0][-1] + revenues_competition[1][-1]] * len(episodes), '--',
             label="Competition")
    for k in range(len(revenues_Q_learning[:2])):
        plt.plot(episodes, np.array(revenues_Q_learning[k][0]) + np.array(revenues_Q_learning[k][1]),
                 label="Case {}".format(k + 1))

    plt.legend(loc='best')
    plt.xlabel("Number of episodes")
    plt.ylabel("Average revenue on {} flights".format(10000))

    plt.show()


if __name__ == '__main__':
    micro_times = 100
    capacity1 = 11
    capacity2 = 11
    action_min = 10
    action_max = 230
    action_offset = 20
    fixed_action = 90
    actions_global = tuple((k, m) for k in range(action_min, action_max + 1, action_offset) for m in
                           range(action_min, action_max + 1, action_offset))
    actions_individual = tuple(k for k in range(action_min, action_max + 1, action_offset))

    arrival_rate = 0.5
    # demand_ratio = 1.8
    # number_of_flights = 2
    # arrival_rate = demand_ratio * (capacity1 * number_of_flights) / micro_times

    beta = 0.02
    k_airline1 = 1.5
    k_airline2 = 1.5
    nested_lamb = 0.3

    global_env = gym.make('gym_CollaborationGlobal3D:CollaborationGlobal3D-v0', micro_times=micro_times,
                          capacity1=capacity1,
                          capacity2=capacity2,
                          actions=actions_global, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2,
                          lamb=arrival_rate,
                          nested_lamb=nested_lamb)

    # Customer choice models : Visualizing MNL VS nested
    # price_1 = 90
    # prices_2 = [k for k in range(10, 231, 20)]
    # values_of_lambda = [1, 0.3]
    # visualize_MNL_VS_nested(price_1, prices_2, values_of_lambda, global_env)

    # Dynamic Programming on the global environment
    V_global, P_global = dynamic_programming_collaboration(global_env)
    P_global = P_global.reshape(micro_times * capacity1 * capacity2)

    # Two optimal policies extracted from the optimal global policy computed by DP on the global environment
    P1, P2 = return_single_policies_from_collab_policy(P_global, global_env)

    # Individual environment representing Flight 1 which is at first NOT aware of competition
    arrival_rate_individual_env = arrival_rate / 2
    individual_2D_env1 = gym.make('gym_CompetitionIndividual2D:CompetitionIndividual2D-v0', capacity=capacity1,
                                  micro_times=micro_times, actions=actions_individual, lamb=arrival_rate_individual_env,
                                  beta=beta,
                                  k=k_airline1,
                                  nested_lamb=nested_lamb,
                                  competition_aware=False)
    V_flight1, P_flight1 = dynamic_programming_env(individual_2D_env1)
    P_flight1[-1][:] = actions_individual[0]
    P_flight1[:, -1] = actions_individual[0]
    P_flight1 = P_flight1.reshape(micro_times * capacity1)

    # Revenue and bookings made by the global policy computed on the global environment
    revenues_global, bookings_global = average_n_episodes_collaboration_global_policy(global_env, P_global,
                                                                                      individual_2D_env1, 10000)

    # Visualizing the bookings made by the global policy
    plot_global_bookings_histograms(individual_2D_env1, bookings_global)

    # Revenues and bookings made by the two individual policies coming from the two individual flights not aware of competition
    P_flight2 = np.zeros((len(P_flight1)), int)
    for k in range(len(P_flight1)):
        P_flight2[k] = int(individual_2D_env1.A.index(P_flight1[k]))

    initial_revenues, initial_bookings = average_n_episodes_collaboration_individual_2D_policies(global_env,
                                                                                                 individual_2D_env1,
                                                                                                 P_flight2, P_flight2,
                                                                                                 10000)

    # Revenues and bookings made by the two individual policies coming from the two flights AWARE of competition and adapting to the other flight's policy
    nb_iterations = 10
    revenues, bookings = compute_revenues_and_bookings_from_competition(nb_iterations, initial_revenues,
                                                                        initial_bookings,
                                                                        individual_2D_env1, P_flight1, global_env)

    # Visualizing the revenues coming from the global policy and the revenues coming from competition
    plot_comparison_global_policy_VS_individual_policies_coming_from_competition(nb_iterations, revenues,
                                                                                 revenues_global)

    # Visualizing the bookings made at each iteration by the two flights in competition
    plot_competition_bookings_histograms(nb_iterations, individual_2D_env1, bookings)

    # revenues_Q_learning = [revenues_Q_learning3D, revenues_Q_learning2D, revenues_Q_learning2D_individual_reward,
    #                        revenues_Q_learning3D_individual_reward]
    #
    # plot_comparison_Q_learning_VS_stabilized_competition(episodes, revenues_global, revenues,
    #                                                      revenues_Q_learning)
