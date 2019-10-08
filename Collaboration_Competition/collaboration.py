import numpy as np
import gym
import matplotlib.pyplot as plt

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.dqn import agent_builder

from Collaboration_Competition.DQN_single_agent import train, update_target, initialize
from Collaboration_Competition.q_learning_collaboration import q_learning_collaboration3D, q_learning_collaboration2D, \
    q_learning_global, q_learning_collaboration2D_shared_Q_table
from Collaboration_Competition.DQN_experience import parameters_dict_builder
from Collaboration_Competition.competition import return_single_policies_from_collab_policy, \
    plot_comparison_Q_learning_VS_stabilized_competition, plot_global_bookings_histograms

from visualization_and_metrics import average_n_episodes_collaboration_individual_3D_policies, \
    average_n_episodes_collaboration_individual_2D_policies, \
    average_n_episodes_collaboration_individual_2D_VS_3D_policies, average_n_episodes_collaboration_global_policy, \
    visualizing_epsilon_decay, average_n_episodes_collaboration_individual_policy_on_n_flights_env

from dynamic_programming_env import dynamic_programming_collaboration, dynamic_programming_env
from value_iteration import value_iteration_discrete_collaboration


def callback_individual3D_agents(_locals, _globals):
    global n_steps, revenues1, revenues2, states, agent1, agent2, individual_3D_env1, individual_3D_env2, callback_frequency, global_env, fixed_policy
    if (n_steps + 1) % callback_frequency == 0 or n_steps == 0:
        policy1, q_values1, _ = agent1.step_model.step(states, deterministic=True)
        policy1 = np.array([individual_3D_env1.A[k] for k in policy1])
        if fixed_policy is None:
            policy2, q_values2, _ = agent2.step_model.step(states, deterministic=True)
            policy2 = np.array([individual_3D_env2.A[k] for k in policy2])
        else:
            policy2 = fixed_policy
        revenues, bookings = average_n_episodes_collaboration_individual_3D_policies(global_env, policy1,
                                                                                     policy2, individual_3D_env1, 10000)
        revenues1.append(revenues[0])
        revenues2.append(revenues[1])
    n_steps += 1
    return True


def callback_individual2D_agents(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, revenues1, revenues2, states1, states2, agent1, agent2, env1, env2, callback_frequency, collab_env, fixed_policy
    if (n_steps + 1) % callback_frequency == 0 or n_steps == 0:
        policy1, q_values1, _ = agent1.step_model.step(states1, deterministic=True)
        policy1 = np.array([env1.A[k] for k in policy1])
        if fixed_policy is None:
            policy2, q_values2, _ = agent2.step_model.step(states2, deterministic=True)
            policy2 = np.array([env2.A[k] for k in policy2])
            revenues, bookings = average_n_episodes_collaboration_individual_2D_policies(collab_env, env1, policy1,
                                                                                         policy2, 10000)
        else:
            policy2 = fixed_policy
            revenues, bookings = average_n_episodes_collaboration_individual_2D_VS_3D_policies(collab_env, policy2,
                                                                                               env1, policy1,
                                                                                               10000)
        revenues1.append(revenues[0])
        revenues2.append(revenues[1])
    n_steps += 1
    return True


def learn_individual3D_agents(agent1, agent2, env1, env2, total_timesteps, collaboration_env, fixed_policy):
    initialize(agent1, total_timesteps)
    initialize(agent2, total_timesteps)

    obs = collaboration_env.reset()

    for step in range(total_timesteps):
        if (step % (total_timesteps // 50)) == 0:
            print(step)
            print(revenues1, revenues2)

        callback_individual3D_agents(locals(), globals())

        update_eps1 = agent1.exploration.value(agent1.num_timesteps)
        update_eps2 = agent2.exploration.value(agent2.num_timesteps)

        with agent1.sess.as_default():
            action1 = agent1.act(np.array(obs)[None], update_eps=update_eps1)[0]
        if fixed_policy is None:
            with agent2.sess.as_default():
                # action2 = agent2.act(np.array(obs2)[None], update_eps=update_eps2)[0]
                action2 = agent2.act(np.array(obs)[None], update_eps=update_eps2)[0]
        else:
            action2 = env2.A.index(fixed_policy[obs])

        action_tuple = (env1.A[action1], env2.A[action2])
        action = collaboration_env.A.index(action_tuple)

        new_obs, rew, done, _ = collaboration_env.step(action)

        rew1, rew2 = rew[0] + rew[1], rew[0] + rew[1]
        done1, done2 = done[0], done[1]

        agent1.replay_buffer.add(obs, action1, rew1, new_obs, float(done1))
        if fixed_policy is None:
            agent2.replay_buffer.add(obs, action2, rew2, new_obs, float(done2))

        obs = new_obs

        if done1 and done2:
            obs = collaboration_env.reset()

        train(agent1)
        update_target(agent1)

        if fixed_policy is None:
            train(agent2)
            update_target(agent2)

        agent1.num_timesteps += 1
        agent2.num_timesteps += 1
    return agent1, agent2


def learn_individual2D_agents(agent1, agent2, env1, env2, total_timesteps, collaboration_env, fixed_policy):
    initialize(agent1, total_timesteps)
    initialize(agent2, total_timesteps)

    obs = collaboration_env.reset()
    t, x1, x2 = collaboration_env.to_coordinate(obs)
    obs1, obs2 = env1.to_idx(t, x1), env2.to_idx(t, x2)

    for step in range(total_timesteps):
        if (step % (total_timesteps // 50)) == 0:
            print(step)
            print(revenues1, revenues2)

        callback_individual2D_agents(locals(), globals())

        update_eps1 = agent1.exploration.value(agent1.num_timesteps)
        update_eps2 = agent2.exploration.value(agent2.num_timesteps)

        with agent1.sess.as_default():
            # action1 = agent1.act(np.array(obs1)[None], update_eps=update_eps1)[0]
            action1 = agent1.act(np.array(obs1)[None], update_eps=update_eps1)[0]
        if fixed_policy is None:
            with agent2.sess.as_default():
                # action2 = agent2.act(np.array(obs2)[None], update_eps=update_eps2)[0]
                action2 = agent2.act(np.array(obs2)[None], update_eps=update_eps2)[0]
        else:
            action2 = env2.A.index(fixed_policy[obs])

        action_tuple = (env1.A[action1], env2.A[action2])
        action = collaboration_env.A.index(action_tuple)

        new_obs, rew, done, _ = collaboration_env.step(action)

        new_t, new_x1, new_x2 = collaboration_env.to_coordinate(new_obs)
        new_obs1, new_obs2 = env1.to_idx(new_t, new_x1), env2.to_idx(new_t, new_x2)
        rew1, rew2 = rew[0] + rew[1], rew[0] + rew[1]
        done1, done2 = done[0], done[1]

        agent1.replay_buffer.add(obs1, action1, rew1, new_obs1, float(done1))
        if fixed_policy is None:
            agent2.replay_buffer.add(obs2, action2, rew2, new_obs2, float(done2))

        obs1 = new_obs1
        obs2 = new_obs2

        if done1 and done2:
            obs = collaboration_env.reset()
            t, x1, x2 = collaboration_env.to_coordinate(obs)
            obs1, obs2 = env1.to_idx(t, x1), env2.to_idx(t, x2)

        train(agent1)
        update_target(agent1)

        if fixed_policy is None:
            train(agent2)
            update_target(agent2)

        agent1.num_timesteps += 1
        agent2.num_timesteps += 1
    return agent1, agent2


def compare_revenues_for_n_flights(list_of_number_of_flights, demand_ratio):
    individual_capacity = 5
    action_min = 50
    action_max = 230
    action_offset = 50
    actions_individual = tuple(k for k in range(action_min, action_max + 1, action_offset))
    beta = 0.02
    k = 1.5
    micro_times = 40
    nested_lamb = 0.3

    optimal_revenues = []
    individual_revenues = []

    for number_of_flights in list_of_number_of_flights:
        arrival_rate = demand_ratio * (individual_capacity * number_of_flights) / micro_times
        n_flights_env = gym.make('gym_CollaborationGlobalNFlights:CollaborationGlobalNFlights-v0',
                                 micro_times=micro_times,
                                 individual_capacity=individual_capacity,
                                 individual_actions=actions_individual,
                                 beta=beta, k=k, arrival_rate=arrival_rate,
                                 nested_lamb=nested_lamb, number_of_flights=number_of_flights)
        arrival_rate_individual_flight = arrival_rate / number_of_flights
        individual_flight_env = gym.make('gym_CompetitionIndividual2D:CompetitionIndividual2D-v0',
                                         capacity=individual_capacity,
                                         micro_times=micro_times, actions=actions_individual,
                                         lamb=arrival_rate_individual_flight,
                                         beta=beta,
                                         k=k,
                                         nested_lamb=nested_lamb,
                                         competition_aware=False)

        values_n_flights = value_iteration_discrete_collaboration(n_flights_env)
        revenue_n_flights = values_n_flights[0]
        optimal_revenues.append(revenue_n_flights)

        values_individual_flight, policy_individual_flight = dynamic_programming_env(individual_flight_env)
        policy_individual_flight[-1][:] = actions_individual[0]
        policy_individual_flight[:, -1] = actions_individual[0]
        policy_individual_flight = policy_individual_flight.reshape(micro_times * individual_capacity)
        revenues_individual_flights, bookings_individual_flights = average_n_episodes_collaboration_individual_policy_on_n_flights_env(
            n_flights_env, individual_flight_env,
            policy_individual_flight, 10000)
        revenue_individual_flights = np.sum(revenues_individual_flights)
        individual_revenues.append(revenue_individual_flights)

    return optimal_revenues, individual_revenues


if __name__ == '__main__':
    # global n_steps, revenues1, revenues2, states, agent1, agent2, individual_3D_env1, individual_3D_env2, global_env, fixed_policy, callback_frequency

    # Parameters

    micro_times = 20
    capacity1 = 5
    capacity2 = 5
    global_capacity = capacity1 + capacity2 - 1

    action_min = 50
    action_max = 230
    action_offset = 30
    fixed_action = 90
    actions_global = tuple((k, m) for k in range(action_min, action_max + 1, action_offset) for m in
                           range(action_min, action_max + 1, action_offset))
    actions_individual = tuple(k for k in range(action_min, action_max + 1, action_offset))

    demand_ratio = 0.65
    lamb = demand_ratio * (capacity1 + capacity2) / micro_times

    beta = 0.02
    k_airline1 = 1.5
    k_airline2 = 1.5
    nested_lamb = 0.3

    global_env = gym.make('gym_CollaborationGlobal3D:CollaborationGlobal3D-v0', micro_times=micro_times,
                          capacity1=capacity1,
                          capacity2=capacity2,
                          actions=actions_global, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2,
                          lamb=lamb,
                          nested_lamb=nested_lamb)

    # Multi-agent RL

    individual_3D_env1 = gym.make('gym_CollaborationIndividual3D:CollaborationIndividual3D-v0',
                                  micro_times=micro_times,
                                  capacity1=capacity1, capacity2=capacity2, actions=actions_individual)
    individual_3D_env2 = gym.make('gym_CollaborationIndividual3D:CollaborationIndividual3D-v0',
                                  micro_times=micro_times,
                                  capacity1=capacity1, capacity2=capacity2, actions=actions_individual)

    parameters_dict1, parameters_dict2 = parameters_dict_builder(), parameters_dict_builder()
    parameters_dict1["env_builder"], parameters_dict2[
        "env_builder"] = individual_3D_env1, individual_3D_env2

    env_vec1, env_vec2 = DummyVecEnv([lambda: individual_3D_env1]), DummyVecEnv(
        [lambda: individual_3D_env2])

    agent1, agent2 = agent_builder(env_vec1, parameters_dict1), agent_builder(env_vec2, parameters_dict2)

    n_steps = 0
    revenues1, revenues2 = [], []
    states = [k for k in range(global_env.nS)]
    callback_frequency = 1000

    total_timesteps = 40000

    # Dynamic Programming solution on global environment

    V_global, P_global = dynamic_programming_collaboration(global_env)
    P_global = P_global.reshape(micro_times * capacity1 * capacity2)
    revenues_global, bookings_global = average_n_episodes_collaboration_global_policy(global_env, P_global,
                                                                                      individual_3D_env1,
                                                                                      10000)
    # Fixed policy

    P1, P2 = return_single_policies_from_collab_policy(P_global, global_env)
    optimal_individual_policy = P1

    constant_policy = np.zeros((global_env.nS))
    for k in range(len(constant_policy)):
        constant_policy[k] = 90

    fixed_policy = constant_policy

    # Deep Q-Learning Learning

    # learn_individual3D_agents(agent1, agent2, individual_3D_env1, individual_3D_env2, total_timesteps, global_env,
    #                           fixed_policy)

    # Dynamic Programming solution for fixed policy

    actions_global_fixed = [((k, fixed_action),) * len(actions_individual) for k in
                            range(action_min, action_max + 1, action_offset)]
    actions_global_fixed = np.array(actions_global_fixed).reshape(len(actions_individual) * len(actions_individual), 2)
    global_env_fixed_policy = gym.make('gym_CollaborationGlobal3D:CollaborationGlobal3D-v0', micro_times=micro_times,
                                       capacity1=capacity1,
                                       capacity2=capacity2,
                                       actions=actions_global_fixed, beta=beta, k_airline1=k_airline1,
                                       k_airline2=k_airline2,
                                       lamb=lamb,
                                       nested_lamb=nested_lamb)
    V_global_fixed_policy, P_global_fixed_policy = dynamic_programming_collaboration(global_env_fixed_policy)
    P_global_fixed_policy = P_global_fixed_policy.reshape(micro_times * capacity1 * capacity2)
    revenues_global_fixed_policy, bookings_global_fixed_policy = average_n_episodes_collaboration_global_policy(
        global_env_fixed_policy, P_global_fixed_policy,
        individual_3D_env1,
        10000)

    # Q-Learning

    gamma = 0.99
    # alpha, alpha_min, alpha_decay = 0.4, 0.02, 0.9999985
    alpha, alpha_min, alpha_decay = 0.8, 0, 0.9999975
    # alpha, alpha_min, alpha_decay = 0.2, 0, 0.9999975
    beta, beta_min, beta_decay = alpha / 2, 0, 0.9999975
    epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.9999975
    total_timesteps_q_learning = 2000001
    fully_collaborative = True

    visualizing_epsilon_decay(total_timesteps_q_learning, epsilon, epsilon_min,
                              epsilon_decay)  # Visualizing epsilon decay
    visualizing_epsilon_decay(total_timesteps_q_learning, alpha, alpha_min,
                              alpha_decay)  # Visualizing alpha decay
    visualizing_epsilon_decay(total_timesteps_q_learning, beta, beta_min,
                              beta_decay)  # Visualizing beta decay

    individual_2D_env1 = gym.make('gym_CollaborationIndividual2D:CollaborationIndividual2D-v0', micro_times=micro_times,
                                  capacity=capacity1, actions=actions_individual)
    individual_2D_env2 = gym.make('gym_CollaborationIndividual2D:CollaborationIndividual2D-v0', micro_times=micro_times,
                                  capacity=capacity2, actions=actions_individual)

    Q_tables_case1_hysteretic, revenues_case1_hysteretic, bookings_case1_hysteretic, episodes = q_learning_collaboration3D(
        global_env,
        individual_3D_env1,
        individual_3D_env2, alpha,
        alpha_min, alpha_decay, beta,
        beta_min, beta_decay, gamma,
        total_timesteps_q_learning,
        epsilon,
        epsilon_min, epsilon_decay,
        fully_collaborative)
    revenues_case1_hysteretic = np.array(revenues_case1_hysteretic)

    Q_tables_case2_hysteretic, revenues_case2_hysteretic, bookings_case2_hysteretic, episodes = q_learning_collaboration2D(
        global_env,
        individual_2D_env1,
        individual_2D_env2, alpha,
        alpha_min, alpha_decay, beta,
        beta_min, beta_decay, gamma,
        total_timesteps_q_learning,
        epsilon,
        epsilon_min, epsilon_decay,
        fully_collaborative)
    revenues_case2_hysteretic = np.array(revenues_case2_hysteretic)

    # Q_tables_case2_hysteretic_shared_Q_table, revenues_case2_hysteretic_shared_Q_table, bookings_case2_hysteretic_shared_Q_table, _ = q_learning_collaboration2D_shared_Q_table(
    #     global_env,
    #     individual_2D_env1,
    #     individual_2D_env2, alpha,
    #     alpha_min, alpha_decay, beta,
    #     beta_min, beta_decay, gamma,
    #     total_timesteps_q_learning,
    #     epsilon,
    #     epsilon_min, epsilon_decay,
    #     fully_collaborative)
    # revenues_case2_hysteretic_shared_Q_table = np.array(revenues_case2_hysteretic_shared_Q_table)

    Q_tables_case3_hysteretic, revenues_case3_hysteretic, bookings_case3_hysteretic, _ = q_learning_collaboration2D(
        global_env,
        individual_2D_env1,
        individual_2D_env2, alpha,
        alpha_min, alpha_decay, beta,
        beta_min, beta_decay, gamma,
        total_timesteps_q_learning,
        epsilon,
        epsilon_min, epsilon_decay,
        fully_collaborative=False)
    revenues_case3_hysteretic = np.array(revenues_case3_hysteretic)

    # Q_tables_case3_hysteretic_shared_Q_table, revenues_case3_hysteretic_shared_Q_table, bookings_case3_hysteretic_shared_Q_table, _ = q_learning_collaboration2D_shared_Q_table(
    #     global_env,
    #     individual_2D_env1,
    #     individual_2D_env2, alpha,
    #     alpha_min, alpha_decay, beta,
    #     beta_min, beta_decay, gamma,
    #     total_timesteps_q_learning,
    #     epsilon,
    #     epsilon_min, epsilon_decay,
    #     fully_collaborative=False)
    # revenues_case3_hysteretic_shared_Q_table = np.array(revenues_case3_hysteretic_shared_Q_table)

    # Q_table_global, revenues_QL_global, bookings_QL_global, _ = q_learning_global(global_env, individual_2D_env1,
    #                                                                               alpha, alpha_min, alpha_decay,
    #                                                                               gamma,
    #                                                                               total_timesteps_q_learning,
    #                                                                               epsilon,
    #                                                                               epsilon_min, epsilon_decay)
    # revenues_QL_global = np.array(revenues_QL_global)

    revenue_global = 677.548 #263
    # revenue_competition = 259 #670

    plt.figure()
    plt.plot(episodes, [revenue_global] * len(episodes), 'g--', label="Optimal solution")
    # plt.plot(episodes, [revenue_competition] * len(episodes), 'r--',
    #          label="Competition")
    # plt.plot(episodes, revenues_case1_hysteretic[0, :] + revenues_case1_hysteretic[1, :], color="m",
    #          label="Case 1")
    plt.plot(episodes, revenues_case2_hysteretic[0, :] + revenues_case2_hysteretic[1, :], color="y",
             label="Case 2")
    # plt.plot(episodes, revenues_case2_hysteretic_shared_Q_table[0, :] + revenues_case2_hysteretic_shared_Q_table[1, :], color="y", linestyle='dashed',
    #          label="Case 2 - Hysteretic QL \n shared Q-table")
    # plt.plot(episodes, revenues_case3_hysteretic[0, :] + revenues_case3_hysteretic[1, :], color="c",
    #          label="Case 3")
    # plt.plot(episodes, revenues_case3_hysteretic_shared_Q_table[0, :] + revenues_case3_hysteretic_shared_Q_table[1, :], color="c",linestyle='dashed',
    #          label="Case 3 - Hysteretic QL \n shared Q-table \n (Flight 1 + Flight 2)")
    plt.plot(episodes, revenues_case2_hysteretic[0, :], color="orange",
             label="Flight 1")
    plt.plot(episodes, revenues_case2_hysteretic[1, :], color="blue",
             label="Flight 2")
    # plt.plot(episodes, revenues_QL_global[0, :] + revenues_QL_global[1, :], color="orange",
    #          label="Centralized Q-Learning")

    plt.legend(loc='best')
    plt.xlabel("Number of episodes")
    plt.ylabel("Average revenue on {} flights".format(10000))

    plt.show()

    # To get rid of
    individual_env = gym.make('gym_CompetitionIndividual2D:CompetitionIndividual2D-v0', capacity=capacity1,
                              micro_times=micro_times, actions=actions_individual, lamb=lamb, beta=beta,
                              k=k_airline1,
                              nested_lamb=nested_lamb,
                              competition_aware=False)

    plot_global_bookings_histograms(individual_env,
                                    [bookings_case3_hysteretic[0][-1],
                                     bookings_case3_hysteretic[1][-1]])
