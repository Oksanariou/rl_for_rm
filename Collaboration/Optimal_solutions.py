import gym
from dynamic_programming_env import dynamic_programming_env
from visualization_and_metrics import average_n_episodes, visualize_policy_RM

if __name__ == '__main__':
    micro_times = 40
    actions = tuple(k for k in range(50, 171, 20))
    alpha = 0.6

    # Big flight
    capacity = 11
    lamb = 0.8

    env_big_flight = gym.make('gym_RMDiscrete:RMDiscrete-v0', capacity=capacity,
                              micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    # Dynamic programming
    V, P_big_flight = dynamic_programming_env(env_big_flight)
    visualize_policy_RM(P_big_flight, env_big_flight.T, env_big_flight.C)
    P_big_flight = P_big_flight.reshape(env_big_flight.T * env_big_flight.C)
    print("Big flight, Average reward over 10000 episodes : " + str(
        average_n_episodes(env_big_flight, P_big_flight, 10000)))

    # Small flights
    capacity_flight1 = int(capacity / 2) + 1
    capacity_flight2 = int(capacity / 2) + 1

    # lamb_flight1 = (lamb * (capacity_flight1 - 1)) / (capacity - 1)
    # lamb_flight2 = (lamb * (capacity_flight2 - 1)) / (capacity - 1)

    lamb_flight1 = lamb
    lamb_flight2 = lamb


    alpha1 = alpha
    alpha2 = alpha

    micro_times1 = int(micro_times / 2) + 1
    micro_times2 = int(micro_times / 2) + 1

    env_flight1 = gym.make('gym_RMDiscrete:RMDiscrete-v0', capacity=capacity_flight1,
                           micro_times=micro_times1, actions=actions, alpha=alpha1, lamb=lamb_flight1)
    env_flight2 = gym.make('gym_RMDiscrete:RMDiscrete-v0', capacity=capacity_flight2,
                           micro_times=micro_times2, actions=actions, alpha=alpha2, lamb=lamb_flight2)

    # Dynamic programming
    V, P_flight1 = dynamic_programming_env(env_flight1)
    visualize_policy_RM(P_flight1, env_flight1.T, env_flight1.C)
    P_flight1 = P_flight1.reshape(env_flight1.T * env_flight1.C)

    V, P_flight2 = dynamic_programming_env(env_flight2)
    visualize_policy_RM(P_flight2, env_flight2.T, env_flight2.C)
    P_flight2 = P_flight2.reshape(env_flight2.T * env_flight2.C)

    print("flight 1, Average reward over 10000 episodes : " + str(
        average_n_episodes(env_flight1, P_flight1, 10000)))
    print("flight 2, Average reward over 10000 episodes : " + str(
        average_n_episodes(env_flight2, P_flight2, 10000)))
    print("Sum of flight 2 and flight 1, Average reward over 10000 episodes : " + str(
        average_n_episodes(env_flight1, P_flight1, 10000) + average_n_episodes(env_flight2, P_flight2, 10000)))
