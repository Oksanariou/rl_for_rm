import gym

micro_times = 5
capacity1 = 3
capacity2 = 4
actions = tuple((k, m) for k in range(50, 231, 50) for m in range(50, 231, 50))
beta = 0.001
k_airline1 = 1.5
k_airline2 = 1.5

env = gym.make('gym_Competition:Competition-v0', micro_times=micro_times, capacity_airline_1=capacity1,
               capacity_airline_2=capacity2,
               actions=actions, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2)