import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent

from dynamic_programming_env_DCP import dynamic_programming_env_DCP

from visualization_and_metrics import average_n_episodes

if __name__ == '__main__':
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
    global_env = gym.make('gym_Multidiscrete:Multidiscrete-v0', micro_times=micro_times,
                          capacity1=capacity1,
                          capacity2=capacity2,
                          actions=actions_global, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2,
                          lamb=lamb,
                          nested_lamb=nested_lamb)
    simple_env = gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', micro_times=1, data_collection_points=10,
                          capacity=8, lamb=0.9, alpha=0.8, actions=[k for k in range(50, 201, 50)])
    V, P_global = dynamic_programming_env_DCP(simple_env)
    P_global = P_global.reshape(simple_env.T * simple_env.C)
    revenues_global, bookings_global = average_n_episodes(simple_env, P_global, 10000)
    ENV_NAME = 'CartPole-v0'

    # env = gym.make(ENV_NAME)
    env = simple_env

    model = Sequential()
    # model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Flatten(input_shape=(1,1)))
    model.add(Dense(16))
    # model.add(Dense(16, input_shape=(2,)))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    # dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
    #                enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    dqn = DQNAgent(nb_actions=env.action_space.n, memory=memory, model=model,
                   enable_double_dqn=True, enable_dueling_network=True, target_model_update=50)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    dqn.test(env, nb_episodes=5, visualize=True)
