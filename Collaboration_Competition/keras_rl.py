import numpy as np
import gym

from dynamic_programming_env import dynamic_programming_collaboration
from q_learning import q_to_v

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent

from dynamic_programming_env_DCP import dynamic_programming_env_DCP

from visualization_and_metrics import average_n_episodes, q_to_policy_RM


def global_env_builder():
    # Parameters of the environment
    micro_times = 50
    capacity1 = 10
    capacity2 = 10
    global_capacity = capacity1 + capacity2 - 1

    actions = tuple(k for k in range(50, 231, 70))


    action_min = 50
    action_max = 230
    action_offset = 50

    actions_global = tuple((k, m) for k in range(action_min, action_max + 1, action_offset) for m in
                           range(action_min, action_max + 1, action_offset))

    alpha = 0.8

    demand_ratio = 0.65
    lamb = demand_ratio * (capacity1 + capacity2) / micro_times

    beta = 0.02
    k_airline1 = 1.5
    k_airline2 = 1.5
    nested_lamb = 0.3


    return gym.make('gym_CollaborationGlobal3DMultiDiscrete:CollaborationGlobal3DMultiDiscrete-v0', micro_times=micro_times,
                          capacity1=capacity1,
                          capacity2=capacity2,
                          actions=actions_global, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2,
                          lamb=lamb,
                          nested_lamb=nested_lamb)

def env_builder():
    # Parameters of the environment
    data_collection_points = 50
    micro_times = 1
    capacity = 20

    action_min = 50
    action_max = 230
    action_offset = 50

    actions = tuple(k for k in range(action_min, action_max, action_offset))
    alpha = 0.8
    lamb = 0.7

    return gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)



if __name__ == '__main__':

    env = env_builder()

    if env.observation_space.shape[0]==2:
        true_V, true_P = dynamic_programming_env_DCP(env)
        true_revenues, true_bookings = average_n_episodes(env, true_P, 100)
    else:
        true_V, true_P = dynamic_programming_collaboration(env)
        true_revenues, true_bookings = average_n_episodes(env, true_P, 100)

    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    hidden_layer_size = 100

    model = Sequential()
    model.add(Flatten(input_shape=((1,) + env.observation_space.shape)))
    model.add(Dense(hidden_layer_size))
    model.add(Activation('relu'))
    model.add(Dense(hidden_layer_size))
    model.add(Activation('relu'))
    model.add(Dense(hidden_layer_size))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=.2)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                   enable_double_dqn=True, enable_dueling_network=True,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    history = dqn.fit(env, nb_steps=20_000, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    # dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    test_history = dqn.test(env, nb_episodes=100, visualize=False)

    import matplotlib.pyplot as plt

    w = 100
    moving_average = np.convolve(history.history['episode_reward'], np.ones(w), 'valid') / w
    plt.plot(list(range(0, w * len(moving_average), w)), moving_average, 'red', lw=4)
    plt.show()
    print("V(0,0)={}".format(max(dqn.compute_q_values([env.states[0]]))))
    print("evaluated revenue={}".format(np.mean(test_history.history['episode_reward'])))

    Q_table = [dqn.compute_q_values([state]) for state in env.states]
    policy = [np.argmax(q) for q in Q_table]
    policy = np.asarray(policy).reshape(env.observation_space.nvec)
    revenues, bookings = average_n_episodes(env, policy, 100)
    V = q_to_v(env, Q_table).reshape(env.observation_space.nvec)


