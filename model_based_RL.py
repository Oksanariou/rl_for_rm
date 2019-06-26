import numpy as np
import gym
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from keras.utils import to_categorical
from value_iteration import value_iteration, value_iteration_discrete
from visualization_and_metrics import visualize_policy_RM, visualisation_value_RM, extract_policy_RM_discrete, average_n_episodes


def collect_transitions_DCP(env, nb_transitions):
    input, output = [], []
    for k in range(nb_transitions):
        state_idx = env.set_random_state()
        action_idx = random.randrange(env.action_space.n)
        next_state_idx, reward, done, _ = env.step(action_idx)
        input.append([state_idx, action_idx])
        output.append(next_state_idx)
    return input, output

def collect_transitions_flights(env, nb_flights):
    input, output = [], []
    for k in range(nb_flights):
        state_idx = env.reset()
        done = False
        while not done:
            action_idx = random.randrange(env.action_space.n)
            next_state_idx, reward, done, _ = env.step(action_idx)
            input.append([state_idx, action_idx])
            output.append(next_state_idx)
            state_idx = next_state_idx
    return input, output

def init_transitions(env, model):
    P = {s: {a: [] for a in range(env.nA)} for s in range(env.nS)}
    for t in range(env.T):
        for x in range(env.C):
            s = env.to_idx(t,x)
            for a in range(env.nA):
                P[s][a] = transitions(env, s, a, model)
    return P

def transitions(env, state, action, model):
    predictions = model.predict_proba(np.array([state, action]).reshape(1, -1))[0]

    list_transitions = []
    t, x = env.to_coordinate(state)
    done = False
    if t == env.T - 1 or x == env.C - 1:
        list_transitions.append((1, state, 0, True))
    else:
        for k in range(len(predictions)):
            reward = 0
            proba_next_state = predictions[k]
            next_state = k + env.C
            if next_state > state + env.C and next_state <= state + 2*env.C:
                reward = env.A[action]*abs(state + env.C - next_state)
            new_t, new_x = env.to_coordinate(next_state)
            if new_t == env.T - 1 or new_x == env.C - 1:
                done = True

            list_transitions.append((proba_next_state, next_state, reward, done))

    return list_transitions


if __name__ == '__main__':
    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 50))
    alpha = 0.8
    lamb = 0.7
    env = gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points,
                   capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    values_DP = value_iteration_discrete(env, 100_000, 1e-20)
    policy_DP = extract_policy_RM_discrete(env, values_DP, 0.99)
    visualisation_value_RM(values_DP, env.T, env.C)
    visualize_policy_RM(policy_DP, env.T, env.C)
    print("Average reward over 10000 episodes : " + str(average_n_episodes(env, policy_DP, 10000)))

    X_train, y_train = collect_transitions_DCP(env, 10_000)
    X_test, y_test = collect_transitions_flights(env, 100)

    clf = MLPClassifier(hidden_layer_sizes=(50,50,))
    # clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    transitions_dic = init_transitions(env, clf)
    values_RL = value_iteration_discrete(env, 100_000, 1e-20, transitions_dic)
    policy_RL = extract_policy_RM_discrete(env, values_RL, 0.99, transitions_dic)
    visualize_policy_RM(policy_RL, env.T, env.C)
    print("Average reward over 10000 episodes : " + str(average_n_episodes(env, policy_RL, 10000)))


