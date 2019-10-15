import numpy as np
import gym
import random

from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from visualization_and_metrics import average_n_episodes
from report_single_agent_results import agent_parameters_dict_DQL, agent_parameters_dict_QL, run_once_QL
from keras_rl_experience import run_once as run_once_DQL
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras_rl_experience import build_model

def env_builder(parameters_dict):
    actions = tuple(k for k in range(parameters_dict["action_min"], parameters_dict["action_max"],
                                     parameters_dict["action_offset"]))
    return gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=parameters_dict["data_collection_points"],
                    capacity=parameters_dict["capacity"],
                    micro_times=parameters_dict["micro_times"], actions=actions, alpha=parameters_dict["alpha"],
                    lamb=parameters_dict["lambda"], compute_P_matrix=parameters_dict["compute_P_matrix"],
                    transition_noise_percentage=parameters_dict["transition_noise_percentage"],
                    parameter_noise_percentage=parameters_dict["parameter_noise_percentage"])


def env_parameters():
    parameters_dict = {}
    parameters_dict["data_collection_points"] = 101
    parameters_dict["micro_times"] = 1
    parameters_dict["capacity"] = 51
    parameters_dict["action_min"] = 50
    parameters_dict["action_max"] = 231
    parameters_dict["action_offset"] = 20
    parameters_dict["lambda"] = 0.6
    parameters_dict["alpha"] = 0.7
    parameters_dict["compute_P_matrix"] = True
    parameters_dict["transition_noise_percentage"] = 0
    parameters_dict["parameter_noise_percentage"] = 0
    return parameters_dict

def collect_transitions(policy, epsilon, number_of_flights):
    transitions = []
    policy = np.asarray(policy, dtype=np.int16).flatten()
    for flight in range(number_of_flights):
        state = env.reset()
        while True:
            # t, x = env.to_coordinate(state)
            state_idx = env.to_idx(*state)
            if np.random.rand() <= epsilon:
                action_idx = random.randrange(env.action_space.n)
            else:
                action_idx = policy[state_idx]
            next_state, reward, done, _ = env.step(action_idx)
            transitions.append(tuple([state, action_idx, next_state, reward]))
            state = next_state
            if done:
                break
    return transitions

if __name__ == '__main__':
    env_param = env_parameters()
    env = env_builder(env_param)
    initial_true_V, initial_true_P = dynamic_programming_env_DCP(env)
    initial_true_revenues, initial_true_bookings = average_n_episodes(env, initial_true_P, 10000, epsilon=0.1)

    param_dict_DQL = agent_parameters_dict_DQL()
    model = build_model(env.nA, env.observation_space.shape, param_dict_DQL["hidden_layer_size"],
                        param_dict_DQL["layers_nb"])
    memory = SequentialMemory(limit=param_dict_DQL["memory_buffer_size"], window_length=1)
    policy = EpsGreedyQPolicy(eps=param_dict_DQL["epsilon"])
    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory,
                   nb_steps_warmup=param_dict_DQL["nb_steps_warmup"],
                   enable_double_dqn=param_dict_DQL["enable_double_dqn"],
                   enable_dueling_network=param_dict_DQL["enable_dueling_network"],
                   target_model_update=param_dict_DQL["target_model_update"], policy=policy)