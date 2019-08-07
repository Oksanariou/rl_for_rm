import gym

from Collaboration.DQN_single_agent import run_DQN_single_agent
from Collaboration.DQN_multi_agent import run_DQN_multi_agent
from Collaboration.DQN_multi_agent_collaboration import run_DQN_multi_agent_collaboration
from dynamic_programming_env import dynamic_programming_collaboration
from visualization_and_metrics import average_n_episodes_collaboration


def collaboration_env_builder():
    micro_times = 30
    capacity1 = 5
    capacity2 = 5
    actions = tuple((k, m) for k in range(50, 231, 20) for m in range(50, 231, 20))
    beta = 0.015
    k_airline1 = 1
    k_airline2 = 1
    lamb = 0.3

    return gym.make('gym_Competition:Competition-v0', micro_times=micro_times,
                    capacity_airline_1=capacity1,
                    capacity_airline_2=capacity2,
                    actions=actions, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2, lamb=lamb)

def discrete_RM_env_builder():
    capacity = 10
    micro_times = 10
    actions = tuple(k for k in range(50, 181, 20))
    alpha = 0.8
    lamb = 0.7

    return gym.make('gym_RMDiscrete:RMDiscrete-v0',capacity=capacity,
                 micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

def env_builder():
    # Parameters of the environment
    data_collection_points = 100
    micro_times = 5
    capacity = 50
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    return gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points,
                    capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)


def small_env_builder():
    # Parameters of the environment
    data_collection_points = 4
    micro_times = 3
    capacity = 4
    actions = tuple(k for k in range(50, 231, 50))
    alpha = 0.8
    lamb = 0.7

    return gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points,
                    capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)


def parameters_dict_builder():
    parameters_dict = {}
    parameters_dict["env_builder"] = discrete_RM_env_builder
    parameters_dict["gamma"] = 0.99
    parameters_dict["learning_rate"] = 0.0001
    parameters_dict["buffer_size"] = 30000
    parameters_dict["exploration_fraction"] = 0.4
    parameters_dict["exploration_final_eps"] = 0.01
    parameters_dict["train_freq"] = 1
    parameters_dict["batch_size"] = 100
    parameters_dict["checkpoint_freq"] = 10000
    parameters_dict["checkpoint_path"] = None
    parameters_dict["learning_starts"] = 100
    parameters_dict["target_network_update_freq"] = 50
    parameters_dict["prioritized_replay"] = False
    parameters_dict["prioritized_replay_alpha"] = 0.6
    parameters_dict["prioritized_replay_beta0"] = 0.4
    parameters_dict["prioritized_replay_beta_iters"] = None
    parameters_dict["prioritized_replay_eps"] = 1e-6
    parameters_dict["param_noise"] = False
    parameters_dict["verbose"] = 0
    parameters_dict["tensorboard_log"] = None
    parameters_dict["policy_kwargs"] = {"dueling": False, "layers": [100, 100]}
    parameters_dict["weights"] = False

    env = parameters_dict["env_builder"]()
    parameters_dict["original_weights"] = env.compute_weights()

    return parameters_dict


if __name__ == '__main__':
    callback_frequency = 1000
    total_timesteps = 40000

    parameters_dict = parameters_dict_builder()
    parameters_dict1, parameters_dict2 = parameters_dict_builder(), parameters_dict_builder()

    collaboration_env = collaboration_env_builder()

    # Run DQN with single agent
    run_DQN_single_agent(parameters_dict, callback_frequency, total_timesteps)

    # Run DQN with two agents
    run_DQN_multi_agent(parameters_dict1, parameters_dict2, callback_frequency, total_timesteps)

    # Run DQN with two agents collaborating
    run_DQN_multi_agent_collaboration(parameters_dict1, parameters_dict2, total_timesteps, collaboration_env)

    # Dynamic Programming
    V, P = dynamic_programming_collaboration(collaboration_env)
    P = P.reshape(collaboration_env.T * collaboration_env.C1 * collaboration_env.C2)
    print("Average reward over 10000 episodes : " + str(average_n_episodes_collaboration(collaboration_env, P, 10000)))

