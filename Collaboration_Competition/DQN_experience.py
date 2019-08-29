import gym

from Collaboration_Competition.DQN_single_agent import run_DQN_single_agent
from Collaboration_Competition.DQN_multi_agent import run_DQN_multi_agent
from dynamic_programming_env import dynamic_programming_collaboration


def CollaborationGlobal3D_env_builder():
    micro_times = 10
    capacity1 = 6
    capacity2 = 6
    actions = tuple((k, m) for k in range(10, 231, 20) for m in range(10, 231, 20))
    beta = 0.02
    k_airline1 = 1.5
    k_airline2 = 1.5
    lamb = 0.4
    nested_lamb = 0.3

    return gym.make('gym_CollaborationGlobal3D:CollaborationGlobal3D-v0', micro_times=micro_times,
                    capacity1=capacity1,
                    capacity2=capacity2,
                    actions=actions, beta=beta, k_airline1=k_airline1, k_airline2=k_airline2, lamb=lamb,
                    nested_lamb=nested_lamb)


def CompetitionIndividual2D_env_builder():
    capacity = 31
    micro_times = 100
    actions = tuple(k for k in range(10, 231, 20))
    lamb = 0.4
    beta = 0.02
    k = 1.5
    nested_lamb = 0.3

    return gym.make('gym_CompetitionIndividual2D:CompetitionIndividual2D-v0', capacity=capacity,
                    micro_times=micro_times, actions=actions, lamb=lamb, beta=beta, k=k,
                    nested_lamb=nested_lamb)


def CollaborationIndividual2D_env_builder():
    capacity = 31
    micro_times = 100
    actions = tuple(k for k in range(10, 231, 20))

    return gym.make('gym_CollaborationIndividual2D:CollaborationIndividual2D-v0', micro_times=micro_times,
                    capacity=capacity, actions=actions)


def CollaborationIndividual3D_env_builder():
    capacity1 = 31
    capacity2 = 31
    micro_times = 100
    actions = tuple(k for k in range(10, 231, 20))

    return gym.make('gym_CollaborationIndividual3D:CollaborationIndividual3D-v0', micro_times=micro_times,
                    capacity1=capacity1, capacity2=capacity2, actions=actions)


def parameters_dict_builder():
    parameters_dict = {}
    parameters_dict["env_builder"] = CollaborationGlobal3D_env_builder
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
    parameters_dict["policy_kwargs"] = {}
    parameters_dict["weights"] = False

    # env = parameters_dict["env_builder"]()
    parameters_dict["original_weights"] = None

    return parameters_dict


if __name__ == '__main__':
    callback_frequency = 1000
    total_timesteps = 40000

    parameters_dict = parameters_dict_builder()
    parameters_dict1, parameters_dict2 = parameters_dict_builder(), parameters_dict_builder()

    collab_global_env = CollaborationGlobal3D_env_builder()

    # Run DQN with single agent
    run_DQN_single_agent(parameters_dict, callback_frequency, total_timesteps)

    # Run DQN with two agents
    run_DQN_multi_agent(parameters_dict1, parameters_dict2, callback_frequency, total_timesteps)

    # Dynamic Programming
    V, P = dynamic_programming_collaboration(collab_global_env)
    P = P.reshape(collab_global_env.T * collab_global_env.C1 * collab_global_env.C2)
