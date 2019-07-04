import gym
from DQL.agent import DQNAgent
from keras.models import load_model
from keras.losses import mean_squared_error
from visualization_and_metrics import q_to_policy_RM, average_n_episodes

if __name__ == '__main__':

    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    parameters_dict = {}
    parameters_dict["env"] = env
    parameters_dict["replay_method"] = "DDQL"
    parameters_dict["batch_size"] = 32
    parameters_dict["memory_size"] = 6_000
    parameters_dict["mini_batch_size"] = 1_000
    parameters_dict["prioritized_experience_replay"] = False
    parameters_dict["target_model_update"] = 90
    parameters_dict["hidden_layer_size"] = 50
    parameters_dict["dueling"] = True
    parameters_dict["loss"] = mean_squared_error
    parameters_dict["learning_rate"] = 1e-3
    parameters_dict["epsilon"] = 0.0
    parameters_dict["epsilon_min"] = 0.0
    parameters_dict["epsilon_decay"] = 0.9995
    parameters_dict["state_weights"] = True

    # minibatch_size = int(parameters_dict["memory_size"] * percent_minibatch_size)
    # parameters_dict["mini_batch_size"] = minibatch_size

    agent = DQNAgent(env=parameters_dict["env"],
                     # state_scaler=env.get_state_scaler(), value_scaler=env.get_value_scaler(),
                     replay_method=parameters_dict["replay_method"], batch_size=parameters_dict["batch_size"],
                     memory_size=parameters_dict["memory_size"], mini_batch_size=parameters_dict["mini_batch_size"],
                     prioritized_experience_replay=parameters_dict["prioritized_experience_replay"],
                     target_model_update=parameters_dict["target_model_update"],
                     hidden_layer_size=parameters_dict["hidden_layer_size"],
                     dueling=parameters_dict["dueling"],
                     loss=parameters_dict["loss"], learning_rate=parameters_dict["learning_rate"],
                     epsilon=parameters_dict["epsilon"], epsilon_min=parameters_dict["epsilon_min"],
                     epsilon_decay=parameters_dict["epsilon_decay"],
                     state_weights=parameters_dict["state_weights"]
                     )

    # Computing the model by initializing it with the true Q table + saving it
    agent.init_network_with_true_Q_table()
    model_name = "DQL/model_initialized_with_true_q_table.h5"
    agent.model.save(model_name)

    Q_table = agent.compute_q_table()
    policy = q_to_policy_RM(env, Q_table)
    print(average_n_episodes(env, policy, 10_000))

    # Loading the model and initializing the agent's network with it
    model = load_model(model_name)
    agent.set_model(model)
    agent.set_target()


