import gym
from DQL.agent import DQNAgent
from keras.models import load_model

if __name__ == '__main__':

    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    agent = DQNAgent(env=env, learning_rate=1e-3)

    # Computing the model by initializing it with the true Q table + saving it
    agent.init_network_with_true_Q_table()
    model_name = "DQL/model_initialized_with_true_q_table.h5"
    agent.model.save(model_name)

    # Loading the model and initializing the agent's network with it
    model = load_model(model_name)
    agent.set_model(model)
    agent.set_target()


