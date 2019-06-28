import optuna
import gym
import numpy as np
from keras.losses import mean_squared_error, logcosh
from DQL.callbacks import TrueCompute, VDisplay, RevenueMonitor, RevenueDisplay, AgentMonitor, QCompute, QErrorDisplay, \
    QErrorMonitor, PolicyDisplay, MemoryMonitor, MemoryDisplay, BatchMonitor, BatchDisplay, TotalBatchDisplay, \
    SumtreeMonitor, SumtreeDisplay
from dynamic_programming_env_DCP import dynamic_programming_env_DCP

from DQL.agent import DQNAgent

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(study.best_params)

def objective(trial):
    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    learning_rate = trial.suggest_loguniform('learning_rate', 0.0001, 0.01)
    memory_size = trial.suggest_int('memory_size', 1_000, 20_000)
    minibatch_size = trial.suggest_int('mini_batch_size', 30, 200)
    target_model_update = trial.suggest_int('target_model_update', 50, 200)

    parameters_dict = {}
    parameters_dict["env"] = env
    parameters_dict["replay_method"] = "DDQL"
    parameters_dict["batch_size"] = 32
    parameters_dict["memory_size"] = memory_size
    parameters_dict["mini_batch_size"] = minibatch_size
    parameters_dict["prioritized_experience_replay"] = False
    parameters_dict["target_model_update"] = target_model_update
    parameters_dict["hidden_layer_size"] = 50
    parameters_dict["dueling"] = True
    parameters_dict["loss"] = mean_squared_error
    parameters_dict["learning_rate"] = learning_rate
    parameters_dict["epsilon"] = 0.01
    parameters_dict["epsilon_min"] = 0.01
    parameters_dict["epsilon_decay"] = 0.9995
    parameters_dict["state_weights"] = True

    agent = DQNAgent(env=parameters_dict["env"],
                     # state_scaler=env.get_state_scaler(), value_scaler=env.get_value_scaler(),
                     replay_method=parameters_dict["replay_method"], batch_size=parameters_dict["batch_size"],
                     memory_size=parameters_dict["memory_size"], mini_batch_size=parameters_dict["mini_batch_size"],
                     prioritized_experience_replay=parameters_dict["prioritized_experience_replay"],
                     target_model_update=parameters_dict["target_model_update"],
                     hidden_layer_size=parameters_dict["hidden_layer_size"], dueling=parameters_dict["dueling"],
                     loss=parameters_dict["loss"], learning_rate=parameters_dict["learning_rate"],
                     epsilon=parameters_dict["epsilon"], epsilon_min=parameters_dict["epsilon_min"],
                     epsilon_decay=parameters_dict["epsilon_decay"],
                     state_weights=parameters_dict["state_weights"])

    nb_episodes = 10_000

    agent.init_network_with_true_Q_table()

    while_training = lambda episode: episode % (nb_episodes / 20) == 0

    q_compute = QCompute(while_training, agent)
    revenue_compute = RevenueMonitor(while_training, agent, q_compute, 10_000)

    callbacks = [q_compute, revenue_compute]

    agent.train(nb_episodes, callbacks)

    return -np.mean(revenue_compute.revenues)

study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(study.best_params)


