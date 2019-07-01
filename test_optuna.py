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
    # x = trial.suggest_uniform('x', -10, 10)
    x = trial.suggest_discrete_uniform('x', -10, 10, 2)

    for step in range(100):
        intermediate_value = (x - 2) ** 2
        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.structs.TrialPruned()

    return (x - 2) ** 2


study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)

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

    # learning_rate = trial.suggest_loguniform('learning_rate', 0.0001, 0.01)
    # memory_size = trial.suggest_int('memory_size', 1_000, 20_000)
    minibatch_size = trial.suggest_discrete_uniform('mini_batch_size', 10, 170, 40)
    # percent_minibatch_size = trial.suggest_uniform('mini_batch_size', 0.0, 0.3)
    target_model_update = trial.suggest_discrete_uniform('target_model_update', 5, 55, 10)

    parameters_dict = {}
    parameters_dict["env"] = env
    parameters_dict["replay_method"] = "DDQL"
    parameters_dict["batch_size"] = 32
    parameters_dict["memory_size"] = 8_000
    parameters_dict["mini_batch_size"] = minibatch_size
    parameters_dict["prioritized_experience_replay"] = False
    parameters_dict["target_model_update"] = int(target_model_update)
    parameters_dict["hidden_layer_size"] = 50
    parameters_dict["dueling"] = True
    parameters_dict["loss"] = mean_squared_error
    parameters_dict["learning_rate"] = 0.001
    parameters_dict["epsilon"] = 0.01
    parameters_dict["epsilon_min"] = 0.01
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
                     hidden_layer_size=parameters_dict["hidden_layer_size"], dueling=parameters_dict["dueling"],
                     loss=parameters_dict["loss"], learning_rate=parameters_dict["learning_rate"],
                     epsilon=parameters_dict["epsilon"], epsilon_min=parameters_dict["epsilon_min"],
                     epsilon_decay=parameters_dict["epsilon_decay"],
                     state_weights=parameters_dict["state_weights"])

    nb_episodes = 10_000
    nb_intermediate_values = 3
    nb_episodes_between_intermediate_values = nb_episodes // nb_intermediate_values

    after_train = lambda episode: episode % nb_episodes_between_intermediate_values == 0
    q_compute = QCompute(after_train, agent)
    revenue_compute = RevenueMonitor(after_train, agent, q_compute, 10_000)

    callbacks = [q_compute, revenue_compute]

    agent.init_network_with_true_Q_table()

    for step in range(nb_intermediate_values):
        agent.train(nb_episodes_between_intermediate_values, callbacks)

        intermediate_value = -revenue_compute.revenues[-1]
        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.structs.TrialPruned()

    # return -np.mean(revenue_compute.revenues[])
    return -revenue_compute.revenues[-1]


study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)

print(study.best_params)
