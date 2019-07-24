from DQL import agent
import random
import numpy as np
import time
from keras.losses import mean_squared_error, logcosh


def DQNAgent_time_builder(env, parameters_dict):
    return DQNAgent_time(env, gamma=parameters_dict["gamma"], epsilon=parameters_dict["epsilon"],
                         epsilon_min=parameters_dict["epsilon_min"], epsilon_decay=parameters_dict["epsilon_decay"],
                         replay_method=parameters_dict["replay_method"],
                         target_model_update=parameters_dict["target_model_update"],
                         batch_size=parameters_dict["batch_size"], state_scaler=parameters_dict["state_scaler"],
                         value_scaler=parameters_dict["value_scaler"], learning_rate=parameters_dict["learning_rate"],
                         dueling=parameters_dict["dueling"], hidden_layer_size=parameters_dict["hidden_layer_size"],
                         prioritized_experience_replay=parameters_dict["prioritized_experience_replay"],
                         memory_size=parameters_dict["memory_size"], mini_batch_size=parameters_dict["mini_batch_size"],
                         loss=parameters_dict["loss"], use_weights=parameters_dict["use_weights"],
                         use_optimal_policy=parameters_dict["use_optimal_policy"],
                         maximum_number_of_total_samples=parameters_dict["maximum_number_of_total_samples"])


class DQNAgent_time(agent.DQNAgent):

    def __init__(self, env, gamma=0.9,
                 epsilon=1., epsilon_min=0.2, epsilon_decay=0.9999,
                 replay_method="DDQL", target_model_update=10, batch_size=32,
                 state_scaler=None, value_scaler=None,
                 learning_rate=0.001, dueling=False, hidden_layer_size=50,
                 prioritized_experience_replay=False, memory_size=500,
                 mini_batch_size=64,
                 loss=mean_squared_error,
                 use_weights=False,
                 use_optimal_policy=False,
                 maximum_number_of_total_samples=1e6):

        self.mean_training_time = 0
        self.sum_training_time = 0

        super(DQNAgent_time, self).__init__(env, gamma, epsilon, epsilon_min, epsilon_decay,
                                            replay_method, target_model_update, batch_size,
                                            state_scaler, value_scaler,
                                            learning_rate, dueling, hidden_layer_size,
                                            prioritized_experience_replay, memory_size,
                                            mini_batch_size,
                                            loss,
                                            use_weights,
                                            use_optimal_policy,
                                            maximum_number_of_total_samples)

    def fill_memory_buffer(self):
        while (len(self.memory) != self.memory_size):
            state = self.env.reset()
            done = False
            while not done:
                action_idx = self.act(state)
                next_state, reward, done, _ = self.env.step(self.env.A[action_idx])

                self.remember(state, action_idx, reward, next_state, done)

                state = next_state

    def train_time(self, number_of_runs):
        training_times = []
        for i in range(number_of_runs):
            print("Current number of total samples of the agent: {}".format(self.number_of_total_samples))

            if self.number_of_total_samples >= self.maximum_number_of_total_samples:
                print("Number of samples superior to max number of samples")
                break

            minibatch = random.sample(self.memory, self.mini_batch_size)

            self.number_of_total_samples += self.mini_batch_size

            state_batch, action_batch, reward_batch, next_state_batch, done_batch, sample_weights = zip(*minibatch)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, sample_weights = np.array(
                state_batch).reshape(self.mini_batch_size, self.input_size), np.array(action_batch), np.array(
                reward_batch), np.array(next_state_batch).reshape(self.mini_batch_size, self.input_size), np.array(
                done_batch), np.array(sample_weights)

            q_values_target = reward_batch + self.get_discounted_max_q_value(next_state_batch)

            q_values_state = self.model.predict(state_batch)

            for k in range(self.mini_batch_size):
                q_values_state[k][action_batch[k]] = reward_batch[k] if done_batch[k] else q_values_target[k]

            start_time = time.time()
            self.model.fit(np.array(state_batch), np.array(q_values_state), epochs=1, verbose=0,
                           sample_weight=np.array(sample_weights), batch_size=self.batch_size)
            end_time = time.time() - start_time
            training_times.append(end_time)

        self.mean_training_time = np.mean(training_times)
        self.sum_training_time = np.sum(training_times)
