from DQL import agent
import random
import numpy as np
import time

class DQNAgent_time(agent.DQNAgent):

    def __init__(self, env, mini_batch_size, batch_size, memory_size, maximum_number_of_total_samples):

        self.env = env
        self.memory_size = memory_size
        self.maximum_number_of_total_samples = maximum_number_of_total_samples
        self.mini_batch_size = mini_batch_size
        self.batch_size = batch_size

        self.training_time = 0

        super(DQNAgent_time, self).__init__(self.env)

    def fill_memory_buffer(self):
        while(len(self.memory) != self.memory_size):
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
            training_times.append(time.time() - start_time)

        self.training_time = np.mean(training_times)