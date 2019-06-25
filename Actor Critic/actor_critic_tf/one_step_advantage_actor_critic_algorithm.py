import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf

# This file contains two classes for TensorFlow implementations of
# artificial neural networks used for the running the REINFORCE
# algorithm as given in chapter 13.3 of Sutton 2017. The first is the
# policy_estimator network and the second is the value_estimator.
# Detailed explanation is given at
# https://www.datahubbs.com/policy-gradients-with-reinforce/

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from visualization_and_metrics import visualize_policy_RM


class actor(object):

    def __init__(self, sess, env):
        # Pass TensorFlow session object
        self.sess = sess
        # Get number of inputs and outputs from environment
        #self.n_inputs = env.observation_space.shape[0]
        self.n_inputs = 2
        self.n_outputs = env.action_space.n
        self.learning_rate = 0.0005

        # Define number of hidden nodes
        self.n_hidden_nodes = 20

        # Set graph scope name
        self.scope = "policy_estimator"

        # Create network
        with tf.variable_scope(self.scope):
            initializer = tf.contrib.layers.xavier_initializer()

            # Define placholder tensors for state, actions, and rewards
            self.state = tf.placeholder(tf.float32, [None, self.n_inputs],
                                        name='state')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            self.actions = tf.placeholder(tf.int32, [None], name='actions')

            layer_1 = fully_connected(self.state, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)
            output_layer = fully_connected(layer_1, self.n_outputs,
                                           activation_fn=None,
                                           weights_initializer=initializer)

            # Get probability of each action
            self.action_probs = tf.squeeze(
                tf.nn.softmax(output_layer - tf.reduce_max(output_layer)))

            # Get indices of actions
            indices = tf.range(0, tf.shape(output_layer)[0]) \
                      * tf.shape(output_layer)[1] + self.actions

            selected_action_prob = tf.gather(tf.reshape(self.action_probs, [-1]),
                                             indices)

            # Define loss function
            self.loss = -tf.reduce_mean(tf.log(selected_action_prob) * self.rewards)

            # Get gradients and variables
            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32,
                                                           name='grads' + str(j)))

            self.gradients = tf.gradients(self.loss, self.tvars)

            # Minimize training error
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state):
        probs = self.sess.run([self.action_probs],
                              feed_dict={
                                  self.state: state
                              })[0]
        return probs

    def update(self, states, actions, rewards):
        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={
                                    self.state: states,
                                    self.actions: actions,
                                    self.rewards: rewards})
        return loss

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, actions, rewards):
        grads = self.sess.run([self.gradients],
                              feed_dict={
                                  self.state: states,
                                  self.actions: actions,
                                  self.rewards: rewards
                              })[0]
        return grads


class critic(object):

    def __init__(self, sess, env):
        # Pass TensorFlow session object
        self.sess = sess
        # Get number of inputs and outputs from environment
        #self.n_inputs = env.observation_space.shape[0]
        self.n_inputs = 2
        self.n_outputs = 1
        self.learning_rate = 0.0005

        # Define number of hidden nodes
        self.n_hidden_nodes = 20

        # Set graph scope name
        self.scope = "value_estimator"

        # Create network
        with tf.variable_scope(self.scope):
            initializer = tf.contrib.layers.xavier_initializer()

            # Define placholder tensors for state, actions, and rewards
            self.state = tf.placeholder(tf.float32, [None, self.n_inputs],
                                        name='state')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')

            layer_1 = fully_connected(self.state, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)
            output_layer = fully_connected(layer_1, self.n_outputs,
                                           activation_fn=None,
                                           weights_initializer=initializer)

            self.state_value_estimation = tf.squeeze(output_layer)

            # Define loss function as squared difference between estimate and
            # actual
            self.loss = tf.reduce_mean(tf.squared_difference(
                self.state_value_estimation, self.rewards))

            # Get gradients and variables
            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32,
                                                           name='grads' + str(j)))

            self.gradients = tf.gradients(self.loss, self.tvars)

            # Minimize training error
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state):
        value_est = self.sess.run([self.state_value_estimation],
                                  feed_dict={
                                      self.state: state
                                  })[0]
        return value_est

    def update(self, states, rewards):
        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={
                                    self.state: states,
                                    self.rewards: rewards})
        return loss

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, rewards):
        grads = self.sess.run([self.gradients],
                              feed_dict={
                                  self.state: states,
                                  self.rewards: rewards
                              })[0]
        return grads

class A2C():
    '''
    Initialization Inputs
    =====================================================
    env: class, OpenAI environment such as CartPole
    actor: class, parameterized policy network
    critic: class, parameterized value network
    '''

    def __init__(self, env, actor, critic):

        self.env = env
        self.actor = actor
        self.critic = critic
        self.action_space = np.arange(env.action_space.n)

    def generate_episode(self):
        '''
        Outputs
        =====================================================
        states: list of arrays of states
        actions: list of actions
        rewards: list of rewards
        dones: list of boolean values indicating if an
            episode completed or not
        next_states: list of arrays of states
        '''

        states, actions, rewards, dones, next_states = [], [], [], [], []
        counter = 0
        total_count = self.batch_size * self.n_steps

        while counter < total_count:
            done = False
            while done == False:
                action_probs = actor.predict(np.array(self.s_0).reshape(1,-1))
                #action_probs = actor.predict(self.s_0)
                action_idx = np.random.choice(self.action_space, p=action_probs)
                #action_idx = self.action_space[np.argmax(action_probs)]
                action = env.A[action_idx]
                s_1, r, done, _ = self.env.step(action)
                self.reward += r
                states.append(self.s_0)
                next_states.append(s_1)
                actions.append(action_idx)
                rewards.append(r)
                dones.append(done)
                self.s_0 = np.atleast_2d(s_1)

                if done:
                    self.ep_rewards.append(self.reward)
                    self.s_0 = np.atleast_2d(self.env.reset())
                    self.reward = 0
                    self.ep_counter += 1

                counter += 1
                if counter >= total_count:
                    break
        return states, actions, rewards, dones, next_states

    def calc_rewards(self, batch):
        '''
        Inputs
        =====================================================
        batch: tuple of state, action, reward, done, and
            next_state values from generate_episode function

        Outputs
        =====================================================
        R: np.array of discounted rewards
        G: np.array of TD-error
        '''

        states, actions, rewards, dones, next_states = batch
        # Convert values to np.arrays
        rewards = np.array(rewards)
        states = np.vstack(states)
        next_states = np.vstack(states)
        actions = np.array(actions)
        dones = np.array(dones)

        total_steps = len(rewards)

        state_values = self.critic.predict(states)
        print(state_values.shape)
        next_state_values = self.critic.predict(next_states)
        next_state_values[dones] = 0

        R = np.zeros_like(rewards, dtype=np.float32)
        G = np.zeros_like(rewards, dtype=np.float32)

        for t in range(total_steps):
            last_step = min(self.n_steps, total_steps - t)

            # Look for end of episode
            check_episode_completion = dones[t:t + last_step]
            if check_episode_completion.size > 0:
                if True in check_episode_completion:
                    next_ep_completion = np.where(check_episode_completion == True)[0][0]
                    last_step = next_ep_completion

            # Sum and discount rewards
            R[t] = sum([rewards[t + n:t + n + 1] * self.gamma ** n for
                        n in range(last_step)])

        if total_steps > self.n_steps:
            R[:total_steps - self.n_steps] += next_state_values[self.n_steps:]

        G = R - state_values
        return R, G

    def train(self, n_steps=10, batch_size=10,
              num_episodes=15000, gamma=0.99):
        '''
        Inputs
        =====================================================
        n_steps: int defining the number of steps to take
            per batch
        batch_size: int sets number of batches to be
            completed before an update is done
        num_episodes: int sets the total number of episodes
            to be run
        gamma: float that defines the discount factor
        '''

        self.n_steps = n_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.ep_rewards = []
        self.policy_loss = []
        self.value_loss = []

        self.s_0 = self.env.reset()
        self.reward = 0
        self.ep_counter = 0
        compt = 0
        while self.ep_counter < num_episodes:
            compt+=20
            print(compt)
            batch = self.generate_episode()
            R, G = self.calc_rewards(batch)
            states = np.vstack(batch[0])
            actions = np.array(batch[1])

            policy_loss = self.actor.update(states, actions, G)
            value_loss = self.critic.update(states, R)

            print("\rMean Rewards: {:.2f} Episode: {:d}    ".format(
                np.mean(self.ep_rewards[-100:]), self.ep_counter), end="")

            if compt % int(num_episodes / 10) == 0:
                policy = np.zeros(env.nS)
                for state_idx in range(env.nS):
                    state = env.to_coordinate(state_idx)
                    action_probs = actor.predict(np.array(state).reshape(1, -1))
                    action_idx = self.action_space[np.argmax(action_probs)]
                    action = env.A[action_idx]
                    policy[state_idx] = action
                visualize_policy_RM(policy, env.T, env.C)

    def smooth_rewards(self):
        smoothed_rewards = [np.mean(self.ep_rewards[i:i + self.batch_size])
                            if i > self.batch_size
                            else np.mean(self.ep_rewards[0:i + 1]) for i in
                            range(len(self.ep_rewards))]
        return smoothed_rewards

if __name__ == '__main__':
    #env = gym.make('CartPole-v0')
    #env = gym.make('FrozenLake-v0')

    micro_times = 50
    capacity = 10
    actions = tuple(k for k in range(50, 231, 20))
    alpha = 0.7
    lamb = 0.7

    env = gym.make('gym_RM:RM-v0', micro_times=micro_times, capacity=capacity, actions=actions, alpha=alpha, lamb=lamb)

    tf.reset_default_graph()

    sess = tf.Session()

    # Set up networks
    actor = actor(sess, env)
    critic = critic(sess, env)

    init = tf.global_variables_initializer()
    sess.run(init)

    n_step_a2c = A2C(env, actor, critic)
    n_step_a2c.train(n_steps=100)

    plt.figure(figsize=(12, 8))
    plt.plot(n_step_a2c.ep_rewards, label='Rewards')
    plt.plot(n_step_a2c.smooth_rewards(), label='Smoothed Rewards')
    plt.title('Total Rewards')
    plt.legend(loc='best')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.ylim([0, 1.05 * max(n_step_a2c.ep_rewards)])
    plt.show()
