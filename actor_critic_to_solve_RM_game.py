"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
from tqdm import tqdm

import tensorflow as tf

import random
from collections import deque

from visualization_and_metrics import visualize_policy_RM


# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
    def __init__(self, env, sess, epsilon, epsilon_decay, gamma, tau, lr):
        self.env = env
        self.sess = sess

        self.learning_rate = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = deque(maxlen=20)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        # self.actor_critic_grad = tf.placeholder(tf.float32,
        #                                         [None, self.env.action_space.shape[
        #                                             0]])  # where we will feed de/dC (from critic)
        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, env.action_space.n])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        #state_input = Input(shape=self.env.observation_space.shape)
        state_input = Input(shape=(len(self.env.observation_space.spaces),))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        #output = Dense(self.env.action_space.shape[0], activation='relu')(h3)
        output = Dense(self.env.action_space.n, activation='sigmoid')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        #state_input = Input(shape=self.env.observation_space.shape)
        state_input = Input(shape=(len(self.env.observation_space.spaces),))
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        #action_input = Input(shape=self.env.action_space.shape)
        action_input = Input(shape=(self.env.action_space.n,))
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            #predicted_action = np.array(np.argmax(predicted_action_proba)).reshape(1,1)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            action_idx = np.argmax(action)
            action_one_hot = np.zeros(4)
            action_one_hot[action_idx] = 1
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                target_action_idx = np.argmax(target_action)
                target_action_one_hot = np.zeros(4)
                target_action_one_hot[target_action_idx] = 1
                #target_action = np.array(np.argmax(target_action_proba)).reshape(1,1)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action_one_hot])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action_one_hot], reward, verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        if np.random.random() < self.epsilon:
            probas = np.zeros(4)
            action = self.env.action_space.sample()
            probas[action] = 1
            return probas
        probas = self.actor_model.predict(cur_state)
        return probas


def train_actor_critic(env, nb_episodes, epsilon, epsilon_decay, gamma, tau, lr):

    sess = tf.Session()
    K.set_session(sess)
    actor_critic = ActorCritic(env, sess, epsilon, epsilon_decay, gamma, tau, lr)

    for k in tqdm(range(nb_episodes)):
        cur_state = env.set_random_state()
        done = False
        total_reward = 0
        while not done:
            #cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            cur_state = np.array(cur_state).reshape((1, 2))
            action_proba = np.array(actor_critic.act(cur_state))
            action_idx = np.argmax(action_proba)
            action = env.A[action_idx]
            action_one_hot = np.zeros(4)
            action_one_hot[action_idx] = 1
            action_one_hot = action_one_hot.reshape(1,4)
            action_proba = action_proba.reshape(1, 4)
            #print(action)

            new_state, reward, done, _ = env.step(action)
            total_reward += reward
            new_state = np.array(new_state).reshape((1, 2))
            reward = np.array(reward).reshape((1,))

            actor_critic.remember(cur_state, action_one_hot, reward, new_state, done)
            actor_critic.train()

            cur_state = new_state

        actor_critic.epsilon = actor_critic.epsilon * actor_critic.epsilon_decay
    final_policy = []
    for state_idx in range(env.nS):
        state = env.to_coordinate(state_idx)
        state = np.array(state).reshape((1,2))
        proba = actor_critic.actor_model.predict(state)
        action_idx = np.argmax(proba)
        final_policy.append(env.A[action_idx])
    return final_policy

