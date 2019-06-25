from collections import deque

import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm

from dynamic_programming import dynamic_programming
from q_learning import q_to_v
from visualization_and_metrics import visualisation_value_RM, extract_policy_RM, visualize_policy_RM, average_n_episodes


def compute_q_table(env, sess, tf_input_state, tf_output_Q):
    shape = [space.n for space in env.observation_space]

    states = [np.asarray((t, x)) for t in range(shape[0]) for x in range(shape[1])]

    return sess.run(tf_output_Q, feed_dict={tf_input_state: states})


def get_target_q_value(sess, reward, gamma, tf_input_state, tf_output_Q, next_state):
    next_act_values = sess.run(tf_output_Q, feed_dict={tf_input_state: next_state})

    return reward + gamma * np.amax(next_act_values)


def get_true_q_value(reward, gamma, V, next_state):
    return reward + gamma * V[next_state[0][0], next_state[0][1]]


def dql(env, gamma, nb_episodes, epsilon, epsilon_min, epsilon_decay):
    state_size = 2
    action_size = env.nA
    layer_size = 10

    V, P = dynamic_programming(env.T, env.C, env.alpha, env.lamb, env.A)

    tf_input_state = tf.placeholder(shape=[None, state_size], dtype=tf.float32)

    w1 = tf.Variable(tf.random_normal([state_size, layer_size]))
    b1 = tf.Variable(tf.random_normal([layer_size]))
    first_layer = tf.nn.sigmoid(tf.matmul(tf_input_state, w1) + b1)

    w2 = tf.Variable(tf.random_normal([layer_size, layer_size]))
    b2 = tf.Variable(tf.random_normal([layer_size]))
    second_layer = tf.nn.sigmoid(tf.matmul(first_layer, w2) + b2)

    w3 = tf.Variable(tf.random_normal([layer_size, env.nA]))
    b3 = tf.Variable(tf.random_normal([env.nA]))

    tf_output_Q = tf.matmul(second_layer, w3) + b3

    tf_best_action = tf.argmax(tf_output_Q, 1)

    tf_nextQ = tf.placeholder(shape=[1, action_size], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(tf_nextQ - tf_output_Q))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.global_variables_initializer()

    memory = deque(maxlen=1000)
    batch_size = 500

    with tf.Session() as sess:
        sess.run(init)
        for episode in tqdm(range(nb_episodes)):
            # Reset environment and get first new observation
            state = env.set_random_state()
            # state = env.reset()
            state = np.reshape(state, [1, state_size])

            done = False

            while not done:
                # Choose an action by greedily (with e chance of random action) from the Q-network
                best_action, act_values = sess.run([tf_best_action, tf_output_Q], feed_dict={tf_input_state: state})
                best_action = best_action[0]

                action = best_action if np.random.rand() > epsilon else env.action_space.sample()

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                memory.append((state, action, reward, next_state, done))

                if len(memory) > batch_size:
                    minibatch = random.sample(memory, batch_size)

                    state_batch, act_values_batch = [], []

                    for state, action, reward, next_state, done in minibatch:
                        act_values = sess.run(tf_output_Q, feed_dict={tf_input_state: state})

                        target_q_value = get_true_q_value(reward, gamma, V, next_state)
                        targetQ = act_values.copy()
                        targetQ[0, action] = reward if done else target_q_value

                        state_batch.append(state[0])
                        act_values_batch.append(act_values[0])

                    sess.run(updateModel, feed_dict={tf_input_state: state, tf_nextQ: targetQ})

                state = next_state

                epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if episode % int(nb_episodes / 10) == 0:
                Q = compute_q_table(env, sess, tf_input_state, tf_output_Q)
                v = q_to_v(env, Q)
                visualisation_value_RM(v, env.T, env.C)
                policy = extract_policy_RM(env, v, gamma)
                visualize_policy_RM(policy, env.T, env.C)

                N = 1000
                revenue = average_n_episodes(env, policy, N)
                print("Average reward over {} episodes after {} episodes : {}".format(N, episode, revenue))

        Q = compute_q_table(env, sess, tf_input_state, tf_output_Q)

    return Q


# WITH EXPERIENCE REPLAY
"""
memory.append((s_2d, a[0], r, s1_2d, d))
if (len(memory)>batch_size):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            Q = sess.run(Qout,feed_dict={inputs1: next_state})
            target = reward + gamma*np.max(Q)
        targetQ = sess.run(Qout, feed_dict={inputs1: state})
        targetQ[0, a[0]] = target
        _, W1 = sess.run([updateModel, W],feed_dict={inputs1: state, nextQ: targetQ})
"""
