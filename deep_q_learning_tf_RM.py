import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from collections import deque
from tqdm import tqdm
import random

def state_nb_to_state_coordinates(C, s):
    t = s//C
    x = s - t*C
    return np.array([[t,x]])

def to_categorical_2(T, C, s):
    t = s // C
    x = s - t * C
    encoded_t = to_categorical([t], max(T, C))
    encoded_x = to_categorical([x], max(T, C))
    new_s = np.array([encoded_t, encoded_x])
    return new_s

def state_one_hot_encoded(T, C, s):
    matrix = np.zeros((T, C), float)
    t = s//C
    x = s - t * C
    matrix[t,x] = 1.
    matrix = matrix.reshape(1, T, C)
    return matrix

def dql(env, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay, T, C):
    state_size = env.observation_space.n
    action_size = env.action_space.n

    inputs1 = tf.placeholder(shape=[None, T, C], dtype=tf.float32)
    flattened = tf.layers.flatten(inputs1)
    W = tf.Variable(tf.random_uniform([state_size, action_size], 0, 0.01))
    Qout = tf.matmul(flattened, W)
    predict = tf.argmax(Qout, 1)

    nextQ = tf.placeholder(shape=[1, action_size], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.initialize_all_variables()

    memory = deque(maxlen=100)
    batch_size = 16
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    with tf.Session() as sess:
        sess.run(init)
        for i in tqdm(range(nb_episodes)):
            # Reset environment and get first new observation
            s = env.reset()
            s_2d = state_one_hot_encoded(T, C, s)
            rAll = 0
            d = False
            j = 0
            while j < nb_steps:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1: s_2d})
                if np.random.rand(1) < epsilon:
                    a[0] = env.action_space.sample()
                # Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                s1_2d = state_one_hot_encoded(T, C, s1)

                #WITHOUT EXPERIENCE REPLAY

                # Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout, feed_dict={inputs1: s1_2d})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + gamma * maxQ1
                # Train our network using target and predicted Q values
                _, W1 = sess.run([updateModel, W],
                                 feed_dict={inputs1: s_2d, nextQ: targetQ})

                #WITH EXPERIENCE REPLAY
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
                rAll += r
                s_2d = s1_2d
                if d == True:
                    break
            if (epsilon > epsilon_min):
                epsilon *= epsilon_decay
            jList.append(j)
            rList.append(rAll)
        Q_table = []
        for s in range(T*C):
            s_2d = state_one_hot_encoded(T, C, s)
            Q = sess.run(Qout, feed_dict={inputs1: s_2d})
            Q_table.append(Q[0])
    return Q_table, rList