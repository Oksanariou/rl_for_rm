import numpy as np
import tensorflow as tf

def state_nb_to_state_coordinates(C, s):
    t = s//C
    x = s - t*C
    return np.array([[t,x]])

def dql(env, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay, C):
    state_size = env.observation_space.n
    action_size = env.action_space.n

    inputs1 = tf.placeholder(shape=(1,2), dtype=tf.float32)
    w1 = tf.Variable(tf.random_uniform([20, 2], 0, 0.01))
    y1 = tf.matmul(w1, tf.transpose(inputs1))
    wo = tf.Variable(tf.random_uniform([action_size, 20], 0, 0.01))
    Qout = tf.matmul(tf.transpose(y1), tf.transpose(wo))
    print(Qout)
    predict = tf.argmax(Qout, 1)

    nextQ = tf.placeholder(shape=[1, action_size], dtype=tf.float32)
    print(nextQ)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.initialize_all_variables()

    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(nb_episodes):
            print(i)
            # Reset environment and get first new observation
            s_1D = env.reset()
            s = state_nb_to_state_coordinates(C, s_1D)
            rAll = 0
            d = False
            j = 0
            while j < nb_steps:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1: s})
                if np.random.rand(1) < epsilon:
                    a[0] = env.action_space.sample()
                # Get new state and reward from environment
                s1_1D, r, d, _ = env.step(a[0])
                s1 = state_nb_to_state_coordinates(C, s1_1D)
                #WITHOUT EXPERIENCE REPLAY

                # Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout, feed_dict={inputs1: s1})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + gamma * maxQ1
                # Train our network using target and predicted Q values
                sess.run(updateModel, feed_dict={inputs1: s, nextQ: targetQ})

                """
                #WITH EXPERIENCE REPLAY
                memory.append((s, a[0], r, s1, d))
                if (len(memory)>batch_size):
                    minibatch = random.sample(memory, batch_size)
                    for state, action, reward, next_state, done in minibatch:
                        target = reward
                        if not done:
                            Q = sess.run(Qout,feed_dict={inputs1:np.identity(16)[next_state:next_state+1]})
                            target = reward + y*np.max(Q)
                        targetQ = sess.run(Qout, feed_dict={inputs1: np.identity(16)[state:state + 1]})
                        targetQ[0, a[0]] = target
                        _, W1 = sess.run([updateModel, W],feed_dict={inputs1: np.identity(16)[state:state + 1], nextQ: targetQ})
                """
                rAll += r
                s_1D = s1_1D
                if d == True:
                    break
            if (epsilon > epsilon_min):
                epsilon *= epsilon_decay
            jList.append(j)
            rList.append(rAll)
        Q_table = []
        for s_1D in range(state_size):
            s = state_nb_to_state_coordinates(C,s_1D)
            Q = sess.run(Qout, feed_dict={inputs1: s})
            Q_table.append(Q[0])
    return Q_table, rList