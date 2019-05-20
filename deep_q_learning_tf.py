import numpy as np
import tensorflow as tf

def dql(env, gamma, nb_episodes, nb_steps, epsilon, epsilon_min, epsilon_decay):
    state_size = env.observation_space.n
    action_size = env.action_space.n

    inputs1 = tf.placeholder(shape=[1, state_size], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([state_size, action_size], 0, 0.01))
    Qout = tf.matmul(inputs1, W)
    predict = tf.argmax(Qout, 1)

    nextQ = tf.placeholder(shape=[1, action_size], dtype=tf.float32)
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
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            while j < nb_steps:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(state_size)[s:s + 1]})
                if np.random.rand(1) < epsilon:
                    a[0] = env.action_space.sample()
                # Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])

                #WITHOUT EXPERIENCE REPLAY

                # Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(state_size)[s1:s1 + 1]})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + gamma * maxQ1
                # Train our network using target and predicted Q values
                _, W1 = sess.run([updateModel, W],
                                 feed_dict={inputs1: np.identity(state_size)[s:s + 1], nextQ: targetQ})

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
                s = s1
                if d == True:
                    break
            if (epsilon > epsilon_min):
                epsilon *= epsilon_decay
            jList.append(j)
            rList.append(rAll)
        Q_table = []
        for s in range(state_size):
            Q = sess.run(Qout, feed_dict={inputs1: np.identity(state_size)[s:s + 1]})
            Q_table.append(Q[0])
    return Q_table, rList