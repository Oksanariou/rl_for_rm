import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque

def q_to_policy_FL(Q):
    policy = []
    for l in Q:
        if l[0] == l[1] == l[2] == l[3] == 0.0:
            policy.append(0)
        else:
            for k in range(0, len(l)):
                if l[k] == max(l):
                    policy.append(k)
                    break
    return policy

def visualize_policy_FL(policy):
    visu = ''
    for k in range(len(policy)):
        if k > 0 and k%4 == 0:
            visu += '\n'
        if k == 5 or k == 7 or k == 11 or k == 12 or k == 15:
            visu+='H'
        elif int(policy[k]) == 0:
            visu += 'L'
        elif int(policy[k]) == 1:
            visu += 'D'
        elif int(policy[k]) == 2:
            visu += 'R'
        elif int(policy[k]) == 3:
            visu += 'U'
    print(visu)

def run_episode(env, policy):
    """ Runs an episode and returns the total reward """
    obs = env.reset()
    total_reward = 0
    while True:
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += reward
        if done:
            break
    return total_reward

def average_n_episodes(env, policy, n_eval):
    """ Runs n episodes and returns the average of the n total rewards"""
    scores = [run_episode(env, policy) for _ in range(n_eval)]
    return np.mean(scores)

def visualizing_epsilon_decay(nb_episodes, epsilon, epsilon_min, epsilon_decay):
    X = [k for k in range(nb_episodes)]
    Y = []
    for k in range(nb_episodes):
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        Y.append(epsilon)
    plt.plot(X, Y, 'b')
    plt.title("Decaying epsilon over the number of episodes")
    plt.xlabel("Number of episodes")
    plt.ylabel("Epsilon")
    plt.grid()
    return plt.show()

if __name__ == '__main__':

    env = gym.make('FrozenLake-v0')
    tf.reset_default_graph()

    inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([16,4],0,0.01))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)

    nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.initialize_all_variables()

    # Set learning parameters
    y = .99
    e = 0.1
    num_episodes = 2000
    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.9995
    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    memory = deque([], maxlen=100)
    batch_size = 16
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            print(i, epsilon)
            #Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Network
            while j < 99:
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                #Get new state and reward from environment
                s1,r,d,_ = env.step(a[0])

                #WITHOUT EXPERIENCE REPLAY
                """
                #Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
                #Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r + y*maxQ1
                #Train our network using target and predicted Q values
                _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
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

                rAll += r
                s = s1
                if d == True:
                    #Reduce chance of random action as we train the model.
                    #e = 1./((i/50) + 10)
                    break
            if (epsilon > epsilon_min):
                epsilon *= epsilon_decay
            jList.append(j)
            rList.append(rAll)
        Q_table = []
        for s in range(16):
            Q = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s:s + 1]})
            Q_table.append(Q[0])
        policy = q_to_policy_FL(Q_table)
        visualize_policy_FL(policy)
        print(average_n_episodes(env, policy, 100))
    print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
    plt.plot(rList)
    plt.show()