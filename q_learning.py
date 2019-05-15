import gym
import numpy as np
import matplotlib.pyplot as plt

def q_learning(env, lr, y, num_episodes):
    #Initialize table with all zeros
    #Q = np.ones([env.observation_space.n,env.action_space.n]) #optimism in the face of uncertainty (1/2)
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    
    #create lists to contain total rewards
    rList = []
    
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        
        #The Q-Table learning algorithm
        while j < 99:
            j+=1
            #Choose an action by  picking from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1))) #epsilon-greedy: used for the tuning of the parameters
            #a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./((i+1)*100))) #epsilon-greedy, epsilon decaying quickly
            #a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./((i+1)*100))) #epsilon-greedy, epsilon decaying slowly
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)) #epsilon not decaying over time
            #a = np.argmax(Q[s,:]) #optimism in the face of uncertainty (2/2)
            
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            if d == True:
                break
            rList.append(rAll)
        """
        policy = []
        for l in Q:
            if l[0] == l[1] == l[2] == l[3] == 0.0:
                policy.append(0)
            else:
                for k in range(0, len(l)):
                    if l[k] == max(l):
                        policy.append(k)
                        break
        """
    return Q

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and returns the total reward """
    gamma = 1.0
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def tuning_lr(env, y, num_episodes):
    lr = []
    scores = []
    for i in range(1,100,5):
        k = i/100
        lr.append(k)
        policy = q_learning(env, k , y, num_episodes)
        score = evaluate_policy(env, policy, gamma=1, n=5000)
        scores.append(score)
    plt.plot(lr, scores, 'b+')
    plt.title("Influence of the learning rate on the optimal policy")
    plt.xlabel("Learning rate")
    plt.ylabel("Average total reward")
    plt.grid()
    return plt.show()

def tuning_gamma(env, lr, num_episodes):
    gamma = []
    scores = []
    for i in range(1,100,5):
        k = i/100
        gamma.append(k)
        policy = q_learning(env, lr , k, num_episodes)
        score = evaluate_policy(env, policy, gamma=1, n=5000)
        scores.append(score)
    plt.plot(gamma, scores, 'b+')
    plt.title("Influence of gamma on the optimal policy")
    plt.xlabel("Gamma")
    plt.ylabel("Average total reward")
    plt.grid()
    return plt.show()

def tuning_nb_episodes(env, lr, y):
    nb_episodes = []
    scores = []
    for i in range(1,60000,1000):
        nb_episodes.append(i)
        policy = q_learning(env, lr , y, i)
        score = evaluate_policy(env, policy, gamma=1, n=5000)
        scores.append(score)
    plt.plot(nb_episodes, scores, 'b+')
    plt.title("Influence of the number of episodes on the optimal policy")
    plt.xlabel("Number of episodes")
    plt.ylabel("Average total reward")
    plt.grid()
    return plt.show()

def running_q_learning_n_times(n):
    scores=[]
    for k in range(n):
        print(k)
        policy = q_learning(env, lr, y, num_episodes)
        score = evaluate_policy(env, policy, gamma=1, n=10000)
        scores.append(score)
    print(np.mean(scores))

def extract_policy(Q):
    policy = []
    for l in Q:
        if l[0] == l[1] == l[2] == 0.0:
            policy.append(5)
        else:
            for k in range(0, len(l)):
                if l[k] == max(l):
                    policy.append(k)
                    break
    return policy

def visualisation_policy(P, T, C):
    P = P.reshape(T, C)
    plt.imshow(P)
    plt.title("Prices coming from the optimal policy")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.colorbar()

    return plt.show()

if __name__ == '__main__':
    #env = gym.make('FrozenLake8x8-v0')
    env = gym.make('gym_RM:RM-v0')
    print(env.P)
    # Set learning parameters
    lr = 0.05
    y = .99
    num_episodes = 200000

    Q = q_learning(env, lr, y, num_episodes)
    policy = np.array(extract_policy(Q))
    T, C = 150, 50
    visualisation_policy(policy, T, C)
    print(evaluate_policy(env, policy, gamma=1.0, n=100))
    #tuning_gamma(env,lr, num_episodes)
    #tuning_lr(env, y, num_episodes)
    #tuning_nb_episodes(env, lr, y)
    #running_q_learning_n_times(100)

