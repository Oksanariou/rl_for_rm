
"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import time

def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
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

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def extract_policy_VI(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                print(env.P)
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-20
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 100000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma) #Compute the utility given the actual policy
        new_policy = extract_policy(old_policy_v, gamma) #Compute the new policy given the utility
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    print(policy)
    return policy

def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v

def tuning_gamma(env):
    gamma = []
    score = []
    score_VI = []
    k = 0
    for i in range(1,100,1):
        k = i/100
        gamma.append(k)
        optimal_v = value_iteration(env, k);
        optimal_policy_VI = extract_policy_VI(optimal_v, k)
        optimal_policy = policy_iteration(env, k)
        scores = evaluate_policy(env, optimal_policy, k, n=2000)
        scores_VI = evaluate_policy(env, optimal_policy_VI, k, n=2000)
        score.append(scores)
        score_VI.append(scores_VI)
    plt.plot(gamma, score, 'r+', gamma, score_VI, 'b+')
    legend = ['Policy iteration','Value iteration']
    plt.legend(legend)
    plt.title("Influence of gamma on the optimal policy")
    plt.xlabel("Gamma")
    plt.ylabel("Expected total reward")
    plt.grid()
    axes = plt.gca()
    axes.set_xlim(0,1.2)
    axes.set_ylim(-6.5,2)
    return plt.show()

def average_time(env, gamma, n):
    av_time_VI = 0
    av_time_PI = 0
    for k in range(n):
        print(k)
        VI_0 = time.time()
        optimal_v = value_iteration(env, gamma);
        policy_VI = extract_policy_VI(optimal_v, gamma)
        VI_1 = time.time()
    
        PI_0 = time.time()
        optimal_policy = policy_iteration(env, gamma)
        PI_1 = time.time()

        diff_VI = VI_1 - VI_0
        diff_PI = PI_1  - PI_0
        minutes_VI, seconds_VI = diff_VI // 60, diff_VI % 60
        minutes_PI, seconds_PI = diff_PI // 60, diff_PI % 60

        av_time_VI += seconds_VI
        av_time_PI += seconds_PI
    av_time_VI = av_time_VI / n
    av_time_PI = av_time_PI / n
    print('Average time VI : ', av_time_VI)
    print('Average time PI : ', av_time_PI)

def gamma_time(env):
    gamma = []
    time_VI = []
    time_PI = []
    for i in range(1,100,10):
        print(i)
        k = i/100
        gamma.append(k)
        VI_0 = time.time()
        optimal_v = value_iteration(env, k);
        policy_VI = extract_policy_VI(optimal_v, k)
        VI_1 = time.time()
    
        PI_0 = time.time()
        optimal_policy = policy_iteration(env, k)
        PI_1 = time.time()

        diff_VI = VI_1 - VI_0
        diff_PI = PI_1  - PI_0
        minutes_VI, seconds_VI = diff_VI // 60, diff_VI % 60
        minutes_PI, seconds_PI = diff_PI // 60, diff_PI % 60

        time_VI.append(seconds_VI)
        time_PI.append(seconds_PI)

    plt.plot(gamma, time_PI, 'r+', gamma, time_VI, 'b+')
    legend = ['Policy iteration','Value iteration']
    plt.legend(legend)
    plt.title("Influence of gamma on the time needed to compute the optimal policy")
    plt.xlabel("Gamma")
    plt.ylabel("Time (s)")
    plt.grid()
    axes = plt.gca()
    return plt.show()

def visualize_policy(policy):
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

if __name__ == '__main__':
    env_name  = 'FrozenLake-v0'
    #env_name  = 'FrozenLake8x8-v0'
    gamma = 1
    env = gym.make(env_name)
    env = env.unwrapped

    pol = policy_iteration(env, gamma)
    visualize_policy(pol)
    #extract_policy_VI(v, gamma)
    #tuning_gamma(env)
    #average_time(env, gamma, 100)
    #gamma_time(env)


