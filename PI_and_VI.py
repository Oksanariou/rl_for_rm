
"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from matplotlib import colors
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


def evaluate(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(U):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        list_sum = np.zeros(env.nA)
        for a in range(env.nA):
            for p, s_prime, r, _ in env.P[s][a]:
                list_sum[a] += p*(r+U[s_prime])
        policy[s] = np.argmax(list_sum)
    return policy

def evaluate_policy(env, policy, gamma, epsilon):
    U = np.zeros(env.nS)
    while True:
        prev_U = np.copy(U)
        for s in range(env.nS):
            a = policy[s]
            U[s] = sum([p * (r + gamma * prev_U[s_]) for p, s_, r, _ in env.P[s][a]])
            #for p, s_prime, r, _ in env.P[s][a]:
                #U[s] += p*(r + gamma*prev_U[s_prime])
        if (np.sum(np.fabs(prev_U - U)) <= epsilon):
            break
    return U

def policy_iteration(env, gamma, max_iter, epsilon):
    policy = np.random.choice(env.nA, env.nS)
    for i in range(max_iter):
        U = evaluate_policy(env, policy, gamma, epsilon)
        new_policy = extract_policy(U)
        if (np.all(policy == new_policy)):
            print("Converged at " + str(i))
            break
        policy = new_policy
    return policy

def value_iteration(env, gamma, max_iter, epsilon):
    U = np.zeros(env.nS)
    for i in range(max_iter):
        prev_U = np.copy(U)
        for s in range(env.nS):
            list_sum = np.zeros(env.nA)
            for a in range(env.nA):
                for p, s_prime, r, _ in env.P[s][a]:
                    list_sum[a] += p*(r + gamma*prev_U[s_prime])
            U[s] = max(list_sum)
        if (np.sum(np.fabs(prev_U - U)) <= epsilon):
            print("Converged at "+str(i))
            break
    return U

def plot_rev(X, alpha):
    R = []
    for p in X:
        R.append(p*10 * np.exp(-alpha * ((p / X[-1]) - 1)))
    plt.plot(X, R, 'r+')
    plt.title("Revenue as a function of price, alpha = "+str(alpha))
    plt.xlabel("Prices")
    plt.grid()
    return plt.show()

def visualisation_value(V, T, C):
    V = V.reshape(T, C)
    print(V)
    plt.title("Values of the states")
    plt.xlabel('Capacity')
    plt.ylabel('Number of micro-times')
    plt.imshow(V)
    plt.colorbar()

    return plt.show()

def visualisation_policy(P, T, C):
    P = P.reshape(T, C)
    plt.imshow(P)
    plt.title("Prices coming from the optimal policy")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of micro-times')
    plt.colorbar()

    return plt.show()

if __name__ == '__main__':
    gamma = 0.99
    env = gym.make('gym_RM:RM-v0')
    #env = env.unwrapped
    T, C = 300, 50

    v = value_iteration(env, gamma, 100000, 1e-20)
    policy = extract_policy(v)
    #visualisation_value(v, T, C)
    #policy = policy_iteration(env, gamma, 100000, 1e-20)
    visualisation_policy(policy, T, C)

    #print(evaluate(env, policy, 100))

    #X = [170, 140, 110, 80, 50]
    #plot_rev(X, 0.4)