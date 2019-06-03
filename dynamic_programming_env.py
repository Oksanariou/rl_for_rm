import numpy as np


def compute_value(env, t, x, V):
    d_list = []
    if x < env.C - 1:
        r = V[t + 1, x + 1]
    else:
        r = 0
    for a in env.A:
        p, reward = env.proba_buy(a)
        d_list.append(p * (r + a) + (1 - p) * V[t + 1, x])
    return np.max(d_list)


def compute_policy(env, t, x, V):
    d_list = []
    if x < env.C - 1:
        r = V[t + 1, x + 1]
    else:
        r = 0
    for a in env.A:
        p, reward = env.proba_buy(a)
        d_list.append(p * (r + a) + (1 - p) * V[t + 1, x])
    action_idx = np.argmax(d_list)
    return env.A[action_idx]


def dynamic_programming_env(env):
    V = np.zeros((env.T, env.C), float)
    P = np.zeros((env.T, env.C), float)

    for t in range(env.T - 2, -1, -1):
        for x in range(env.C - 1):
            V[t, x] = compute_value(env, t, x, V)
            P[t, x] = compute_policy(env, t, x, V)

    return V, P
