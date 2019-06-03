import numpy as np
import scipy.special


def compute_value(env, t, x, V):
    d_list = []
    for a in env.A:
        p, reward = env.proba_buy(a)
        sum = 0
        for k in range(env.M):
            if x + k < env.C - 1:
                r = V[t + 1, x + k]
            else:
                r = 0
            sum += ((1 - p) ** (env.M-1 - k)) * (p ** k) * (r + k * a) * scipy.special.binom(env.M-1, k)
        d_list.append(sum)
    return np.max(d_list)


def compute_policy(env, t, x, V):
    d_list = []
    for a in env.A:
        p, reward = env.proba_buy(a)
        sum = 0
        for k in range(env.M):
            if x + k < env.C - 1:
                r = V[t + 1, x + k]
            else:
                r = 0
            sum += ((1 - p) ** (env.M-1 - k)) * (p ** k) * (r + k * a) * scipy.special.binom(env.M-1, k)
        d_list.append(sum)
    action_idx = np.argmax(d_list)
    return env.A[action_idx]


def dynamic_programming_env_DCP(env):
    V = np.zeros((env.T, env.C), float)
    P = np.zeros((env.T, env.C), float)

    for t in range(env.T - 2, -1, -1):
        for x in range(env.C - 1):
            V[t, x] = compute_value(env, t, x, V)
            P[t, x] = compute_policy(env, t, x, V)

    return V, P
