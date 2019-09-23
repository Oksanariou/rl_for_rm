import numpy as np
import scipy.special


def compute_d_list(env, t, x, V):
    d_list = []
    for a in env.A:
        p, reward = env.proba_buy(a)
        sum = 0
        for k in range(env.M+1):
            r = k * a + V[t + 1, x + k] if x + k <= env.C - 1 else V[t + 1, env.C - 1] + a * (env.C - 1 - x)
            sum += ((1 - p) ** (env.M - k)) * (p ** k) * r * scipy.special.binom(env.M, k)
        d_list.append(sum)
    return d_list


def compute_value(env, t, x, V):
    d_list = compute_d_list(env, t, x, V)
    return np.max(d_list)


def compute_policy(env, t, x, V):
    d_list = compute_d_list(env, t, x, V)
    return np.argmax(d_list)


def dynamic_programming_env_DCP(env):
    V = np.zeros((env.T, env.C), float)
    P = np.zeros((env.T, env.C), float)

    for t in range(env.T - 2, -1, -1):
        for x in range(env.C - 1):
            V[t, x] = compute_value(env, t, x, V)
            P[t, x] = compute_policy(env, t, x, V)

    return V, P
