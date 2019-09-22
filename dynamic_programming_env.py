import numpy as np


def compute_d_list(env, t, x, V):
    d_list = []
    for a in env.A:
        p = env.P[env.to_idx(t,x)][env.A.index(a)][0][0]
        d_list.append(p * (V[t + 1, x + 1] + a) + (1 - p) * V[t + 1, x])
    return d_list


def compute_value(env, t, x, V):
    d_list = compute_d_list(env, t, x, V)
    return np.max(d_list)


def compute_policy(env, t, x, V):
    d_list = compute_d_list(env, t, x, V)
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


def compute_d_list_collaboration(env, t, x1, x2, V):
    d_list = []
    for action_idx in range(env.nA):

        state_idx = env.to_idx(t, x1, x2)

        p_nobuy = env.P[state_idx][action_idx][0][0]

        if len(env.P[state_idx][action_idx]) == 3:
            p_buys1 = env.P[state_idx][action_idx][1][0]
            p_buys2 = env.P[state_idx][action_idx][2][0]
            d_list.append(
                p_nobuy * (V[t + 1, x1, x2]) + p_buys1 * (V[t + 1, x1 + 1, x2] + env.A[action_idx][0]) + p_buys2 * (
                    V[t + 1, x1, x2 + 1] + env.A[action_idx][1]))

        elif len(env.P[state_idx][action_idx]) == 2:
            p_buys = env.P[state_idx][action_idx][1][0]
            if x1 == env.C1 - 1:
                d_list.append(
                    p_nobuy * (V[t + 1, x1, x2]) + p_buys * (
                    V[t + 1, x1, x2 + 1] + env.A[action_idx][1]))
            elif x2 == env.C2 - 1:
                d_list.append(
                    p_nobuy * (V[t + 1, x1, x2]) + p_buys * (
                        V[t + 1, x1 + 1, x2] + env.A[action_idx][0]))

    return d_list


def compute_value_collaboration(d_list):
    return np.max(d_list)


def compute_policy_collaboration(d_list):
    return np.argmax(d_list)


def dynamic_programming_collaboration(env):
    V = np.zeros((env.T, env.C1, env.C2), float)
    P = np.zeros((env.T, env.C1, env.C2), float)

    for t in range(env.T - 2, -1, -1):
        for x1 in range(env.C1 - 1, -1, -1):
            for x2 in range(env.C2):
                if x1 == env.C1 - 1 and x2 == env.C2 - 1:
                    break
                d_list = compute_d_list_collaboration(env, t, x1, x2, V)
                V[t][x1][x2] = compute_value_collaboration(d_list)
                P[t][x1][x2] = compute_policy_collaboration(d_list)

    return V, P

def dynamic_programming_collaboration_n_flights(env):
    V = np.zeros((env.T, env.C1, env.C2), float)
    P = np.zeros((env.T, env.C1, env.C2), float)

    for t in range(env.T - 2, -1, -1):
        for x1 in range(env.C1 - 1, -1, -1):
            for x2 in range(env.C2):
                if x1 == env.C1 - 1 and x2 == env.C2 - 1:
                    break
                d_list = compute_d_list_collaboration(env, t, x1, x2, V)
                V[t][x1][x2] = compute_value_collaboration(d_list)
                P[t][x1][x2] = compute_policy_collaboration(d_list)

    return V, P
