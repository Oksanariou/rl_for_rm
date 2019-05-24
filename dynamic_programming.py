import numpy as np
from visualization_and_metrics import *


def d(f, a, lamb, A):
    return lamb * np.exp(-a * ((f / A[0]) - 1))


def V(t, x, a, lamb, C, VM, A):
    d_list = []
    if x < C - 1:
        r = VM[t + 1, x + 1]
    else:
        r = 0
    for f in A:
        d_list.append(d(f, a, lamb, A) * (f + r) + (1 - d(f, a, lamb, A)) * VM[t + 1, x])
    return np.max(d_list)


def P(t, x, a, lamb, C, VM, A):
    d_list = []
    if x < C - 1:
        r = VM[t + 1, x + 1]
    else:
        r = 0
    for f in A:
        d_list.append(d(f, a, lamb, A) * (f + r) + (1 - d(f, a, lamb, A)) * VM[t + 1, x])
    return np.argmax(d_list) * 20 + 50


def dynamic_programming(T, C, alpha, lamb, prices):
    VM = np.zeros((T, C), float)
    PM = np.zeros((T, C), float)

    for time in range(T - 2, -1, -1):
        for x in range(C - 1):
            VM[time, x] = V(time, x, alpha, lamb, C, VM, prices)
            PM[time, x] = P(time, x, alpha, lamb, C, VM, prices)

    return VM, PM
