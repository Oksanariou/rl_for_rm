import numpy as np

def value_iteration(env, max_iter, epsilon):
    U = np.zeros(env.nS)
    for i in range(max_iter):
        prev_U = np.copy(U)
        for state_idx in range(env.nS):
            state = env.to_coordinate(state_idx)
            q_sa = [sum([p * (r + prev_U[env.to_idx(*s_)]) for p, s_, r, _ in env.P[state][a]]) for a in env.A]
            U[state_idx] = max(q_sa)

        delta = np.sum(np.fabs(prev_U - U))
        #if delta > epsilon:
            #print("delta = {:.2}".format(delta))
        if delta <= epsilon:
            print("Converged at " + str(i))
            break

    return U

def value_iteration_discrete(env, max_iter = 100_000, epsilon = 1e-20, P={}):
    if P == {}:
        P = env.P
    U = np.zeros(env.nS)
    for i in range(max_iter):
        prev_U = np.copy(U)
        for state_idx in range(env.nS):
            q_sa = [sum([p * (r + prev_U[s_]) for p, s_, r, _ in P[state_idx][a]]) for a in range(env.nA)]
            U[state_idx] = max(q_sa)

        delta = np.sum(np.fabs(prev_U - U))
        #if delta > epsilon:
            #print("delta = {:.2}".format(delta))
        if delta <= epsilon:
            print("Converged at " + str(i))
            break

    return U

def value_iteration_discrete_collaboration(env, max_iter = 100_000, epsilon = 1e-20, P={}):
    if P == {}:
        P = env.P
    U = np.zeros(env.nS)
    for i in range(max_iter):
        prev_U = np.copy(U)
        for state_idx in range(env.nS):
            q_sa = [sum([p * (r[0] + r[1] + prev_U[s_]) for p, s_, r, _ in P[state_idx][a]]) for a in range(env.nA)]
            U[state_idx] = max(q_sa)

        delta = np.sum(np.fabs(prev_U - U))
        #if delta > epsilon:
            #print("delta = {:.2}".format(delta))
        if delta <= epsilon:
            print("Converged at " + str(i))
            break

    return U

