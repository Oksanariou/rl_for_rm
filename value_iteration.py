import numpy as np

def value_iteration(env, gamma, max_iter, epsilon):
    U = np.zeros(env.nS)
    for i in range(max_iter):
        prev_U = np.copy(U)
        for s in range(env.nS):
            q_sa = [sum([p*(r + prev_U[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            U[s] = max(q_sa)
        if (np.sum(np.fabs(prev_U - U)) <= epsilon):
            print("Converged at "+str(i))
            break
    return U

def extract_policy(env, U, gamma):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        list_sum = np.zeros(env.nA)
        for a in range(env.nA):
            for p, s_prime, r, _ in env.P[s][a]:
                list_sum[a] += p*(r+gamma*U[s_prime])
        policy[s] = np.argmax(list_sum)
        #policy[s] = 50 + 20 * policy[s]
    return policy