import numpy as np

def extract_policy(env, U, gamma):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        list_sum = np.zeros(env.nA)
        for a in range(env.nA):
            for p, s_prime, r, _ in env.P[s][a]:
                list_sum[a] += p*(r+gamma*U[s_prime])
        policy[s] = np.argmax(list_sum)
    return policy

def evaluate_policy(env, policy, gamma, epsilon):
    U = np.zeros(env.nS)
    while True:
        prev_U = np.copy(U)
        for s in range(env.nS):
            a = policy[s]
            U[s] = sum([p * (r + gamma * prev_U[s_]) for p, s_, r, _ in env.P[s][a]])
        if (np.sum(np.fabs(prev_U - U)) <= epsilon):
            break
    return U

def policy_iteration(env, gamma, max_iter, epsilon):
    policy = np.random.choice(env.nA, env.nS)
    for i in range(max_iter):
        U = evaluate_policy(env, policy, gamma, epsilon)
        new_policy = extract_policy(env, U, gamma)
        if (np.all(policy == new_policy)):
            print("Converged at " + str(i))
            break
        policy = new_policy
    return policy