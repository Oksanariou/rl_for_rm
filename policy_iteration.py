import numpy as np


def extract_policy(env, U, gamma):
    policy_idx = np.zeros(env.nS, int)
    for state_idx in range(env.nS):
        state = env.to_coordinate(state_idx)

        list_sum = np.zeros(env.nA)
        for action_idx in range(env.nA):
            action = env.A[action_idx]
            for p, s_prime, r, _ in env.P[state][action]:
                list_sum[action_idx] += p * (r + gamma * U[env.to_idx(*s_prime)])
        policy_idx[state_idx] = np.argmax(list_sum)
        policy = [env.A[k] for k in policy_idx]
        # policy[s] = 50 + 20 * policy[s]

    return policy


def evaluate_policy(env, policy, gamma, epsilon):
    U = np.zeros(env.nS)
    while True:
        prev_U = np.copy(U)
        for state_idx in range(env.nS):
            state = env.to_coordinate(state_idx)
            a = policy[state_idx]
            U[state_idx] = sum([p * (r + gamma * prev_U[env.to_idx(*s_)]) for p, s_, r, _ in env.P[state][a]])
        if (np.sum(np.fabs(prev_U - U)) <= epsilon):
            break

    return U


def policy_iteration(env, gamma, max_iter, epsilon):
    policy_idx = np.random.choice(env.nA, env.nS)
    policy = [env.A[k] for k in policy_idx]
    for i in range(max_iter):
        U = evaluate_policy(env, policy, gamma, epsilon)
        new_policy = extract_policy(env, U, gamma)
        if (np.all(policy == new_policy)):
            print("Converged at " + str(i))
            break
        policy = new_policy
    return policy
