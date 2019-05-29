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
        # if delta > epsilon:
        #     print("delta = {:.2}".format(delta))
        # else:
        #     print("Converged at " + str(i))
        #     break

    return U


def extract_policy(env, U, gamma):
    policy = np.zeros(env.nS)
    for state_idx in range(env.nS):
        state = env.to_coordinate(state_idx)
        list_sum = []
        for a in env.A:
            for p, s_prime, r, _ in env.P[state][a]:
                list_sum.append(p * (r + gamma * U[env.to_idx(*s_prime)]))
        policy[state_idx] = np.argmax(list_sum)
        # policy[state_idx] = 50 + 20 * policy[state_idx]

    return policy
