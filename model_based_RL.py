import gym
import random
from sklearn.neural_network import MLPClassifier


def collect_experience(env, size_of_experience):
    input, output = [], []
    for k in range(size_of_experience):
        state_idx = env.set_random_state()
        action_idx = random.randrange(env.action_space.n)
        action = env.A[action_idx]
        next_state, reward, done, _ = env.step(action)

        input.append([state_idx, action])
        output.append(next_state)
    return input, output


if __name__ == '__main__':
    data_collection_points = 10
    micro_times = 5
    capacity = 10
    actions = tuple(k for k in range(50, 231, 50))
    alpha = 0.8
    lamb = 0.7
    env = gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points,
                   capacity=capacity,
                   micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)


