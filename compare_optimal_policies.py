import gym
from q_learning import *
from deep_q_learning_tf_RM import *
from dynamic_programming import *

if __name__ == '__main__':

    env = gym.make('gym_RM:RM-v0')
    # env = gym.make('FrozenLake-v0')
    T, C = 50, 10

    prices, alpha, lamb = [k for k in range(50, 231, 20)], 0.4, 0.2
    V, P = dynamic_programming(T, C, alpha, lamb, prices)
    P = P.reshape(1, T * C)
    P = P[0]
    reward_DP = average_n_episodes(env, P, 50000)

    alpha, gamma, nb_steps = 0.07, 0.99, 10000
    epsilon, epsilon_min = 1, 0.01

    #nb_episodes = [1000, 10000, 50000, 100000]
    nb_episodes = [k for k in range(1000, 50001, 1000)]
    #eps_decay = [0.995, 0.9995, 0.9999, 0.99995]
    eps_decay = [0.995]
    avg_reward_ql = []
    avg_reward_dql = []
    average_reward_DP = [reward_DP for k in range(len(nb_episodes))]

    for k in range(len(nb_episodes)):
        print(k)
        epsilon_decay = eps_decay[k]

        q_table, rList = q_learning(env, alpha, gamma, nb_episodes[k], nb_steps, epsilon, epsilon_min, epsilon_decay)
        v_q = q_to_v(env, q_table)
        policy_q = extract_policy_RM(env, v_q, gamma)
        avg_reward_ql.append(average_n_episodes(env, policy_q, 50000))

        #q_table_dql, rList = dql(env, gamma, nb_episodes[k], nb_steps, epsilon, epsilon_min, epsilon_decay, T, C)
        #v_dql = q_to_v(env, q_table_dql)
        #policy_dql = extract_policy_RM(env, v_dql, gamma)
        #avg_reward_dql.append(average_n_episodes(env, policy_dql, 50000))

    plt.plot(nb_episodes, average_reward_DP, nb_episodes, avg_reward_ql)
    plt.legend(['Dynamic Programming', 'Q-Learning'])
    plt.xlabel('Number of episodes')
    plt.ylabel('Average reward over 50000 games')
    plt.title('Comparison of the performances of the optimal policies')
    plt.show()