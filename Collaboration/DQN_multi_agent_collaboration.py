import numpy as np

from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.dqn import agent_builder

from Collaboration.DQN_single_agent import train, update_target
# from Collaboration.DQN_multi_agent import callback_multi_agent
from visualization_and_metrics import average_n_episodes


def play_game_multi_agent_collaboration(policy1, policy2, env1, env2, collaboration_env):

    obs = collaboration_env.reset()
    t, x1, x2 = collaboration_env.to_coordinate(obs)
    obs1, obs2 = env1.to_idx(t, x1), env2.to_idx(t, x2)

    total_reward = 0

    while True:
        action1, action2 = policy1[obs1], policy2[obs2]
        action_tuple = (env1.A[action1], env2.A[action2])
        action = collaboration_env.A.index(action_tuple)

        new_obs, rew, done, _ = collaboration_env.step(action)

        total_reward += rew[0] + rew[1]
        done1, done2 = done[0], done[1]
        new_t, new_x1, new_x2 = collaboration_env.to_coordinate(new_obs)
        new_obs1, new_obs2 = env1.to_idx(new_t, new_x1), env2.to_idx(new_t, new_x2)
        obs1 = new_obs1
        obs2 = new_obs2

        if done1 and done2:
            break
    return total_reward

def learn_multi_agent_collaboration(agent1, agent2, env1, env2, total_timesteps, collaboration_env):
    agent1.num_timesteps = 0
    agent1._setup_learn(seed=None)
    agent1.replay_buffer = ReplayBuffer(agent1.buffer_size)
    agent1.exploration = LinearSchedule(schedule_timesteps=int(agent1.exploration_fraction * total_timesteps),
                                        initial_p=1.0,
                                        final_p=agent1.exploration_final_eps)
    agent2.num_timesteps = 0
    agent2._setup_learn(seed=None)
    agent2.replay_buffer = ReplayBuffer(agent2.buffer_size)
    agent2.exploration = LinearSchedule(schedule_timesteps=int(agent2.exploration_fraction * total_timesteps),
                                        initial_p=1.0,
                                        final_p=agent2.exploration_final_eps)

    obs = collaboration_env.reset()
    t, x1, x2 = collaboration_env.to_coordinate(obs)
    obs1, obs2 = env1.to_idx(t, x1), env2.to_idx(t, x2)

    for step in range(total_timesteps):
        print(step)
        if (step % (total_timesteps // 100)) == 0:
            print(revenues1, revenues2)

        # callback(locals(), globals())

        update_eps1 = agent1.exploration.value(agent1.num_timesteps)
        update_eps2 = agent2.exploration.value(agent2.num_timesteps)

        with agent1.sess.as_default():
            action1 = agent1.act(np.array(obs1)[None], update_eps=update_eps1)[0]
        with agent2.sess.as_default():
            action2 = agent2.act(np.array(obs2)[None], update_eps=update_eps2)[0]

        action_tuple = (env1.A[action1], env2.A[action2])
        action = collaboration_env.A.index(action_tuple)

        new_obs, rew, done, _ = collaboration_env.step(action)

        new_t, new_x1, new_x2 = collaboration_env.to_coordinate(new_obs)
        new_obs1, new_obs2 = env1.to_idx(new_t, new_x1), env2.to_idx(new_t, new_x2)
        rew1, rew2 = rew[0] + rew[1], rew[0] + rew[1]
        done1, done2 = done[0], done[1]

        agent1.replay_buffer.add(obs1, action1, rew1, new_obs1, float(done1))
        agent2.replay_buffer.add(obs2, action2, rew2, new_obs2, float(done2))

        obs1 = new_obs1
        obs2 = new_obs2

        if done1 and done2:
            obs = collaboration_env.reset()
            t, x1, x2 = collaboration_env.to_coordinate(obs)
            obs1, obs2 = env1.to_idx(t, x1), env2.to_idx(t, x2)

        train(agent1)
        train(agent2)

        update_target(agent1)
        update_target(agent2)

        agent1.num_timesteps += 1
        agent2.num_timesteps += 1
    return agent1, agent2


def run_DQN_multi_agent_collaboration(parameters_dict1, parameters_dict2, total_timesteps,
                                      collaboration_env):
    global n_steps, revenues1, revenues2, states1, states2, agent1, agent2, env1, env2

    env1, env2 = parameters_dict1["env_builder"](), parameters_dict2["env_builder"]()

    env_vec1, env_vec2 = DummyVecEnv([lambda: env1]), DummyVecEnv([lambda: env2])

    n_steps = 0
    revenues1, revenues2 = [], []
    states1 = [k for k in range(env1.T * env1.C)]
    states2 = [k for k in range(env2.T * env2.C)]

    agent1 = agent_builder(env_vec1, parameters_dict1)
    agent2 = agent_builder(env_vec2, parameters_dict2)

    learn_multi_agent_collaboration(agent1, agent2, env1, env2, total_timesteps,
                                    collaboration_env)
