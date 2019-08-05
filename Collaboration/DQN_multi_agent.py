import numpy as np

from visualization_and_metrics import average_n_episodes
from Collaboration.DQN_single_agent import train, update_target

from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.dqn import agent_builder


def step_function(action, agent1, agent2):
    action1, action2 = action[0], action[1]
    new_obs1, rew1, done1, _ = agent1.env.step(action1)
    new_obs2, rew2, done2, _ = agent2.env.step(action2)

    return (new_obs1, new_obs2), (rew1, rew2), (done1, done2), _


def callback_multi_agent(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, revenues1, revenues2, states1, states2, agent1, agent2, env1, env2, callback_frequency
    if n_steps == 0:
        policy1, q_values1, _ = agent1.step_model.step(states1, deterministic=True)
        policy2, q_values2, _ = agent2.step_model.step(states2, deterministic=True)
        # save_values(env, q_values, '../Results', str(n_steps))
        policy1 = np.array([env1.A[k] for k in policy1])
        policy2 = np.array([env2.A[k] for k in policy2])
        revenues1.append(average_n_episodes(env1, policy1, 10000))
        revenues2.append(average_n_episodes(env2, policy2, 10000))
    # Print stats every 1000 calls
    if (n_steps + 1) % callback_frequency == 0:
        policy1, q_values1, _ = agent1.step_model.step(states1, deterministic=True)
        policy2, q_values2, _ = agent2.step_model.step(states2, deterministic=True)
        # save_values(env, q_values, '../Results', str(n_steps))
        policy1 = np.array([env1.A[k] for k in policy1])
        policy2 = np.array([env2.A[k] for k in policy2])
        revenues1.append(average_n_episodes(env1, policy1, 10000))
        revenues2.append(average_n_episodes(env2, policy2, 10000))
    n_steps += 1
    return True


def learn_multi_agent(agents1, agents2, total_timesteps, callback):
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

    obs1, obs2 = agent1.env.reset(), agent2.env.reset()

    for step in range(total_timesteps):
        print(step)
        if (step % (total_timesteps // 100)) == 0:
            print(revenues1, revenues2)

        callback(locals(), globals())

        update_eps1 = agent1.exploration.value(agent1.num_timesteps)
        update_eps2 = agent2.exploration.value(agent2.num_timesteps)

        with agent1.sess.as_default():
            action1 = agent1.act(np.array(obs1)[None], update_eps=update_eps1)[0]
        with agent2.sess.as_default():
            action2 = agent2.act(np.array(obs2)[None], update_eps=update_eps2)[0]

        action = (action1, action2)

        new_obs, rew, done, _ = step_function(action, agent1, agent2)

        new_obs1, new_obs2 = new_obs[0], new_obs[1]
        rew1, rew2 = rew[0], rew[1]
        done1, done2 = done[0], done[1]

        agent1.replay_buffer.add(obs1, action1, rew1, new_obs1, float(done1))
        agent2.replay_buffer.add(obs2, action2, rew2, new_obs2, float(done2))

        obs1 = new_obs1
        obs2 = new_obs2

        if done1:
            obs1 = agent1.env.reset()
        if done2:
            obs2 = agent2.env.reset()

        train(agent1)
        train(agent2)

        update_target(agent1)
        update_target(agent2)

        agent1.num_timesteps += 1
        agent2.num_timesteps += 1
    return agent1, agent2

def run_DQN_multi_agent(parameters_dict1, parameters_dict2, frequency, total_timesteps):
    global n_steps, revenues1, revenues2, states1, states2, agent1, agent2, env1, env2, callback_frequency

    env1, env2 = parameters_dict1["env_builder"](), parameters_dict2["env_builder"]()
    env_vec1, env_vec2 = DummyVecEnv([lambda: env1]), DummyVecEnv([lambda: env2])

    callback_frequency = frequency
    n_steps = 0
    revenues1, revenues2 = [], []
    states1 = [k for k in range(env1.T * env1.C)]
    states2 = [k for k in range(env2.T * env2.C)]

    agent1 = agent_builder(env_vec1, parameters_dict1)
    agent2 = agent_builder(env_vec2, parameters_dict2)

    learn_multi_agent(agent1, agent2, total_timesteps, callback_multi_agent)
