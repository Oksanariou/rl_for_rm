import numpy as np

from visualization_and_metrics import average_n_episodes, average_n_episodes_collaboration_global_policy

from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.dqn import agent_builder

def initialize(agent, total_timesteps):
    agent.num_timesteps = 0
    agent._setup_learn(seed=None)
    agent.replay_buffer = ReplayBuffer(agent.buffer_size)
    agent.exploration = LinearSchedule(schedule_timesteps=int(agent.exploration_fraction * total_timesteps),
                                        initial_p=1.0,
                                        final_p=agent.exploration_final_eps)

def train(agent):
    can_sample = agent.replay_buffer.can_sample(agent.batch_size)

    if can_sample and agent.num_timesteps > agent.learning_starts \
            and agent.num_timesteps % agent.train_freq == 0:
        obses_t, actions, rewards, obses_tp1, dones = agent.replay_buffer.sample(agent.batch_size)
        weights, batch_idxes = np.ones_like(rewards), None

        _, td_errors = agent._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                         sess=agent.sess)


def update_target(agent):
    can_sample = agent.replay_buffer.can_sample(agent.batch_size)

    if can_sample and agent.num_timesteps > agent.learning_starts and \
            agent.num_timesteps % agent.target_network_update_freq == 0:
        # Update target network periodically.
        agent.update_target(sess=agent.sess)


def callback_single_agent_collaboration_global(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, revenues, env, states, agent, callback_frequency, individual_3D_env1
    if n_steps == 0:
        policy, q_values, _ = agent.step_model.step(states, deterministic=True)
        # save_values(env, q_values, '../Results', str(n_steps))
        revenues.append(average_n_episodes_collaboration_global_policy(env, policy,individual_3D_env1, 10000))
    # Print stats every 1000 calls
    if (n_steps + 1) % callback_frequency == 0:
        policy, q_values, _ = agent.step_model.step(states, deterministic=True)
        # save_values(env, q_values, '../Results', str(n_steps))
        revenues.append(average_n_episodes_collaboration_global_policy(env, policy,individual_3D_env1, 10000))
    n_steps += 1
    return True

def callback_single_agent(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, revenues, env, states, agent, callback_frequency
    if n_steps == 0:
        policy, q_values, _ = agent.step_model.step(states, deterministic=True)
        # policy = [env.A[k] for k in policy]
        # save_values(env, q_values, '../Results', str(n_steps))
        revenues.append(average_n_episodes(env, policy, 10000))
    # Print stats every 1000 calls
    if (n_steps + 1) % callback_frequency == 0:
        policy, q_values, _ = agent.step_model.step(states, deterministic=True)
        # save_values(env, q_values, '../Results', str(n_steps))
        # policy = [env.A[k] for k in policy]
        revenues.append(average_n_episodes(env, policy, 10000))
    n_steps += 1
    return True


def learn_single_agent(agent, total_timesteps, callback, env):
    agent.num_timesteps = 0
    agent._setup_learn(seed=None)
    agent.replay_buffer = ReplayBuffer(agent.buffer_size)
    agent.exploration = LinearSchedule(schedule_timesteps=int(agent.exploration_fraction * total_timesteps),
                                       initial_p=1.0,
                                       final_p=agent.exploration_final_eps)
    obs = agent.env.reset()

    for step in range(total_timesteps):
        if (step % (total_timesteps // 10)) == 0:
            print(step)

        callback(locals(), globals())

        update_eps = agent.exploration.value(agent.num_timesteps)

        with agent.sess.as_default():
            action = agent.act(np.array(obs)[None], update_eps=update_eps)[0]

        new_obs, rew, done, info = env.step(action)

        agent.replay_buffer.add(obs, action, rew, new_obs, float(done))

        obs = new_obs

        if done:
            obs = agent.env.reset()

        train(agent)

        update_target(agent)

        agent.num_timesteps += 1
    return agent


def learn_single_agent_collaboration_global(agent, total_timesteps, callback, global_env):
    agent.num_timesteps = 0
    agent._setup_learn(seed=None)
    agent.replay_buffer = ReplayBuffer(agent.buffer_size)
    agent.exploration = LinearSchedule(schedule_timesteps=int(agent.exploration_fraction * total_timesteps),
                                       initial_p=1.0,
                                       final_p=agent.exploration_final_eps)
    obs = agent.env.reset()

    for step in range(total_timesteps):
        if (step % (total_timesteps // 10)) == 0:
            print(step)

        callback(locals(), globals())

        update_eps = agent.exploration.value(agent.num_timesteps)

        with agent.sess.as_default():
            action = agent.act(np.array(obs)[None], update_eps=update_eps)[0]

        new_obs, rews, dones, info = global_env.step(action)
        rew = rews[0] + rews[1]
        done = dones[0] and dones[1]

        agent.replay_buffer.add(obs, action, rew, new_obs, float(done))

        obs = new_obs

        if done:
            obs = agent.env.reset()

        train(agent)

        update_target(agent)

        agent.num_timesteps += 1
    return agent


def run_DQN_single_agent(parameters_dict, frequency, total_timesteps, individual_env):
    global callback_frequency, n_steps, revenues, states, agent, env, individual_3D_env1

    env = parameters_dict["env_builder"]()
    env_vec = DummyVecEnv([lambda: env])
    individual_3D_env1 = individual_env
    callback_frequency = frequency
    n_steps = 0
    revenues = []
    # states = [k for k in range(env.nS)]
    states = [[t, x1, x2] for t in range(env.T) for x1 in range(env.C1) for x2 in range(env.C2)]

    agent = agent_builder(env_vec, parameters_dict)

    # learn_single_agent_collaboration_global(agent, total_timesteps, callback_single_agent_collaboration_global, env)
    learn_single_agent(agent, total_timesteps, callback_single_agent, env)

    return revenues
