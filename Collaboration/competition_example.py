import gym
import numpy as np

from stable_baselines import DQN

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from visualization_and_metrics import average_n_episodes
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer


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


if __name__ == '__main__':
    collaboration_env = collaboration_env_builder()

    parameters_dict1 = parameters_dict_builder()
    parameters_dict2 = parameters_dict_builder()
    # parameters_dict2["env_builder"] = small_env_builder

    env1, env2 = parameters_dict1["env_builder"](), parameters_dict2["env_builder"]()
    env1.T = collaboration_env.T
    env1.C = collaboration_env.C1
    env1.A = tuple(k for k in range(50, 231, 20))
    env_vec1, env_vec2 = DummyVecEnv([lambda: env1]), DummyVecEnv([lambda: env2])

    callback_frequency = 1000
    n_steps = 0
    revenues1, revenues2 = [], []
    states1 = [k for k in range(env1.T * env1.C)]
    states2 = [k for k in range(env2.T * env2.C)]

    total_timesteps = 40000

    agent1 = agent_builder(env_vec1, parameters_dict1)
    agent2 = agent_builder(env_vec2, parameters_dict2)

    # learn_single_agent(agent, total_timesteps, callback_single_agent)
    learn_multi_agent(agent1, agent2, total_timesteps, callback_multi_agent, collaboration_env)







