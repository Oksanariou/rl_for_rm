import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)


def observation_split(observation):
    time = observation[0]
    splited_observations = []
    for x in range(1, len(observation)):
        splited_observations.append((time, observation[x]))
    return splited_observations


def action_merge(actions, global_env):
    nb_flight = len(actions)
    if nb_flight == 1:
        action = actions[0]
        individual_actions = global_env.prices[0][action]
    else:
        individual_actions = tuple(global_env.prices[k][actions[k]] for k in range(nb_flight))
    return global_env.A.index(individual_actions)


def fit_multi_agent(agents, global_env, nb_steps, verbose=1, log_interval=10000, callbacks=None, nb_max_episode_steps=None):
    agents_nb = len(agents)

    callbacks = [] if not callbacks else callbacks[:]

    for agent in agents:
        if not agent.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')

        agent.training = True
        agent._on_train_begin()
        agent.step = np.int16(0)

    if verbose == 1:
        callbacks += [TrainIntervalLogger(interval=log_interval)]
    elif verbose > 1:
        callbacks += [TrainEpisodeLogger()]

    history = History()
    callbacks += [history]
    callbacks = CallbackList(callbacks)
    # if hasattr(callbacks, 'set_model'):
    #     callbacks.set_model(agents)
    # else:
    #     callbacks._set_model(agents)
    # callbacks._set_env(global_env)
    # params = {
    #     'nb_steps': nb_steps,
    # }
    # if hasattr(callbacks, 'set_params'):
    #     callbacks.set_params(params)
    # else:
    #     callbacks._set_params(params)

    callbacks.on_train_begin()

    episode = np.int16(0)
    observation = None
    episode_reward = None
    episode_step = None
    did_abort = False
    try:
        while agents[0].step < nb_steps:
            if observation is None:  # start of a new episode
                callbacks.on_episode_begin(episode)
                episode_step = np.int16(0)
                episode_reward = [np.float32(0) for k in range(agents_nb)]
                for agent in agents:
                    agent.reset_states()
                observation = deepcopy(global_env.reset())
                assert observation is not None

            assert episode_reward is not None
            assert episode_step is not None
            assert observation is not None

            callbacks.on_step_begin(episode_step)
            actions = [agent.forward(obs) for agent, obs in zip(agents, observation_split(observation))]

            done = False

            callbacks.on_action_begin(actions)
            observation, r, done, info = global_env.step(action_merge(actions, global_env))
            observation = deepcopy(observation)

            callbacks.on_action_end(actions)
            if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                done = True

            metrics = {}
            for k in range(agents_nb):
                reward = info.get('individual_reward' + str(k + 1), r / agents_nb)
                metrics['metrics' + str(k + 1)] = agents[k].backward(reward, terminal=done)
                episode_reward[k] += reward

            step_logs = {
                'action': actions,
                'observation': observation,
                'episode': episode,
                'reward': r
            }

            step_logs.update(metrics)

            callbacks.on_step_end(episode_step, step_logs)
            episode_step += 1
            for agent in agents:
                agent.step += 1

            if done:
                episode_logs = {
                    'nb_episode_steps': episode_step,
                    'nb_steps': agents[0].step,
                    'episode_reward': np.sum(episode_reward)
                }
                individual_observations = observation_split(observation)
                for k in range(agents_nb):
                    agents[k].forward(individual_observations[k])
                    agents[k].backward(0., terminal=False)
                    episode_logs['episode_reward' + str(k + 1)] = episode_reward[k]

                callbacks.on_episode_end(episode, episode_logs)

                episode += 1
                observation = None
                episode_step = None
                episode_reward = None
    except KeyboardInterrupt:
        did_abort = True
    callbacks.on_train_end(logs={'did_abort': did_abort})
    for agent in agents:
        agent._on_train_end()

    return history
