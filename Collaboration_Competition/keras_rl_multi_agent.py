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


def fit(agents, env, nb_steps, callbacks=None, verbose=1,
        log_interval=10000,
        nb_max_episode_steps=None):

    callbacks = [] if not callbacks else callbacks[:]

    for agent in agents:
        if not agent.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')

        agent.training = True

    # if verbose == 1:
    #     callbacks += [TrainIntervalLogger(interval=log_interval)]
    # elif verbose > 1:
    #     callbacks += [TrainEpisodeLogger()]
    #
    # history = History()
    # callbacks += [history]
    # callbacks = CallbackList(callbacks)
    # if hasattr(callbacks, 'set_model'):
    #     callbacks.set_model(self)
    # else:
    #     callbacks._set_model(self)
    # callbacks._set_env(env)
    # params = {
    #     'nb_steps': nb_steps,
    # }
    # if hasattr(callbacks, 'set_params'):
    #     callbacks.set_params(params)
    # else:
    #     callbacks._set_params(params)
        agent._on_train_begin()
    # agent1._on_train_begin()
    # agent2._on_train_begin()
    callbacks.on_train_begin()

    episode = np.int16(0)
    self.step = np.int16(0)
    # agent1.step, agent2.step = np.int16(0), np.int16(0)
    observation = None
    episode_reward = None
    episode_step = None
    did_abort = False
    try:
        while self.step < nb_steps:
            # while agent1.step < nb_steps
            if observation is None:  # start of a new episode
                callbacks.on_episode_begin(episode)
                episode_step = np.int16(0)
                episode_reward = np.float32(0)

                # Obtain the initial observation by resetting the environment.
                self.reset_states()
                # agent1.reset_states()
                # agent2.reset_states()
                observation = deepcopy(env.reset())
                # observation1, observation2 = env1.to_idx(observation[0], observation[1]), env2.to_idx(observation[0], observation[2])
                assert observation is not None

            # At this point, we expect to be fully initialized.
            assert episode_reward is not None
            assert episode_step is not None
            assert observation is not None

            # Run a single step.
            callbacks.on_step_begin(episode_step)
            # This is were all of the work happens. We first perceive and compute the action
            # (forward step) and then use the reward to improve (backward step).
            action = self.forward(observation)
            # action1, action2 = agent1.forward(observation1), agent2.forward(observation2)
            reward = np.float32(0)
            accumulated_info = {}
            done = False

            # action = env.A.index((env1.A[action1], env2.A[action2]))
            callbacks.on_action_begin(action)
            observation, r, done, info = env.step(action)
            observation = deepcopy(observation)
            # observation1, observation2 = env1.to_idx(observation[0], observation[1]), env2.to_idx(observation[0], observation[2])
            # r1, r2 = info["individual_reward1"], info["individual_reward2"]
            for key, value in info.items():
                if not np.isreal(value):
                    continue
                if key not in accumulated_info:
                    accumulated_info[key] = np.zeros_like(value)
                accumulated_info[key] += value
            callbacks.on_action_end(action)
            reward += r
            # reward1 += r1
            # reward2 += r2
            if done:
                break
            if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                # Force a terminal state.
                done = True
            metrics = self.backward(reward, terminal=done)
            # metrics1, metrics2 = agent1.backward(reward1, terminal=done1), agent2.backward(reward2, terminal=done2)
            episode_reward += reward

            step_logs = {
                'action': action,
                'observation': observation,
                'reward': reward,
                'metrics': metrics,
                'episode': episode,
                'info': accumulated_info,
            }
            callbacks.on_step_end(episode_step, step_logs)
            episode_step += 1
            self.step += 1
            # agent1.step += 1
            # agent2.step += 1

            if done:
                # We are in a terminal state but the agent hasn't yet seen it. We therefore
                # perform one more forward-backward call and simply ignore the action before
                # resetting the environment. We need to pass in `terminal=False` here since
                # the *next* state, that is the state of the newly reset environment, is
                # always non-terminal by convention.
                self.forward(observation)
                # action1, action2 = agent1.forward(observation1), agent2.forward(observation2)
                self.backward(0., terminal=False)
                # agent1.backward(0., terminal=False), agent2.backward(0., terminal=False)

                # This episode is finished, report and reset.
                episode_logs = {
                    'episode_reward': episode_reward,
                    'nb_episode_steps': episode_step,
                    'nb_steps': self.step,
                }
                callbacks.on_episode_end(episode, episode_logs)

                episode += 1
                observation = None
                episode_step = None
                episode_reward = None
    except KeyboardInterrupt:
        # We catch keyboard interrupts here so that training can be be safely aborted.
        # This is so common that we've built this right into this function, which ensures that
        # the `on_train_end` method is properly called.
        did_abort = True
    callbacks.on_train_end(logs={'did_abort': did_abort})
    self._on_train_end()

    return history
