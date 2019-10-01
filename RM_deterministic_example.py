# myapp.py

import matplotlib.pyplot as plt
import numpy as np
import gym
from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from visualization_and_metrics import average_n_episodes, visualize_policy_RM

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import Slider, CustomJS
from bokeh.layouts import row, column

from random import random

from bokeh.layouts import column
from bokeh.models import Button
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.server.server import Server

def env_builder():
    # Parameters of the environment
    data_collection_points = 20
    micro_times = 1
    capacity = 20

    action_min = 100
    action_max = 401
    action_offset = 100

    actions = tuple(k for k in range(action_min, action_max, action_offset))
    alpha = 0.9
    lamb = 1

    return gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

def plot_bookings(env, bookings):
    plt.figure()
    width = 0.2
    capacity = env.C
    labels = [str(env.A[k]) for k in range(env.nA)]
    plt.bar([k for k in range(env.nA)], bookings, width)
    plt.xlabel("Prices")
    plt.ylabel("Average number of bookings")
    plt.title("Overall load factor: {:.2}".format(np.sum(bookings) / capacity))
    plt.xticks([k for k in range(env.nA)], labels)
    plt.show()

if __name__ == '__main__':

    data_collection_points = 20
    micro_times = 1
    capacity = 10

    action_min = 100
    action_max = 401
    action_offset = 100

    actions = tuple(k for k in range(action_min, action_max, action_offset))
    alpha = 0.6
    lamb = 1

    env = gym.make('gym_RMDCP:RMDCP-v0', data_collection_points=data_collection_points, capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)

    true_V, true_P = dynamic_programming_env_DCP(env)
    true_revenues, true_bookings = env.average_n_episodes(true_P, 10000)

    plot_bookings(env, true_bookings)
    visualize_policy_RM(true_P, data_collection_points, capacity)

    # Bid price vector
    for time in range(1):
        Y = []
        X = [k for k in range(19)]
        for k in range(len(true_V[time]) - 1):
            Y.append(true_V[time][19 - k - 1] - true_V[time][19 - k])
        plt.plot(X, Y)
        plt.xticks(X)
        plt.show()

    # Convex hull
    X = [3.4, 6, 11, 20]
    plt.figure()
    Y = [1360, 1800, 2200, 2000]
    plt.plot(X, Y, 'X')
    plt.xlim(0, 23)
    plt.xlabel("Demand")
    plt.ylabel("Revenue")
    plt.title("The convex hull")
    plt.ylim(0, 2400)
    plt.text(X[0] + 0.2, Y[0] + 0.2, "{Y}")
    plt.text(X[1] + 0.2, Y[1] + 0.2, "{B}")
    plt.text(X[2] + 0.2, Y[2] + 0.2, "{M}")
    plt.text(X[3] + 0.2, Y[3] + 0.2, "{Q}")
    plt.show()
