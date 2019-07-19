import gym
import numpy as np
import os
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.bench import Monitor
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN, A2C, ACER
from dynamic_programming_env_DCP import dynamic_programming_env_DCP
from visualization_and_metrics import visualisation_value_RM, visualize_policy_RM, average_n_episodes
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from pathlib import Path
import sys
import ast
from functools import partial
from multiprocessing import Pool
import glob
import csv
from q_learning import q_to_v


def agent_builder(env_vec, parameters_dict):
    return DQN(MlpPolicy, env_vec, weights=parameters_dict["weights"],
               original_weights=parameters_dict["original_weights"],
               gamma=parameters_dict["gamma"], learning_rate=parameters_dict["learning_rate"],
               buffer_size=parameters_dict["buffer_size"],
               exploration_fraction=parameters_dict["exploration_fraction"],
               exploration_final_eps=parameters_dict["exploration_final_eps"],
               train_freq=parameters_dict["train_freq"], batch_size=parameters_dict["batch_size"],
               checkpoint_freq=parameters_dict["checkpoint_freq"],
               checkpoint_path=parameters_dict["checkpoint_path"], learning_starts=parameters_dict["learning_starts"],
               target_network_update_freq=parameters_dict["target_network_update_freq"],
               prioritized_replay=parameters_dict["prioritized_replay"],
               prioritized_replay_alpha=parameters_dict["prioritized_replay_alpha"],
               prioritized_replay_beta0=parameters_dict["prioritized_replay_beta0"],
               prioritized_replay_beta_iters=parameters_dict["prioritized_replay_beta_iters"],
               prioritized_replay_eps=parameters_dict["prioritized_replay_eps"],
               param_noise=parameters_dict["param_noise"], verbose=parameters_dict["verbose"],
               policy_kwargs=parameters_dict["policy_kwargs"],
               tensorboard_log=parameters_dict["tensorboard_log"])

def save_values(env, q_values, dir_name, figure_name):
    values = q_to_v(env, q_values).reshape(env.T, env.C)
    plt.figure()
    plt.title("Values of the states")
    plt.xlabel('Number of bookings')
    plt.ylabel('Number of DCP')
    plt.imshow(values, aspect='auto')
    plt.colorbar()
    if len(figure_name) < 5:
        for k in range(5 - len(figure_name)):
            figure_name = str(0) + figure_name
    plt.savefig(dir_name+ '/' + figure_name + '.png')

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, rewards, env, states, model, callback_frequency
    if n_steps == 0:
        policy, q_values, _ = model.step_model.step(states, deterministic=True)
        # save_values(env, q_values, '../Results', str(n_steps))
        policy = np.array([env.A[k] for k in policy])
        rewards.append(average_n_episodes(env, policy, 10000))
    # Print stats every 1000 calls
    if (n_steps + 1) % callback_frequency == 0:
        policy, q_values, _ = model.step_model.step(states, deterministic=True)
        # save_values(env, q_values, '../Results', str(n_steps))
        policy = np.array([env.A[k] for k in policy])
        rewards.append(average_n_episodes(env, policy, 10000))
    n_steps += 1
    return True


def run_once(parameters_dict, nb_timesteps, general_dir_name, parameter, value, frequency, k):
    global env, rewards, n_steps, steps, model, states, count, callback_frequency
    callback_frequency = frequency

    env = parameters_dict["env_builder"]()
    env_vec = DummyVecEnv([lambda: env])

    rewards, n_steps = [], 0

    states = [k for k in range(env.T * env.C)]

    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    model = agent_builder(env_vec, parameters_dict)
    model.learn(total_timesteps=nb_timesteps, callback=callback)

    np.save(general_dir_name / parameter / str(value) / ("Run" + str(k) + ".npy"), rewards)


def run_n_times(parameters_dict, nb_timesteps, general_dir_name, parameter, number_of_runs, value, callback_frequency):
    (general_dir_name / parameter / str(value)).mkdir(parents=True, exist_ok=True)

    f = partial(run_once, parameters_dict, nb_timesteps, general_dir_name, parameter, value, callback_frequency)

    with Pool(number_of_runs) as pool:
        pool.map(f, range(number_of_runs))


def collect_list_of_mean_revenues(general_dir_name, parameter, value):
    list_of_rewards = []
    for np_name in glob.glob('../' + general_dir_name.name + '/' + parameter + '/' + str(value) + '/*.np[yz]'):
        list_of_rewards.append(list(np.load(np_name)))

    nb_collection_points = len(list_of_rewards[0])

    all_rewards_combined_at_each_collection_point = [[] for i in range(nb_collection_points)]

    for k in range(len(list_of_rewards)):
        rewards = list_of_rewards[k]
        for i in range(nb_collection_points):
            all_rewards_combined_at_each_collection_point[i].append(rewards[i])

    mean_revenues = [np.mean(list) for list in all_rewards_combined_at_each_collection_point]
    std_revenues = [sem(list) for list in all_rewards_combined_at_each_collection_point]
    confidence_revenues = [std_revenues[k] * t.ppf((1 + 0.95) / 2, nb_collection_points - 1) for k in
                           range(nb_collection_points)]
    min_revenues = [mean_revenues[k] - confidence_revenues[k] for k in range(nb_collection_points)]
    max_revenues = [mean_revenues[k] + confidence_revenues[k] for k in range(nb_collection_points)]

    return mean_revenues, min_revenues, max_revenues


def plot_revenues(parameters_dict, nb_timesteps, mean_revenues, min_revenues, max_revenues, callback_frequency):
    env = parameters_dict["env_builder"]()
    V, P_ref = dynamic_programming_env_DCP(env)
    P_DP = P_ref.reshape(env.T * env.C)

    steps = [0]
    for k in range(callback_frequency - 1, nb_timesteps, callback_frequency):
        steps.append(k)

    fig = plt.figure()
    plt.plot(steps, mean_revenues, color="gray", label='DQL mean revenue')
    plt.fill_between(steps, min_revenues, max_revenues, label='95% confidence interval', color="gray", alpha=0.2)
    plt.plot(steps, [average_n_episodes(env, P_DP, 10000)] * len(steps), label="DP Revenue")
    plt.legend()
    plt.ylabel("Revenue computed over 10000 episodes")
    plt.xlabel("Number of timesteps")
    return fig


def tune_parameter(general_dir_name, parameter, parameter_values, parameters_dict, nb_timesteps, number_of_runs, callback_frequency):
    # results_dir_name = "../Daily meetings/Stabilization experiences/" + parameter
    # os.mkdir(general_dir_name + "/" + parameter)
    (general_dir_name / parameter).mkdir(parents=True, exist_ok=True)
    parameter_path = Path(parameter)

    for value in parameter_values:
        print(parameter + " = "+str(value))
        parameters_dict[parameter] = value

        if parameter == "policy_kwargs":
            value = value['dueling']

        experience_dir_name = parameter + " = " + str(value)

        run_n_times(parameters_dict, nb_timesteps, general_dir_name, parameter, number_of_runs, value, callback_frequency)
        mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(general_dir_name, parameter,
                                                                                  value)
        fig = plot_revenues(parameters_dict, nb_timesteps, mean_revenues, min_revenues, max_revenues, callback_frequency)
        plt.savefig('../' + general_dir_name.name + '/' + parameter + '/' + experience_dir_name + '.png')

        mean_revenue, speed = compute_metric(mean_revenues)
        metrics_file_name = '../' + general_dir_name.name + '/metrics_file.csv'
        save_metrics(metrics_file_name, parameter, value, mean_revenue, speed)


def compute_metric(list_of_revenues):
    starting_point_slope = 0
    ending_point_slope = 5

    starting_point_mean = 20

    mean_revenue = np.mean([list_of_revenues[-starting_point_mean:]])
    speed = (list_of_revenues[ending_point_slope] - list_of_revenues[starting_point_slope]) / (
                ending_point_slope - starting_point_slope)

    return mean_revenue, speed

def save_metrics(metrics_file_name, parameter, value, mean_revenues, speed):
    file_exists = os.path.isfile(metrics_file_name)
    fieldnames = ['parameter', 'value', 'average', 'slope']

    if file_exists:
        with open(metrics_file_name, 'a') as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            writer.writerow({'parameter': parameter, 'value': value, 'average': mean_revenues, 'slope': speed})
    else:
        with open(metrics_file_name, 'a') as metrics_file:
            writer = csv.DictWriter(metrics_file, delimiter=',', lineterminator='\n', fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'parameter': parameter, 'value': value, 'average': mean_revenues, 'slope': speed})

def compare_plots(general_dir_name, parameter, values, nb_timesteps, callback_frequency):
    plt.figure()

    for value in values:
        if parameter == "policy_kwargs":
            value = value["dueling"]
        steps = [0]
        for k in range(callback_frequency - 1, nb_timesteps, callback_frequency):
            steps.append(k)
        mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(general_dir_name, parameter, value)
        plt.plot(steps, mean_revenues, label=str(value))
        plt.fill_between(steps, min_revenues, max_revenues, alpha=0.2)

    plt.legend()
    plt.ylabel("Revenue computed over 10000 episodes")
    plt.xlabel("Number of timesteps")
    plt.title(parameter)

    plt.savefig('../' + general_dir_name.name + '/' + parameter + '.png')

def env_builder():
    # Parameters of the environment
    data_collection_points = 100
    micro_times = 5
    capacity = 50
    actions = tuple(k for k in range(50, 231, 10))
    alpha = 0.8
    lamb = 0.7

    return gym.make('gym_RMDCPDiscrete:RMDCPDiscrete-v0', data_collection_points=data_collection_points,
                    capacity=capacity,
                    micro_times=micro_times, actions=actions, alpha=alpha, lamb=lamb)


def compute_weights(env):
    compute_weight = lambda x: 1 + max(1. * x[0] / env.T, 1. * x[1] / env.C)
    weights = [(s, compute_weight((env.to_coordinate(s)[0], env.to_coordinate(s)[1]))) for s in range(env.T*env.C)]
    return weights


def parameters_dict_builder():
    parameters_dict = {}
    parameters_dict["env_builder"] = env_builder
    parameters_dict["gamma"] = 0.99
    parameters_dict["learning_rate"] = 0.0001
    parameters_dict["buffer_size"] = 10000
    parameters_dict["exploration_fraction"] = 0.4
    parameters_dict["exploration_final_eps"] = 0.01
    parameters_dict["train_freq"] = 1
    parameters_dict["batch_size"] = 100
    parameters_dict["checkpoint_freq"] = 10000
    parameters_dict["checkpoint_path"] = None
    parameters_dict["learning_starts"] = 100
    parameters_dict["target_network_update_freq"] = 50
    parameters_dict["prioritized_replay"] = False
    parameters_dict["prioritized_replay_alpha"] = 0.6
    parameters_dict["prioritized_replay_beta0"] = 0.4
    parameters_dict["prioritized_replay_beta_iters"] = None
    parameters_dict["prioritized_replay_eps"] = 1e-6
    parameters_dict["param_noise"] = False
    parameters_dict["verbose"] = 0
    # parameters_dict["tensorboard_log"] = "./../log_tensorboard/"
    parameters_dict["tensorboard_log"] = None
    parameters_dict["policy_kwargs"] = {"dueling": False}
    parameters_dict["weights"] = False

    env = parameters_dict["env_builder"]()
    parameters_dict["original_weights"] = compute_weights(env)

    return parameters_dict


if __name__ == '__main__':

    results_path = Path("../Results_18_07_19")
    results_path.mkdir(parents=True, exist_ok=True)


    # Tuning of the parameters
    # parameter = sys.argv[1]
    # parameter_values_string = sys.argv[2]
    # print(parameter_values_string)
    # parameter_values = ast.literal_eval(parameter_values_string)

    nb_timesteps = 40000
    nb_runs = 2
    callback_frequency = 500

    # parameter = "exploration_final_eps"
    # parameter_values = [0.05, 0.1, 0.2, 0.5]
    # parameters_dict = parameters_dict_builder()
    #
    # # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_timesteps, nb_runs)
    # # compare_plots(results_path, parameter, parameter_values, nb_timesteps, callback_frequency)
    #
    # parameter = "weights"
    # parameter_values = [True, False]
    # parameters_dict = parameters_dict_builder()
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_timesteps, nb_runs, callback_frequency)
    #
    # parameter = "prioritized_replay"
    # parameter_values = [True]
    # parameters_dict = parameters_dict_builder()
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_timesteps, nb_runs, callback_frequency)
    #
    # parameter = "param_noise"
    # parameter_values = [True]
    # parameters_dict = parameters_dict_builder()
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_timesteps, nb_runs, callback_frequency)
    #
    # parameter = "policy_kwargs"
    # parameter_values = [{"dueling": True}]
    # parameters_dict = parameters_dict_builder()
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_timesteps, nb_runs, callback_frequency)

    parameter = "buffer_size"
    parameter_values = [1000, 10000, 20000, 30000]
    parameters_dict = parameters_dict_builder()
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_timesteps, nb_runs, callback_frequency)
    # compare_plots(results_path, parameter, parameter_values, nb_timesteps, callback_frequency)
    general_dir_name = results_path
    value = 1000
    experience_dir_name = parameter + " = " + str(value)
    mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(general_dir_name, parameter,
                                                                              value)
    fig = plot_revenues(parameters_dict, nb_timesteps, mean_revenues, min_revenues, max_revenues, callback_frequency)
    plt.savefig('../' + general_dir_name.name + '/' + parameter + '/' + experience_dir_name + '.png')

    mean_revenue, speed = compute_metric(mean_revenues)
    metrics_file_name = '../' + general_dir_name.name + '/metrics_file.csv'
    save_metrics(metrics_file_name, parameter, value, mean_revenue, speed)

    # parameter = "batch_size"
    # parameter_values = [10, 100, 10000]
    # parameters_dict = parameters_dict_builder()
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_timesteps, nb_runs, callback_frequency)
    # compare_plots(results_path, parameter, parameter_values, nb_timesteps, callback_frequency)
    #
    # parameter = "learning_rate"
    # parameter_values = [1e-4, 1e-3, 1e-2]
    # parameters_dict = parameters_dict_builder()
    # tune_parameter(results_path, parameter, parameter_values, parameters_dict, nb_timesteps, nb_runs, callback_frequency)
    # compare_plots(results_path, parameter, parameter_values, nb_timesteps, callback_frequency)

    # import cv2
    # import os
    #
    # image_folder = '../Results'
    # video_name = 'evolution_vof_values.avi'
    #
    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # frame = cv2.imread(os.path.join(image_folder, images[0]))
    # height, width, layers = frame.shape
    #
    # video = cv2.VideoWriter(video_name, 0, 5, (width, height))
    #
    # for image in images:
    #     video.write(cv2.imread(os.path.join(image_folder, image)))
    #
    # cv2.destroyAllWindows()
    # video.release()

    # plt.figure()
    #
    # parameters = ["weights", "prioritized_replay", "dueling"]
    # nb_timesteps = 30000
    #
    # for parameter in parameters:
    #     values = [True]
    #     if parameter == "dueling":
    #         values = [{"dueling": True}]
    #     if parameter == "weights":
    #         values = [True, False]
    #     for value in values:
    #         steps = [0]
    #         for k in range(1000 - 1, nb_timesteps, 1000):
    #             steps.append(k)
    #         mean_revenues, min_revenues, max_revenues = collect_list_of_mean_revenues(results_path, parameter, value)
    #         plt.plot(steps, mean_revenues, label=str(parameter)+str(value))
    #         plt.fill_between(steps, min_revenues, max_revenues, alpha=0.2)
    #
    # plt.legend()
    # plt.ylabel("Revenue computed over 10000 episodes")
    # plt.xlabel("Number of timesteps")
    # plt.title(parameter)
    #
    # plt.savefig('../' + results_path.name + '/extensions.png')





