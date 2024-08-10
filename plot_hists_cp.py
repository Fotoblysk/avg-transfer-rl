import csv
import ast
import itertools
import os
import sys

import matplotlib.pyplot as plt
import re

import numpy as np
from cycler import cycler

from good_rainbow_src.dqn_alg import softmax


def preprocess_list_string(list_string):
    # Remove newlines and extra spaces
    list_string = list_string.replace('\n', ' ').strip()
    # Replace multiple spaces with a single space
    list_string = re.sub(r'\s+', ' ', list_string)
    # Replace spaces with commas
    list_string = list_string.replace(' ', ',')
    # Ensure the string is in a valid list format
    if not (list_string.startswith('[') and list_string.endswith(']')):
        list_string = f'[{list_string}]'
    # Remove any double commas introduced by the replacement
    list_string = re.sub(r',+', ',', list_string)
    # Remove commas after opening brackets and before closing brackets
    list_string = re.sub(r'\[,', '[', list_string)
    list_string = re.sub(r',\]', ']', list_string)
    return list_string


def read_csv(file_path):
    data = {
        'frame_idx': [],
        'score': [],
        'disc_score': [],
        'ep_len': [],
        'action_hist': [],
        'models_usage_hist': [],
        'models_avg_reward': [],
        'model_choose_probs': []
    }

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data['frame_idx'].append(int(row['frame_idx']))
            data['score'].append(float(row['score']))
            data['disc_score'].append(float(row['disc_score']))
            data['ep_len'].append(int(row['ep_len']))
            data['action_hist'].append(ast.literal_eval(preprocess_list_string(row['action_hist'])))
            data['models_usage_hist'].append(ast.literal_eval(preprocess_list_string(row['models_usage_hist'])))
            data['models_avg_reward'].append(ast.literal_eval(preprocess_list_string(row['models_avg_reward'])))
            data['model_choose_probs'].append(ast.literal_eval(preprocess_list_string(row['model_choose_probs'])))

    return data


def plot_data(data):
    NUM_COLORS = len(data["models_usage_hist"][0])

    #cm = plt.cm.nipy_spectral
    import seaborn as sns
    #cm = sns.color_palette('hls', NUM_COLORS)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']

    # Create a list of (color, linestyle) pairs
    color_linestyle_pairs = [(color, linestyle) for linestyle in linestyles for color in colors]

    # Separate the pairs into two lists: one for colors and one for linestyles
    extended_colors, extended_linestyles = zip(*color_linestyle_pairs)

    custom_cycler = cycler(color=extended_colors) + cycler(linestyle=extended_linestyles)


    print(len(colors))

    plt.rc('axes', prop_cycle=custom_cycler)# [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]))

    # Plot action_hist
    print(len(data['frame_idx']))
    plt.figure(figsize=(10, 6))
    plt.plot(data['frame_idx'], data["action_hist"])
    plt.title('Action History')
    plt.xlabel('Frame Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Plot models_usage_hist
    plt.figure(figsize=(10, 6))
    plt.plot(data['frame_idx'], data["models_usage_hist"])
    plt.title('Models Usage History')
    plt.xlabel('Frame Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Plot models_avg_reward

    plt.figure(figsize=(10, 6))
    if 'minigrid' in file_path:
        labels = ['RANDOM', 'STUDENT', *[i for i in list(sorted(os.listdir('sorted_models/minigrid/'))) if file_path.split('/')[1].split('_')[0] not in i]]
    if 'lunar_lander' in file_path:
        print(file_path)
        print("_".join(file_path.split('/')[1].split('_')[0:5]))
        labels = ['RANDOM', 'STUDENT',
                  *[i for i in list(sorted(os.listdir('sorted_models/lunar_lander/'))) if "_".join(file_path.split('/')[1].split('_')[0:5]) not in i]]
    if 'FrozenLake' in file_path:
        labels = ['RANDOM', 'STUDENT',
                  *[i for i in list(sorted(os.listdir('sorted_models/frozen_lake/'))) if "_".join(file_path.split('/')[1].split('_')[0:2]) not in i]]

    plt.plot(data['frame_idx'], data["models_avg_reward"], label=labels)
    plt.title('Models Average Reward')
    plt.xlabel('Frame Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data['frame_idx'], data["model_choose_probs"], label=labels[1:])
    plt.title('Models Probs')
    plt.xlabel('Frame Index')
    plt.ylabel('Probs')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_01-46/stats/train_ep_data.csv'
    #csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_03-16/stats/train_ep_data.csv'
    #csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_10-08/stats/train_ep_data.csv'
    #csv_file_path = 'results/MiniGrid-Fetch-8x8-N3-v0_07-08-2024_10-31/stats/train_ep_data.csv'
    #csv_file_path = 'results/MiniGrid-Empty-8x8-v0_07-08-2024_16-52/stats/train_ep_data.csv'
    #csv_file_path = 'results/MiniGrid-Empty-Random-6x6-v0_07-08-2024_17-39/stats/train_ep_data.csv'
    #csv_file_path = 'results/MiniGrid-Fetch-5x5-N2-v0_08-08-2024_11-57/stats/train_ep_data.csv'
    #csv_file_path = 'results/lunar_lander_8_20_2_08-08-2024_14-42/stats/train_ep_data.csv'
    #csv_file_path = 'results/FrozenLake-map_3_08-08-2024_15-41/stats/train_ep_data.csv'
    #csv_file_path = 'results/FrozenLake-map_3_08-08-2024_15-41/stats/train_ep_data.csv'
    #csv_file_path = 'results/FrozenLake-map_2_08-08-2024_16-15/stats/train_ep_data.csv'
    #csv_file_path = 'results/FrozenLake-map_1_08-08-2024_16-23/stats/train_ep_data.csv'
    csv_file_paths = sys.argv[1:]
    #file_path = csv_file_path
    for file_path in csv_file_paths:
        data = read_csv(file_path)
        plot_data(data)