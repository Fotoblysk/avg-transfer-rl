import math
import os
import pickle
import random
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import FrameStack, ResizeObservation, AtariPreprocessing
# from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_

from good_rainbow_src.dqn_alg import DQNAgent
from good_rainbow_src.layers import NoisyLinear
from good_rainbow_src.memory_replay import PrioritizedReplayBuffer, ReplayBuffer
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation

from good_rainbow_src.wrappers import SkipWrapper

plt.ion()

paper_preprocessing = {
    "frame_stack": 4,
    "frameskip": 4,
    # "grayscale": True,
    # "resize_obs": (84, 84),
    "atari": True
}

cp_data = {
    "env_kwargs": {
        "id": "CartPole-v1",
    },
    "meta": {
        "v_min": -10,  # -21,  # min value including discount factor 10 is used in paper
        "v_max": 10,  # 21,  # max value including discount factor -10 is used in paper
        "atom_size": 51,  # number of grains, grain size is v_max-vmin/atom_size # dont want to count if equal range
        # "reward_clip": (0, 1),  # reward clip should be enough buf for better score distribution we might want to
        "reward_scale": (1 / 100, -100)
        # rescale reward
    },
}

lunar_lander_data = {
    "env_kwargs": {
        "id": "LunarLander-v2",
        "continuous": False,
        "gravity": -10.0,
        "enable_wind": False,
        "wind_power": 15.0,
        "turbulence_power": 1.5
    },
    "meta": {
        "v_min": -10,
        # -21,  # min value including discount factor 10 is used in paper # rest of alg is adapted to -10:10
        "v_max": 10,  # 21,  # max value including discount factor -10 is used in paper
        "atom_size": 51,  # number of grains, grain size is v_max-vmin/atom_size # dont want to count if equal range
        # "reward_clip": (-1, 1),  # reward clip should be enough buf for better score distribution we might want to
        "reward_scale": (1 * 10 / ((140 + 100 + 2 * 10)), 0)  # 1/140+100+2*10 # *10
        # rescale reward
    },
}

pong_data = {  # TODO orginal paper clips reward (0,1) we might want to normalize :)
    "env_kwargs": {
        "id": "ALE/Pong-v5",
    },
    "meta": {
        "v_min": -10,  # -21,  # min value including discount factor 10 is used in paper
        "v_max": 10,  # 21,  # max value including discount factor -10 is used in paper
        "atom_size": 51,  # number of grains, grain size is v_max-vmin/atom_size # dont want to count if equal range
    },
    "preprocess":
        {
            "frame_stack": 4,
            "grayscale": True,
            "resize_obs": (84, 84)
        }
}

demon_attack_data = {
    "env_kwargs": {
        "id": "ALE/DemonAttack-v5",
    },
    "meta": {
        "v_min": -10,  # -21,  # min value including discount factor 10 is used in paper
        "v_max": 10,  # 21,  # max value including discount factor -10 is used in paper
        "atom_size": 51,  # number of grains, grain size is v_max-vmin/atom_size # dont want to count if equal range
        "reward_clip": (-1, 1)  # reward clip should be enough buf for better score distribution we might want to
        # rescale reward
    },
    "preprocess":
        {
            "frame_stack": 4,
            "grayscale": True,
            "resize_obs": (84, 84)
        }
}

demon_attack_v4_data = {  # TODO orginal paper clips reward (0,1) we might want to normalize :)
    "env_kwargs": {
        "id": "ALE/DemonAttack-v5",
    },
    "meta": {
        "v_min": 0,  # -21,  # min value including discount factor 10 is used in paper # in reality it's more like 0-10
        "v_max": 10,  # 21,  # max value including discount factor -10 is used in paper
        "atom_size": 51,  # number of grains, grain size is v_max-vmin/atom_size # dont want to count if equal range
        "reward_clip": (-1, 1),  # reward clip should be enough buf for better score distribution we might want to the same tho clipping shouldn't matter
        # rescale reward
    },
    "preprocess": paper_preprocessing
}

pitfall_data = {
    "env_kwargs": {
        "id": "ALE/Pitfall-v5",
    },
    "meta": {
        "v_min": -0.1,  # min value including discount factor
        "v_max": 144,  # max value including discount factor
        "atom_size": 51,
        # number of grains, grain size is v_max-vmin/atom_size # dont want to count if equal range #this might be small for life loss
        "reward_scale": (1 / 1000, 0),
        # "reward_clip": (-1,1)

    },
    "preprocess":
        {
            "frame_stack": 4,
            "grayscale": True,
            "resize_obs": (84, 84)
        }
}

rc_data = {
    "env_kwargs": {
        "id": "CarRacing-v2",
        "continuous": False
    },
    "meta": {
        "v_min": -10,  # min value including discount factor
        "v_max": 10,  # max value including discount factor
        "atom_size": 51,  # number of grains, grain size is v_max-vmin/atom_size
        "correct_dims": True,
        "reward_scale": (1 / 1000, 0),
    }
}

env_data = demon_attack_v4_data

atari_extra_kwargs = {}
if env_data["env_kwargs"]["id"] in {"CarRacing-v2"} or env_data["env_kwargs"]["id"][0:3] == "ALE":
    conv_space = True
    if env_data["env_kwargs"]["id"] in {"CarRacing-v2"} or env_data["env_kwargs"]["id"][0:3] == "ALE":
        atari_extra_kwargs = {"frameskip":1, "repeat_action_probability":0}
else:
    conv_space = False

env = gym.make(**env_data["env_kwargs"], render_mode="rgb_array",
               **atari_extra_kwargs)
if "preprocess" in env_data:
    if env_data["preprocess"].get("atari") is True:  # orginal settings
        print("atari")
        env = AtariPreprocessing(env, 30, 1, 84, True, True)
        # I have no idea if it should be 30 or 0,
        # not mentioned for training in research paper # it migh be not - enabled for testing
    else:
        if env_data["preprocess"].get("grayscale") == True:
            print("gray")
            if env_data["preprocess"].get("frame_stack", 0) != 0:
                env = GrayScaleObservation(env, keep_dim=False)
            else:
                env = GrayScaleObservation(env, keep_dim=True)
        if env_data["preprocess"].get("resize_obs", None) != None:
            print("res")
            env = ResizeObservation(env, env_data["preprocess"].get("resize_obs", None))

    if env_data["preprocess"].get("frame_stack", 1) != 1:
        print("stack")
        lz4_compress = True
        env = FrameStack(env, num_stack=env_data["preprocess"].get("frame_stack", 1), lz4_compress=lz4_compress)

    if env_data["preprocess"].get("frameskip", 1) != 1:
        print("skip")
        env = SkipWrapper(env, frame_skip=env_data["preprocess"].get("frameskip", 1))

print(f'Grain size: {(env_data["meta"]["v_max"] - env_data["meta"]["v_min"]) / env_data["meta"]["atom_size"]}')

seed = 777


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

# parameters
num_frames = 20000000  # 10000 * 100 * 20  # 5M
memory_size = int(
    1000000)  # # in orginal is 1M sadly not anough ram 0.4M should be possible tho might get to swap so 0.3 for safety # goes to swap :( 0.09 for now
batch_size = 32  # * 8  # * 12 # on internet  128 *12 test, 32 in paper
learning_interval = 4  # * 8  # * 12# because of bigger batch size? -2 just for safety
target_update = 32000 // learning_interval  # we divide of learning interval and how we count # 32k was in orginal rainbow, 10k in orginal dqn
min_memory_size = min(int(memory_size * 0.9), 80000)  # memory_size * 9//10 # big start memory for debugging

network_guide = False

guided_network = None
if network_guide:
    save = torch.load('./model_save')
    guided_network = {"model": save['dqn_target'], 'reward': save['end_avg_reward']}

# train
agent = DQNAgent(env, memory_size, batch_size, seed, target_update, **env_data["meta"], conv_space=conv_space,
                 learning_interval=learning_interval, min_memory_size=min_memory_size, save_fig='plot.png',
                 guided_network=guided_network)

# load if present
load_save = False
continue_or_dummy_transfer_learning_if_false = False

if load_save:
    try:
        save = torch.load('model_save')
        save_loaded = True
    except FileNotFoundError:
        save_loaded = False
        print('No save breaking loading')

    if save_loaded:
        print('loading prev model')
        agent.dqn.load_state_dict(save['dqn']),
        agent.dqn_target.load_state_dict(save['dqn_target']),
        agent.optimizer.load_state_dict(save['optimizer']),
        # agent.memory = pickle.loads(save['memory']),
        # agent.memory_n = pickle.loads(save['memory_n']),
        # TODO define pickle setstate getstate to support this also remember that orginal are in LazyFrames so we also want to store duplicate states like this
        # also we probably want to use pickle.dump & pickle.load as swap will go brr anyway during this
        save = None
        if not continue_or_dummy_transfer_learning_if_false:
            print('reseting layers except 2 first conv')


            def reset_except_first_two_conv(network):  # TODO this is bad place to implement this
                """Reset all weights except the weights of the first two conv layers."""
                # List to hold the layers that we do not want to reset
                do_not_reset = {id(layer) for layer in network.transferable_conv_layers}

                # Reset parameters for layers except the ones in do_not_reset list
                for name, module in network.named_modules():
                    if id(module) in do_not_reset:
                        # print('not_reset', module)
                        continue  # Skip the first two conv layers
                    # If the module has 'reset_parameters' method, call it to reset the module's weights
                    if hasattr(module, 'reset_parameters'):
                        # print('reset', module)
                        module.reset_parameters()


            reset_except_first_two_conv(agent.dqn)
            reset_except_first_two_conv(agent.dqn_target)

video_folder = "videos/rainbow_tmp"


def continuous_testing(epizode):
    # this is param
    testing_ep_interval = 50
    if epizode % testing_ep_interval == 0:
        agent.test(video_folder=video_folder, video_prefix=f'rl-video-ep-{epizode}')


# agent.test(video_folder=video_folder)
agent.train(num_frames, testing_function=continuous_testing, plotting_interval=10000)

# save model after training (TODO note this will not save the optim ectr nor mem buffer
# FIXME not saving rewards, ect., this also should be a method or class
state = {
    'dqn': agent.dqn.state_dict(),
    'dqn_target': agent.dqn_target.state_dict(),
    'optimizer': agent.optimizer.state_dict(),
    'end_avg_reward': agent.end_avg_reward
    # 'memory': pickle.dumps(agent.memory), # TODO LEV not implemented yet look for pickle serialization it can
    #  be in diffrent file ofc make sure that duplicates are wrapped in LazyFrame so only one object is created
    # 'memory_n': pickle.dumps(agent.memory_n) # TODO LEV the same
}  # TODO I think mem replay save is still missing
torch.save(state, 'model_save')

# agent.test(video_folder=video_folder)


import base64
import glob
import io
import os


# from IPython.display import HTML, display


# TODO not working
def ipython_show_video(path: str) -> None:
    """Show a video at `path` within IPython Notebook."""
    if not os.path.isfile(path):
        raise NameError("Cannot access: {}".format(path))

    video = io.open(path, "r+b").read()
    encoded = base64.b64encode(video)

    # display(HTML(
    #    data="""
    #    <video width="320" height="240" alt="test" controls>
    #    <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
    #    </video>
    #    """.format(encoded.decode("ascii"))
    # ))


def show_latest_video(video_folder: str) -> str:
    """Show the most recently recorded video from video folder."""
    list_of_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    latest_file = max(list_of_files, key=os.path.getctime)
    ipython_show_video(latest_file)
    return latest_file

# latest_file = show_latest_video(video_folder=video_folder)
# print("Played:", latest_file)
