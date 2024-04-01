import os
import random
import threading

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import FrameStack, ResizeObservation, AtariPreprocessing

from good_rainbow_src.dqn_alg import DQNAgent
from good_rainbow_src.memory_replay import ReplayBuffer
from good_rainbow_src.wrappers import SkipWrapper
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation


def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Runner():
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.agent = self.make_agent()
        self.load()
        self.running = False

    def make_agent(self):
        os.makedirs(f'results/{self.name}/videos', exist_ok=True)

        env_data = self.config
        config = self.config

        if env_data["env_kwargs"]["id"] in {"CarRacing-v2"} or env_data["env_kwargs"]["id"][0:3] == "ALE":
            conv_space = True
        else:
            conv_space = False

        env = gym.make(**env_data["env_kwargs"], render_mode="rgb_array")
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

        print(config)
        init_seed(config["params"]["seed"])

        return DQNAgent(
            env,
            config["params"]["memory_size"],
            config["params"]["batch_size"],
            config["params"]["seed"],
            config["params"]["target_update"],
            **env_data["meta"],
            conv_space=conv_space,
            learning_interval=config["params"]["learning_interval"],
            min_memory_size=config["params"]["min_memory_size"],
            save_fig=f'results/{self.name}/plot.png'
        )

    def run_t(self):
        # plt.ion()

        def continuous_testing(epizode):
            # this is param
            testing_ep_interval = 50
            if epizode % testing_ep_interval == 0:
                self.agent.test(
                    video_folder=f'results/{self.name}/videos', video_prefix=f'rl-video-ep-{epizode}'
                )

        # agent.test(video_folder=video_folder)
        self.agent.train(self.config["params"]["num_frames"], testing_function=continuous_testing,
                         plotting_interval=self.config["plot"]["interval"])
        self.running = False

    def run(self):
        if not self.running:
            self.t = threading.Thread(target=self.run_t)
            self.running = True
            self.t.start()
            print('Agent started')
        else:
            print('Agent is running already')

    def stop(self):
        if self.running:
            print('Terminating agent (be patient)...')
            self.agent.stop = True

            try:
                self.t.join()
            except RuntimeError:
                pass
            print('Agent terminated')
        else:
            print('Agent is running already')

    def save(self, fname=None):
        running = self.running
        if self.running:
            self.stop()

        if fname is None:
            fname = f'results/{self.name}/model'

        print('Saving model & agent state')

        # save model after training (TODO note this will not save the optim ectr nor mem buffer
        # FIXME not saving rewards, ect., this also should be a method or class
        print(f'agent.memory: {str(self.agent.memory)}')
        print(f'agent.memory_n: {str(self.agent.memory_n)}')

        state = {
            'dqn': self.agent.dqn.state_dict(),
            'dqn_target': self.agent.dqn_target.state_dict(),
            'optimizer': self.agent.optimizer.state_dict(),
            'memory': self.agent.memory,
            'memory_n': self.agent.memory_n
        }
        torch.save(state, fname)

        print(np.array(self.agent.memory_n.obs_buf[0]))
        if running:
            self.run()

    def load(self, fname=None):
        if fname is None:
            fname = f'results/{self.name}/model'

        try:
            state = torch.load(fname)
            print('Loading prev model...')
            self.agent.dqn.load_state_dict(state['dqn'])
            self.agent.dqn_target.load_state_dict(state['dqn_target'])
            self.agent.optimizer.load_state_dict(state['optimizer'])
            self.agent.memory = state['memory']
            print(np.array(state['memory'].obs_buf[0]))
            self.agent.memory_n = state['memory_n']

            print(f'agent.memory: {str(self.agent.memory)}')
            print(f'agent.memory_n: {str(self.agent.memory_n)}')

        except FileNotFoundError:
            save_loaded = False
            print(f'File "{fname}" not found: starting from 0...')
