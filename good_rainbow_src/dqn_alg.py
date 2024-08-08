import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import matplotlib
import torch.nn.functional as F

from good_rainbow_src.utils import downsample_data, sum_rewards

from torch import optim
from typing import Dict, List, Tuple

from torch.nn.utils import clip_grad_norm_

from good_rainbow_src.memory_replay import PrioritizedReplayBuffer, ReplayBuffer
from good_rainbow_src.network_models import Network


def softmax(x):
    """
    Compute the softmax of vector x.

    Parameters:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Softmax of the input tensor.
    """
    return torch.nn.functional.softmax(x, dim=0)


class FrozenClass(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


class TrainingState(FrozenClass):
    def __init__(self, env, conv_space, correct_dims, action_dim):
        self.is_test = False

        self.state, _ = env.reset()  # seed=self.seed)

        self.update_cnt = 0
        self.ep_step = 0
        self.ep_id = 0
        # TODO we need class for that
        self.losses = []
        self.score = 0
        self.last_frame_ep_end = 1
        self.action_hist = np.zeros(action_dim)
        self._freeze()  # no new attributes after this point.


class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            seed: int,
            target_update: int,
            gamma: float = 0.99,
            # PER parameters
            alpha: float = 0.5,  # 0.2,
            beta: float = 0.4,  # 0.6,
            prior_eps: float = 1e-6,
            # Categorical DQN parameters
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 3,
            conv_space: bool = False,
            min_memory_size: int = 80000,
            learning_interval: int = 4,  # is for 32 batch_size
            epsilon_decay: float = 1 / 100000,  # min after 1M epizodes
            max_epsilon: float = 1,
            min_epsilon: float = 0.01,
            correct_dims: bool = False,
            reward_clip=None,
            reward_scale=None,
            save_stats_path=None,
            # guided network experiment
            guided_network=None,  # ['model_path', 'other_model_path' ]
            guided_temp_decay: float = 0.02,  # min after 1M epizodes
            guided_max_temp: float = 1,
            guided_min_temp: float = 0,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """

        self.start_time = time.time()
        self.max_steps = env.spec.max_episode_steps if env.spec.max_episode_steps is not None else env.max_steps
        self.ep_rewards_buffer = np.zeros(self.max_steps, dtype=np.float32)
        self.save_stats_path = save_stats_path
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.correct_dims = correct_dims
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay#*len(guided_network)
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        self.min_memory_size = min_memory_size
        self.learning_interval = learning_interval
        self.conv_space = conv_space
        if env.observation_space.shape == ():
            obs_dim = 1  # raw int as input for state shape, use one-hot wrapper if want
        else:
            obs_dim = env.observation_space.shape
        print("DIMS:")
        print(obs_dim)

        if len(obs_dim) > 1 and correct_dims:
            obs_dim = (obs_dim[2], obs_dim[0], obs_dim[1])
        action_dim = env.action_space.n
        self.action_dim = action_dim

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        obs_type = np.uint8 if env.observation_space.dtype == np.int32 else env.observation_space.dtype
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma, obs_type=obs_type
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma, obs_type=env.observation_space.dtype
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = None

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support, conv_space=conv_space
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support, conv_space=conv_space
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer # it was lr=0.0000625
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-3*4, eps=1.5 * 1e-4)

        # transition to store in memory
        self.transition = list()

        # ploting utils
        self.target_update_frames = []

        self.teacher_network = None
        # guided network experiment
        print("pre-guid")
        print(guided_network)
        if guided_network is not None:
            # self.end_avg_reward = 0
            print("guid")
            self.guided_temp_decay = guided_temp_decay
            self.guided_max_temp = guided_max_temp
            self.guided_min_temp = guided_min_temp
            self.guided_temperature = self.guided_min_temp

            self.teacher_network = []

            self.teacher_avg_reward = np.zeros(len(guided_network))
            self.teacher_avg_reward.fill(1 / (len(guided_network) + 1))
            self.current_ep_teacher_times_used = np.zeros(len(guided_network))
            self.teacher_usage_ratio_sum = np.zeros(len(guided_network))
            self.teacher_usage_ratio_sum.fill(1/ (len(guided_network) + 1))

            self.current_ep_student_times_used = 0
            self.student_avg_reward = 1 / (len(guided_network) + 1)  # TODO (v_max-v_min)
            self.student_usage_ratio_sum =  1/(len(guided_network) + 1)

            self.current_ep_random_times_used = 0
            self.random_avg_reward = 1 / (len(guided_network) + 1)  # TODO (v_max-v_min)
            self.random_usage_ratio_sum = 1/(len(guided_network) + 1) # just for consistency
            self.probs = None

            for fname in guided_network:
                state = torch.load(fname)

                # effic
                temperatured_values = [self.student_avg_reward, *self.teacher_avg_reward]
                self.probs = softmax(
                    torch.tensor([i * self.guided_temperature for i in temperatured_values], dtype=torch.float64))

                self.teacher_network.append(Network(
                    obs_dim, action_dim, self.atom_size, self.support, conv_space=conv_space
                ).to(self.device))
                self.teacher_network[-1].load_state_dict(state["dqn"])
            self.chosen_policy = 0

        # end of guided network experiment
        self.training_state = TrainingState(self.env, self.conv_space, self.correct_dims, self.action_dim)

        self.ep_stats_labels = ['frame_idx', 'score', 'disc_score', 'ep_len', 'action_hist',
                                'models_usage_hist', 'models_avg_reward', 'model_choose_probs']  # TODO finish when more
        self.loss_stats_labels = ['frame_idx', 'loss_value']

        with open(f"{self.save_stats_path}/train_ep_data.csv", mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.ep_stats_labels)
            writer.writeheader()

        with open(f"{self.save_stats_path}/train_loss_data.csv", mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.loss_stats_labels)
            writer.writeheader()

        with open(f"{self.save_stats_path}/target_updates_frames.csv", mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['frame_idx'])
            writer.writeheader()

    def select_action(self, state: np.ndarray, force_epsilon=None, network=None) -> np.ndarray:
        if network is None:
            network = self.dqn
        """Select an action from the input state."""
        epsilon = force_epsilon if force_epsilon is not None else self.epsilon
        # NoisyNet: no epsilon greedy action selection
        if len(self.memory) < self.batch_size or len(
                self.memory) < self.min_memory_size:  # this might not be needed if noice reseted
            # get random action for beginning of learning (NoisyNet feels not enough)
            # selected_action = random.randrange(self.env.action_space.n) # old one probably worse
            # selected_action = self.env.action_space.sample()
            self.dqn.reset_noise()  # FIXME workaround to ensure exploration during initial frames
            self.dqn_target.reset_noise()

        if epsilon > np.random.random():  # fixme added back epsilon probably not needed but lets keep it just for test
            selected_action = self.env.action_space.sample()
            self.chosen_policy = -1
        else:
            selected_action = network(
                torch.FloatTensor(np.array(state, dtype=np.float32)).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        # mode: train / test
        if not self.training_state.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(
            int(action))  # TODO this is just workaround for frozen lake
        if self.reward_clip is not None:
            reward = min(max(reward, self.reward_clip[0]), self.reward_clip[1])

        if self.reward_scale is not None:
            reward = reward + self.reward_scale[1]
            reward = reward * self.reward_scale[0]

        done = terminated or truncated

        if not self.training_state.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # TODO no transitions when buffer isn't full (transitions in buffer are not applied for last elements?)
            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train_init(self):
        self.training_state = TrainingState(self.env, self.conv_space, self.correct_dims, self.action_dim)

    def save_stats(self, frame_idx):
        # Prepare data
        self.training_state.action_hist = [i / (sum(self.training_state.action_hist)) for i in
                                           self.training_state.action_hist]
        self.training_state.action_hist = np.flip(np.cumsum(np.flip(self.training_state.action_hist)))

        # Save data
        with open(f"{self.save_stats_path}/train_ep_data.csv", mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.ep_stats_labels)
            # frame idx
            writer.writerow({
                'frame_idx': frame_idx,
                'score': self.training_state.score,
                'disc_score': self.training_state.score,  # TODO fix it g_score
                'ep_len': frame_idx - self.training_state.last_frame_ep_end,
                'action_hist': self.training_state.action_hist,
                'models_usage_hist': [i/(frame_idx - self.training_state.last_frame_ep_end) for i in[self.current_ep_random_times_used, self.current_ep_student_times_used,
                                      *self.current_ep_teacher_times_used]] if self.teacher_network is not None else None,
                'models_avg_reward': [self.random_avg_reward, self.student_avg_reward, *self.teacher_avg_reward ] if self.teacher_network is not None else None,
                'model_choose_probs': [float(i) for i in self.probs] if self.probs is not None else None
            })

        with open(f"{self.save_stats_path}/train_loss_data.csv", mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.loss_stats_labels)
            for i, l in self.training_state.losses:
                writer.writerow({
                    'frame_idx': i,
                    'loss_value': l,
                })
            self.training_state.losses.clear()

        with open(f"{self.save_stats_path}/target_updates_frames.csv", mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['frame_idx'])
            for i in self.target_update_frames:
                writer.writerow({
                    'frame_idx': i,
                })
            self.target_update_frames.clear()
        with open(f'{self.save_stats_path}/current_speed.csv', 'w') as file:
            # Write a string to the file
            file.write(f"{int(60*60* frame_idx/(time.time()-self.start_time))}")

    def reset_interep_stats(self, frame_idx):
        self.training_state.score = 0
        self.training_state.last_frame_ep_end = frame_idx
        self.training_state.action_hist.fill(0)
        if self.teacher_network is not None:
            self.current_ep_random_times_used = 0
            self.current_ep_student_times_used = 0
            self.current_ep_teacher_times_used.fill(0)

    def update_stats(self, frame_idx):
        self.save_stats(frame_idx)
        self.reset_interep_stats(frame_idx)

    def adjust_array(self, arr, some_var):
        # Step 1: Calculate the minimum threshold
        arr = arr.numpy()
        min_threshold = some_var

        # Step 2: Adjust elements below the threshold
        adjusted_arr = np.maximum(arr, min_threshold)

        # Step 3: Calculate the excess
        excess = np.sum(adjusted_arr) - 1

        # Step 4: Distribute the excess proportionally
        if excess > 0:
            # Find elements that were above the threshold
            above_threshold_indices = arr > min_threshold
            above_threshold_values = arr[above_threshold_indices]

            # Calculate the total sum of elements above the threshold
            total_above_threshold = np.sum(above_threshold_values)

            # Reduce the elements proportionally
            reduction_factors = above_threshold_values / total_above_threshold
            reduction_amounts = reduction_factors * excess

            # Apply the reductions
            adjusted_arr[above_threshold_indices] -= reduction_amounts

        return adjusted_arr

    def adjust_probabilities(self, probs):
        old_probs = probs
        n = len(probs)
        min_first_prob = 1 / (n+1)
        # Ensure the sum of the probabilities is 1
        probs = self.adjust_array(probs, self.min_epsilon/n)
        if probs[0] < min_first_prob:
            total_sum = sum(probs)
            # Calculate the sum of the original remaining probabilities
            original_remaining_sum = total_sum - probs[0]
            # Set the first element to min_first_prob
            probs[0] = max(probs[0], min_first_prob)
            # Calculate the sum of the remaining probabilities
            remaining_sum = 1 - probs[0]
            # Adjust the remaining probabilities proportionally
            for i in range(1, n):
                probs[i] = probs[i] * remaining_sum / original_remaining_sum
        if np.isnan(probs).any() or not np.isclose(sum(probs), 1):
            print("Computation error use uncorrected probs")
            return old_probs
        else:
            return probs

    def guided_epsilon_policy_choose(self):
        temperatured_values = [self.student_avg_reward, *self.teacher_avg_reward]
        if self.probs is None:
            self.probs = softmax(torch.tensor([i * self.guided_temperature for i in temperatured_values], dtype=torch.float64))
            self.probs = self.adjust_probabilities(self.probs)


        self.chosen_policy = np.random.choice(range(len(self.probs)), p=self.probs)
        if self.chosen_policy == 0:
            network = self.dqn
        else:
            network = self.teacher_network[self.chosen_policy - 1]
        action = self.select_action(self.training_state.state, network=network)
        return action

    def action_selection_policy_choose(self):
        if self.teacher_network is not None:
            action = self.guided_epsilon_policy_choose()
        # other strategies
        else:
            action = self.select_action(self.training_state.state)
        return action

    def reward_update(self, total_avg, weight_sum, epizod_n, current_times_used, computed_rewards, ep_steps):
        return (0.999*total_avg * weight_sum + (current_times_used / ep_steps) * computed_rewards)/(
                0.999*weight_sum + (current_times_used / ep_steps)
        ), 0.999*weight_sum + (current_times_used / ep_steps)
        #return (0.99*total_avg * epizod_n + 1.01*(current_times_used / ep_steps) * computed_rewards) / (epizod_n + 1)

    def train_step(self, frame_idx, num_frames: int, testing_function=None):
        action = self.action_selection_policy_choose()

        self.training_state.action_hist[action] += 1
        next_state, reward, done = self.step(action)

        self.training_state.state = next_state
        self.training_state.score += reward
        self.ep_rewards_buffer[self.training_state.ep_step] = reward

        # NoisyNet: removed decrease of epsilon # TODO added as NoisyNet encapsulate randomnes and network can learn to remove randomnes from the input to much in early learing
        self.epsilon = max(
            self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
            ) * self.epsilon_decay
        )
            #self.guided_temperature = max(
            #    self.guided_min_temp, self.epsilon - (
            #            self.guided_max_temp - self.guided_min_temp
            #    ) * self.guided_temp_decay
            #)

        # PER: increase beta
        fraction = min(frame_idx / num_frames, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
        self.training_state.ep_step += 1

        if self.teacher_network is not None:
            if self.chosen_policy == -1:
                self.current_ep_random_times_used += 1
            elif self.chosen_policy == 0:
                self.current_ep_student_times_used += 1
            else:
                self.current_ep_teacher_times_used[self.chosen_policy - 1] += 1

        # if episode ends
        if done:
            # TODO Am I sure?
            if self.teacher_network is not None:
                self.guided_temperature += self.guided_temp_decay
            if self.teacher_network is not None:
                self.random_avg_reward, self.random_usage_ratio_sum = self.reward_update(self.random_avg_reward, self.random_usage_ratio_sum, self.training_state.ep_id,
                                                            self.current_ep_random_times_used,
                                                            self.training_state.score, self.training_state.ep_step)

                self.student_avg_reward, self.student_usage_ratio_sum = self.reward_update(self.student_avg_reward, self.student_usage_ratio_sum, self.training_state.ep_id,
                                                             self.current_ep_student_times_used,
                                                             self.training_state.score, self.training_state.ep_step)

                self.teacher_avg_reward, self.teacher_usage_ratio_sum = self.reward_update(self.teacher_avg_reward, self.teacher_usage_ratio_sum, self.training_state.ep_id,
                                                             # should work as vector operation
                                                             self.current_ep_teacher_times_used,
                                                             self.training_state.score, self.training_state.ep_step)

            self.update_stats(frame_idx)
            self.probs = None

            self.training_state.ep_id += 1
            self.training_state.ep_step = 0
            self.ep_rewards_buffer.fill(0)

            # FIXME need to test before reset god knows why
            if testing_function is not None and len(self.memory) >= self.min_memory_size:  # we dont want random
                testing_function(self.training_state.ep_id)
            self.training_state.state, _ = self.env.reset()  # seed=self.seed)# TODO ensure that is deterministic commenting because of all same env

        # if training is ready
        if len(self.memory) >= self.batch_size and len(
                self.memory) >= self.min_memory_size and frame_idx % self.learning_interval == 0:
            loss = self.update_model()
            self.training_state.losses.append((frame_idx, loss))
            self.training_state.update_cnt += 1

            # if hard update is needed
            if self.training_state.update_cnt % self.target_update == 0:
                self.target_update_frames.append(frame_idx)
                self._target_hard_update()

    def train(self, num_frames: int, plotting_interval: int = 1000000, testing_function=None):
        """Train the agent."""
        self.train_init()
        for frame_idx in range(1, num_frames + 1):
            self.train_step(frame_idx, num_frames, testing_function)
        self.env.close()

    # TODO we dont really need test for recording during learning as RecordVideo enables ep_number trigger
    def test(self, video_folder: str, video_prefix="rl-video") -> None:
        """Test the agent."""
        self.training_state.is_test = True

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, name_prefix=video_prefix, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env
        self.training_state.is_test = False

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(np.stack(samples["obs"])).to(device)
        next_state = torch.FloatTensor(np.stack(samples["next_obs"])).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
