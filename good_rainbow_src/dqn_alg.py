import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import matplotlib
import torch.nn.functional as F

from good_rainbow_src.utils import downsample_data

matplotlib.pyplot.rcParams['agg.path.chunksize'] = 200000

matplotlib.use('Agg')
matplotlib.pyplot.rcParams['agg.path.chunksize'] = 200000

from matplotlib import pyplot as plt
plt.rcParams['agg.path.chunksize'] = 200000

from torch import optim
from typing import Dict, List, Tuple

from torch.nn.utils import clip_grad_norm_

from good_rainbow_src.memory_replay import PrioritizedReplayBuffer, ReplayBuffer
from good_rainbow_src.network_models import Network


class FrozenClass(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


class TrainingState(FrozenClass):
    def __init__(self, env, conv_space, correct_dims, action_dim):
        self.is_test = False
        self.stop = False

        self.state, _ = env.reset()  # seed=self.seed)
        if conv_space is True and correct_dims:
            self.state = self.state.transpose(2, 0, 1)

        self.update_cnt = 0
        # TODO we need class for that
        self.losses = []
        self.scores = []
        self.ep_lens = []
        self.action_histograms = []
        self.guide_epsilons = []
        self.score = 0
        self.last_frame_ep_end = 1
        self.action_hist = np.zeros(action_dim)
        self.guide_epsilon = 1
        self._freeze() # no new attributes after this point.

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
            min_epsilon: float = 0.001,
            correct_dims: bool = False,
            reward_clip=None,
            reward_scale=None,
            save_fig=None,
            # guided network experiment
            guided_network=None
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

        self.save_fig = save_fig
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.correct_dims = correct_dims
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        self.min_memory_size = min_memory_size
        self.learning_interval = learning_interval
        self.conv_space = conv_space
        if env.observation_space.shape == ():
            obs_dim = 1 # raw int as input for state shape, use one-hot wrapper if want
        else:
            obs_dim = env.observation_space.shape
        print("DIMS:")
        print(obs_dim)
        #print(env.observation_space)

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
        if atom_size is not None:
            self.support = torch.linspace(
                self.v_min, self.v_max, self.atom_size
            ).to(self.device)
        else:
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

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.0000625, eps=1.5 * 10 ** -4)

        # transition to store in memory
        self.transition = list()


        # ploting utils
        self.target_update_eps = []
        self.target_update_steps = []
        self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 5))

        self.teacher_network = None
        # guided network experiment
        if guided_network is not None:
            # self.end_avg_reward = 0
            self.teacher_network = Network(
                obs_dim, action_dim, self.atom_size, self.support, conv_space=conv_space
            ).to(self.device)
            self.teacher_network.load_state_dict(guided_network["model"])
            self.teacher_avg_reward = guided_network["reward"]
            self.init_avg_reward = sys.float_info.min / 100
            self.end_avg_reward = sys.float_info.min / 100
        # end of guided network experiment
        self.training_state = TrainingState(self.env, self.conv_space, self.correct_dims, self.action_dim)
        self.training_state.stop = False

    def select_action(self, state: np.ndarray, force_epsilon=None, network=None) -> np.ndarray:
        if network is None:
            network = self.dqn
        """Select an action from the input state."""
        epsilon = force_epsilon if force_epsilon is not None else self.epsilon
        # NoisyNet: no epsilon greedy action selection
        if len(self.memory) < self.batch_size and len(
                self.memory) < self.min_memory_size:  # this might not be needed if noice reseted
            # get random action for beginning of learning (NoisyNet feels not enough)
            # selected_action = random.randrange(self.env.action_space.n) # old one probably worse
            # selected_action = self.env.action_space.sample()
            self.dqn.reset_noise()  # FIXME workaround to ensure exploration during initial frames
            self.dqn_target.reset_noise()

        if epsilon > np.random.random():  # fixme added back epsilon probably not needed but lets keep it just for test
            selected_action = self.env.action_space.sample()
        else:
            selected_action = network(
                torch.FloatTensor(np.array(state, dtype=np.float32)).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        # mode: train / test
        self.training_state.is_test = False
        if not self.training_state.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(int(action)) # TODO this is just workaround for frozen lake
        if self.reward_clip is not None:
            reward = min(max(reward, self.reward_clip[0]), self.reward_clip[1])

        if self.reward_scale is not None:
            reward = reward + self.reward_scale[1]
            reward = reward * self.reward_scale[0]

        if self.conv_space is True and self.correct_dims:
            next_state = next_state.transpose(2, 0, 1)  # TODO maybe just transpose it in nn
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

    def train_step(self, frame_idx, num_frames: int, plotting_interval: int = 10000, testing_function=None):
        if self.teacher_network is not None and self.end_avg_reward < self.teacher_avg_reward:
            self.training_state.guide_epsilon = min(max((self.teacher_avg_reward - self.end_avg_reward) / (
                    self.teacher_avg_reward - self.init_avg_reward), 0), 1)
            if self.training_state.guide_epsilon > np.random.random() and len(self.memory) >= self.min_memory_size:
                teacher = self.teacher_network
                action = self.select_action(self.training_state.state, network=teacher)
            else:
                action = self.select_action(self.training_state.state, network=self.dqn, force_epsilon=0.75 * self.training_state.guide_epsilon)
        else:
            action = self.select_action(self.training_state.state)
        self.training_state.action_hist[action] += 1
        next_state, reward, done = self.step(action)

        self.training_state.state = next_state
        self.training_state.score += reward

        # NoisyNet: removed decrease of epsilon # TODO added as NoisyNet encapsulate randomnes and network can learn to remove randomnes from the input to much in early learing
        self.epsilon = max(
            self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
            ) * self.epsilon_decay
        )

        # PER: increase beta
        fraction = min(frame_idx / num_frames, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        # if episode ends
        if done:
            # FIXME need to test before reset god knows why
            if testing_function is not None and len(self.memory) >= self.min_memory_size:  # we dont want random
                testing_function(len(self.training_state.scores))
            self.training_state.state, _ = self.env.reset()  # seed=self.seed)# TODO ensure that is deterministic commenting because of all same env
            if self.conv_space is True and self.correct_dims:
                self.training_state.state = self.training_state.state.transpose(2, 0, 1)
            self.training_state.scores.append(self.training_state.score)
            self.training_state.ep_lens.append(frame_idx - self.training_state.last_frame_ep_end)
            self.training_state.action_hist = [i / (sum(self.training_state.action_hist)) for i in self.training_state.action_hist]
            self.training_state.action_hist = np.flip(np.cumsum(np.flip(self.training_state.action_hist)))
            self.training_state.action_histograms.append(self.training_state.action_hist)
            self.training_state.guide_epsilons.append(self.training_state.guide_epsilon)
            self.training_state.score = 0
            self.training_state.last_frame_ep_end = frame_idx
            self.training_state.action_hist = np.zeros(self.action_dim)
            self.end_avg_reward = sum(self.training_state.scores[-10:]) / 10

        # if training is ready
        if len(self.memory) >= self.batch_size and len(
                self.memory) >= self.min_memory_size and frame_idx % self.learning_interval == 0:
            loss = self.update_model()
            self.training_state.losses.append(loss)
            self.training_state.update_cnt += 1

            # if hard update is needed
            if self.training_state.update_cnt % self.target_update == 0:
                self.target_update_steps.append(len(self.training_state.losses))
                self.target_update_eps.append(len(self.training_state.scores))
                self._target_hard_update()

        # plotting
        if frame_idx % plotting_interval == 0:
            self._plot(frame_idx, self.training_state.scores, self.training_state.losses, self.training_state.ep_lens,
                       self.training_state.action_histograms)  # ,guide_epsilons)  # add action distribution, and when plotting we can add noise param avg and variance

    def train(self, num_frames: int, plotting_interval: int = 1000000, testing_function=None):
        """Train the agent."""
        self.train_init()
        for frame_idx in range(1, num_frames + 1):
            self.train_step(frame_idx, num_frames, plotting_interval, testing_function)
            if self.training_state.stop and frame_idx % plotting_interval == 0:
                break  # TODO breaking not workin after moving
        self.env.close()

    # TODO we dont really need test for recording during learning as RecordVideo enables ep_number trigger
    def test(self, video_folder: str, video_prefix="rl-video") -> None:
        """Test the agent."""
        self.training_state.is_test = True

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, name_prefix=video_prefix, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        if self.conv_space is True and self.correct_dims:
            state = state.transpose(2, 0, 1)
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
        if self.atom_size is not None:
            action = torch.LongTensor(samples["acts"]).to(device)
        else:
            action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        if self.atom_size is not None:
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
            with torch.no_grad():
                # Double DQN
                next_action = self.dqn(next_state).argmax(1)
                next_dist = self.dqn_target.dist(next_state)
                next_dist = next_dist[range(self.batch_size), next_action]

                t_z = reward + (1 - done) * gamma * self.support
                t_z = t_z.clamp(min=self.v_min, max=self.v_max)
                b = (t_z - self.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                offset = (
                    torch.linspace(
                        0, (self.batch_size - 1) * self.atom_size, self.batch_size
                    ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
                )

                proj_dist = torch.zeros(next_dist.size(), device=self.device)
                proj_dist.view(-1).index_add_(
                    0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                )
                proj_dist.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )

            dist = self.dqn.dist(state)
            log_p = torch.log(dist[range(self.batch_size), action])
            elementwise_loss = -(proj_dist * log_p).sum(1)
        else:
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

    def _plot(self, frame_idx, rewards, losses, ep_lens, action_histograms):
        if self.axes is not None:
            def plot_trends(ax_index, data):
                cumsum_vec = np.cumsum(np.insert(data, 0, 0))
                #ma_vec_50 = (cumsum_vec[50:] - cumsum_vec[:-50]) / 50
                #ma_vec_100 = (cumsum_vec[100:] - cumsum_vec[:-100]) / 100
                ma_vec_200 = (cumsum_vec[200:] - cumsum_vec[:-200]) / 200
                #if len(ma_vec_50) > 1:
                #    self.axes[ax_index].plot(np.pad(ma_vec_50, (len(data) - len(ma_vec_50), 0), 'edge'), color='y')
                #if len(ma_vec_100) > 1:
                #    self.axes[ax_index].plot(np.pad(ma_vec_100, (len(data) - len(ma_vec_100), 0), 'edge'), color='r')
                if len(ma_vec_200) > 1:
                    self.axes[ax_index].plot(np.pad(ma_vec_200, (len(data) - len(ma_vec_200), 0), 'edge'), color='k')

            self.axes[0].clear()
            self.axes[0].set_title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))

            #for target_update_ep in self.target_update_eps:
            #    self.axes[0].axvline(x=target_update_ep, color='g')
            self.axes[0].plot(rewards)
            plot_trends(0, rewards)

            self.axes[1].clear()
            self.axes[1].set_title('loss')

            #for target_update_step in self.target_update_steps:
            #    self.axes[1].axvline(x=target_update_step, color='g')

            self.axes[1].plot(losses)
            plot_trends(1, losses)

            self.axes[2].clear()
            self.axes[2].set_title('Ep len')

            #for target_update_ep in self.target_update_eps:
            #    self.axes[2].axvline(x=target_update_ep, color='g')

            self.axes[2].plot(ep_lens)
            plot_trends(2, ep_lens)

            self.axes[3].clear()
            self.axes[3].set_title('Action histogram')

            # for target_update_ep in self.target_update_eps:
            #    self.axes[3].axvline(x=target_update_ep, color='g')

            action_histograms_tmp, action_idx = downsample_data(np.array(action_histograms),100000)
            self.axes[3].plot(action_idx, action_histograms_tmp)
            for i in range(action_histograms_tmp.shape[1]):
                if 1 + i < action_histograms_tmp.shape[1]:
                    self.axes[3].fill_between(np.arange(action_histograms_tmp.shape[0]), action_histograms_tmp[:, i],
                                              action_histograms_tmp[:, i + 1])
                else:
                    self.axes[3].fill_between(np.arange(action_histograms_tmp.shape[0]), action_histograms_tmp[:, i])

            # plt.draw()
            if self.save_fig is not None:
                self.fig.savefig(self.save_fig)
            # plt.pause(0.01)

            # Guided network experiment
            cumsum_vec = np.cumsum(np.insert(rewards, 0, 0))
            ma_vec_100 = (cumsum_vec[100:] - cumsum_vec[:-100]) / 100

            if len(ma_vec_100) > 0 and len(self.memory) >= self.min_memory_size:
                if self.teacher_network is not None:
                    if self.init_avg_reward == sys.float_info.min / 100:
                        self.init_avg_reward = ma_vec_100[-1]
                # self.end_avg_reward = ma_vec_100[-1]

    # TODO not needed
    def _plot_ipython3(
            self,
            frame_idx: int,
            scores: List[float],
            losses: List[float],
    ):
        """Plot the training progresses."""
        # clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()
