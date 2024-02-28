import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
from torch.autograd import Variable

plt.ion() 
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LR = 0.0001
NUM_EPISODES = 1000000  # TODO it's num of frames now
LEARN_START = 1000
BETA_START = 0.4
BETA_FRAMES = 10000
ALPHA = 0.6  # You can adjust this value as needed

# NoisyNet: All layers are replaced with NoisyLinear layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise() # TODO some sources say noise should be reset on every step not only when learning (investigate)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

# Dueling Network with NoisyNet
class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.LeakyReLU() #nn.ReLU() # TODO eval change
        )
        self.advantage = nn.Sequential(
            NoisyLinear(128, 128),
            nn.LeakyReLU(),#nn.ReLU(),# TODO eval change
            NoisyLinear(128, num_outputs)
        )
        self.value = nn.Sequential(
            NoisyLinear(128, 128),
            nn.LeakyReLU(), #nn.ReLU(),# TODO eval change
            NoisyLinear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

# Prioritized Experience Replay
class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, alpha=ALPHA):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # Store the alpha value
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.buffer else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = np.array(self.priorities)
        else:
            #prios = np.array(self.priorities)[:self.pos]
            prios = np.array(self.priorities)[:len(self.buffer)]  # Use the actual buffer length
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]


        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.vstack(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.vstack(batch[3])
        dones = np.array(batch[4], dtype=np.uint8)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# Update Target Network
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

# Compute Loss
def compute_td_loss(batch_size, beta):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    weights = Variable(torch.FloatTensor(weights))

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + GAMMA * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss

# Plotting
def plot(frame_idx, rewards, losses, axes):
    axes[0].clear()
    axes[0].set_title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    axes[0].plot(rewards)

    axes[1].clear()
    axes[1].set_title('loss')
    axes[1].plot(losses)

    plt.draw()
    plt.pause(0.01)

# Main Training Loop
env_id = "MountainCar-v0"
env = gym.make(env_id, render_mode="human")


current_model = DuelingDQN(env.observation_space.shape[0], env.action_space.n)
target_model = DuelingDQN(env.observation_space.shape[0], env.action_space.n)

optimizer = optim.Adam(current_model.parameters(), lr=LR)
replay_buffer = NaivePrioritizedBuffer(MEMORY_SIZE)

update_target(current_model, target_model)

losses = []
all_rewards = []
episode_reward = 0

render = False  # Set the initial render flag to False
state, _ = env.reset()
for frame_idx in range(1, NUM_EPISODES + 1):
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx / EPS_DECAY)
    beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

    with torch.no_grad():
        if random.random() > epsilon:
            state_v = torch.FloatTensor(np.float32(state)).unsqueeze(0)
            q_value = current_model(state_v)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)

    next_state, reward, terminated, truncated, info = env.step(action)
    #done = terminated #or truncated # TODO this is temporary
    done = terminated or truncated
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if render:
        env.render()  # Render the environment

    if done:
        if len(all_rewards) % 100 == 0:  # Check if it's time to render
            # Close the previous environment and create a new one with rendering enabled
            env.close()
            env = gym.make(env_id, render_mode="human") # TODO encapsulate this
        else:
            # Close the previous environment and create a new one without rendering
            env.close()
            env = gym.make(env_id)
        render = False  # Reset the render flag after the episode is done
        state, _ = env.reset()
        all_rewards.append(episode_reward)
        print(len(all_rewards))
        episode_reward = 0

    if len(replay_buffer) > LEARN_START:
        loss = compute_td_loss(BATCH_SIZE, beta)
        losses.append(loss.item())  # Make sure to use .item() to get the scalar value
        #loss = compute_td_loss(BATCH_SIZE, beta)
        #losses.append(loss.data[0])

    if frame_idx % 100 == 0:
        plot(frame_idx, all_rewards, losses, axes)

    if frame_idx % TARGET_UPDATE == 0:
        update_target(current_model, target_model)

env.close()