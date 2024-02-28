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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

GPU_MULTIPLY = 1
HIDDEN_SIZE = 128 * GPU_MULTIPLY//2
HIDDEN_SIZE_2 = 128 * GPU_MULTIPLY//2
# Hyperparameters
BATCH_SIZE = 32 * GPU_MULTIPLY
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 10
MEMORY_SIZE = 10000 * GPU_MULTIPLY
LR = 0.0001
NUM_EPISODES = 1000000  # TODO it's num of frames now
LEARN_START = 1000
BETA_START = 0.4
BETA_FRAMES = 10000 * GPU_MULTIPLY
ALPHA = 0.6  # You can adjust this value as needed
N_STEP = 10


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
        self.reset_noise()

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
    def __init__(self, num_inputs, num_outputs, conv_layers=False, conv_layers_shape=None, ):
        # Define the convolutional layers
        super(DuelingDQN, self).__init__()
        self.conv_layers = None
        if conv_layers:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(conv_layers_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )

            # Calculate the size of the output from conv_layers
            self.conv_output_size = self._get_conv_output(conv_layers_shape)

            self.feature = nn.Sequential(
                nn.Linear(self.conv_output_size, HIDDEN_SIZE),
                nn.LeakyReLU()  # nn.ReLU() # TODO eval change
            )
        else:
            self.feature = nn.Sequential(
                nn.Linear(num_inputs, HIDDEN_SIZE),
                nn.LeakyReLU()  # nn.ReLU() # TODO eval change
            )

        self.advantage = nn.Sequential(
            NoisyLinear(HIDDEN_SIZE, HIDDEN_SIZE_2),
            nn.LeakyReLU(),  # nn.ReLU(),# TODO eval change
            NoisyLinear(HIDDEN_SIZE_2, num_outputs)
        )
        self.value = nn.Sequential(
            NoisyLinear(HIDDEN_SIZE, HIDDEN_SIZE_2),
            nn.LeakyReLU(),  # nn.ReLU(),# TODO eval change
            NoisyLinear(HIDDEN_SIZE_2, 1)
        )

    def forward(self, x):
        # Normalize the input if it's in uint8
        if self.conv_layers is not None:
            x = x.float() / 255.0
            x = self.conv_layers(x)
            x = x.reshape(x.size(0), -1)  # Flatten the conv layer output

        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv_layers(input)
            return int(np.prod(output.size()))


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
def compute_td_loss(batch_size, beta, n_step=N_STEP):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta)

    state = torch.tensor(state, dtype=torch.float32).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)
    weights = torch.FloatTensor(weights).to(device)

    q_values = current_model(state)
    next_q_values = target_model(next_state).detach()

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + GAMMA * next_q_value * (1 - done)

    loss = (q_value - expected_q_value).pow(2) * weights
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
cp_kwargs = {"id": "CartPole-v1"}

acro_kwargs = {"id": "Acrobot-v1"}

mc_kwargs = {"id": "MountainCar-v0"}

lunar_lander_kwargs = {
    "id": "LunarLander-v2",
    "continuous": False,
    "gravity": -10.0,
    "enable_wind": False,
    "wind_power": 15.0,
    "turbulence_power": 1.5

}
rc_kwargs = {
    "id": "CarRacing-v2",
    "continuous": False
}
assault_kwargs = {
    "id": "ALE/Assault-v5"
}
pong_kwargs = {
    "id": "ALE/Pong-v5"
}
bowling_kwargs = {
    "id": "ALE/Bowling-v5"
}
mario_kwargs = {
    "id": "ALE/MarioBros-v5"
}
si_kwargs = {
    "id": "ALE/SpaceInvaders-v5"
}
pacman_kwargs = {
    "id": "ALE/Pacman-v5"
}
ms_pacman_kwargs = {
    "id": "ALE/MsPacman-v5"
}
donkey_kwargs = {
    "id": "ALE/DonkeyKong-v5"
}
defender_kwargs = {
    "id": "ALE/Defender-v5"
}
asteroids_kwargs = {
    "id": "ALE/Asteroids-v5"
}
breakout_kwargs = {
    "id": "ALE/Breakout-v5"
}

env_kwargs = rc_kwargs
env = gym.make(**env_kwargs)
if env_kwargs["id"] in {"CarRacing-v2"} or env_kwargs["id"][0:3] == "ALE":
    conv_space = True
else:
    conv_space = False

if conv_space:
    observation_shape = env.observation_space.shape
    input_shape = (observation_shape[2], observation_shape[0], observation_shape[1])
    current_model = DuelingDQN(env.observation_space.shape[0], env.action_space.n, conv_layers=True,
                               conv_layers_shape=input_shape).to(device)
    target_model = DuelingDQN(env.observation_space.shape[0], env.action_space.n, conv_layers=True,
                              conv_layers_shape=input_shape).to(device)
else:
    current_model = DuelingDQN(env.observation_space.shape[0], env.action_space.n, conv_layers=False).to(device)
    target_model = DuelingDQN(env.observation_space.shape[0], env.action_space.n, conv_layers=False).to(device)

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
            # state_v = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            state_v = torch.tensor(state, dtype=torch.float32).to(device)
            if conv_space:
                print(state_v)
                state_v = state_v.permute(2, 0, 1)
            state_v = state_v.unsqueeze(0)

            q_value = current_model(state_v)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)

    next_state, reward, terminated, truncated, info = env.step(action)
    # done = terminated #or truncated # TODO this is temporary
    done = terminated or truncated

    if conv_space:
        state_v = torch.tensor(state).permute(2, 0, 1).unsqueeze(0)
        next_state_v = torch.tensor(next_state).permute(2, 0, 1).unsqueeze(0)
    else:
        state_v = torch.tensor(state)
        next_state_v = torch.tensor(next_state)

    replay_buffer.push(state_v, action, reward, next_state_v, done)  # permute state

    state = next_state
    episode_reward += reward

    if done:
        if len(all_rewards) % 10 == 2:  # Check if it's time to render
            # Close the previous environment and create a new one with rendering enabled
            env.close()
            env = gym.make(**env_kwargs, render_mode="human")  # TODO encapsulate this
        else:
            # Close the previous environment and create a new one without rendering
            env.close()
            env = gym.make(**env_kwargs)
        render = False  # Reset the render flag after the episode is done
        state, _ = env.reset()
        all_rewards.append(episode_reward)
        print(len(all_rewards))
        episode_reward = 0

    if len(replay_buffer) > LEARN_START:
        loss = compute_td_loss(BATCH_SIZE, beta)
        losses.append(loss.item())  # Make sure to use .item() to get the scalar value
        # loss = compute_td_loss(BATCH_SIZE, beta)
        # losses.append(loss.data[0])

    if frame_idx % 100 == 0:
        plot(frame_idx, all_rewards, losses, axes)

    if frame_idx % TARGET_UPDATE == 0:
        update_target(current_model, target_model)

env.close()
