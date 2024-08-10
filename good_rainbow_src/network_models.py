import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from good_rainbow_src.layers import NoisyLinear

#FC_SIZE = 512
FC_SIZE = 256
class Network(nn.Module):
    def __init__(
            self,
            in_dim: tuple,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor,
            conv_space=False
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.conv_layers = None
        feature_in_dim = in_dim[0]

        # set common feature layer # TODO experiment with multiple layers
        self.feature_layer = nn.Sequential(
            nn.Linear(feature_in_dim, FC_SIZE),
            nn.LeakyReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(FC_SIZE, FC_SIZE) # TODO it might be helpful to lessen noise to 0.1
        self.value_hidden_layer = NoisyLinear(FC_SIZE, FC_SIZE)

        self.advantage_layer = NoisyLinear(FC_SIZE, out_dim)
        self.value_layer = NoisyLinear(FC_SIZE, 1)

        # set value layer
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv_layers(input)
            return int(np.prod(output.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = dist

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""

        feature = self.feature_layer(x)
        adv_hid = F.leaky_relu(self.advantage_hidden_layer(feature))
        val_hid = F.leaky_relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid)
        value = self.value_layer(val_hid)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

