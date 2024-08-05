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
        if conv_space:
            conv_in_dim = in_dim
            self.transferable_conv_layers = nn.Sequential(
                nn.Conv2d(conv_in_dim[0], 32, kernel_size=8, stride=4),
                nn.LeakyReLU(),  # TODO maybe relu better as sparse
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.LeakyReLU(),
            )
            self.untransferable_conv_layers = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.LeakyReLU()
            )
            self.conv_layers = nn.Sequential( # TODO do we need so much?
                #nn.Conv2d(conv_in_dim[0], 16, kernel_size=5, stride=2),
                #nn.LeakyReLU(),
                # next ones are prev wihout dividers
                self.transferable_conv_layers,
                self.untransferable_conv_layers
            )
            conv_output_size = self._get_conv_output(conv_in_dim)
            feature_in_dim = conv_output_size

        # set common feature layer # TODO experiment with multiple layers
        self.feature_layer = nn.Sequential(
            nn.Linear(feature_in_dim, FC_SIZE),
            nn.LeakyReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(FC_SIZE, FC_SIZE) # TODO it might be helpful to lessen noise to 0.1
        self.value_hidden_layer = NoisyLinear(FC_SIZE, FC_SIZE)

        if self.atom_size is not None:
            self.advantage_layer = NoisyLinear(FC_SIZE, out_dim * atom_size)
            self.value_layer = NoisyLinear(FC_SIZE, atom_size)
        else:
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
        if self.atom_size is not None:
            q = torch.sum(dist * self.support, dim=2)
        else:
            q = dist

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        if self.conv_layers is not None:
            x = x.float() / 255.0
            x = self.conv_layers(x)
            if len(x.size()) > 3: # TODO this is not pretty
                x = x.view(x.size(0), -1)  # Flatten the conv layer output
            else:
                x = x.view(-1)  # Flatten the conv layer output

        feature = self.feature_layer(x)
        adv_hid = F.leaky_relu(self.advantage_hidden_layer(feature))
        val_hid = F.leaky_relu(self.value_hidden_layer(feature))

        if self.atom_size is not None:
            advantage = self.advantage_layer(adv_hid).view(
                -1, self.out_dim, self.atom_size
            )
            value = self.value_layer(val_hid).view(-1, 1, self.atom_size)

            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

            dist = F.softmax(q_atoms, dim=-1)
            dist = dist.clamp(min=1e-3)  # for avoiding nans
            return dist
        else:
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

