import numpy as np
import torch.nn as nn
import torch
from utils.init_weights import init_weights

class Actor(nn.Module):
    def __init__(
        self,
        action_space: int,
        observation_space: int,
        hidden_dim_1: int = 300,
        hidden_dim_2: int = 400
    ):
        """
        Actor network for DDPG algorithm. The actor takes the state as input and outputs the action.
        """
        # Note: The actor network is a single-input network, so we only need to pass the state vector.
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.fc1_dim = hidden_dim_1
        self.fc2_dim = hidden_dim_2

        self.network = nn.Sequential(
            nn.Linear(self.observation_space, self.fc1_dim),
            nn.LayerNorm(self.fc1_dim),
            nn.ReLU(),
            nn.Linear(self.fc1_dim, self.fc2_dim),
            nn.LayerNorm(self.fc2_dim),
            nn.ReLU(),
            nn.Linear(self.fc2_dim, self.action_space),
            nn.Tanh() # Output layer with Tanh activation for continuous action space [-1, 1]
        )

        # Initialize weights (all of this comming from the paper "Continuous Control with Deep Reinforcement Learning" - Part 7 of the paper)
        # https://arxiv.org/pdf/1509.02971.pdf
        init_weights(self.network[0])
        init_weights(self.network[3])
        init_weights(self.network[6], last_layer=True)
        
    def forward(self, x):
        return self.network(x)


class ActorCNN(nn.Module):
    def __init__(
            self,
            action_space: int,
            observation_space: int,
            hidden_fc_dim: int = 200
    ):
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        width, height, frames = observation_space
        self.n_frames = frames

        self.conv_layers = nn.Sequential(
            nn.Conv2d(frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_output_size = self._get_conv_output_size((frames, height, width))

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, hidden_fc_dim),
            nn.ReLU(),
            nn.Linear(hidden_fc_dim, hidden_fc_dim),
            nn.ReLU(),
            nn.Linear(hidden_fc_dim, self.action_space),
            nn.Tanh()
        )

        # Initialize weights for convolutional layers
        init_weights(self.conv_layers[0])
        init_weights(self.conv_layers[2])
        init_weights(self.conv_layers[4])

        # Initialize weights for fully connected layers
        init_weights(self.fc_layers[0])
        init_weights(self.fc_layers[2])
        init_weights(self.fc_layers[4], last_layer=True)


    def _get_conv_output_size(self, shape):
        """Calculate the size of the flattened features after conv layers"""
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)

        return int(np.prod(o.shape))
    

    def forward(self, x):
        # Input expected in shape (batch, height, width, frames)
        # Convert to PyTorch's expected format (batch, channels, height, width)
        if x.dim() == 4 and x.shape[3] == self.n_frames:
            x = x.permute(0, 3, 1, 2)

        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x