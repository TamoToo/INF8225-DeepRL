import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.init_weights import init_weights

class Critic(nn.Module):
    def __init__(
        self,
        action_space: int,
        observation_space: int,
        hidden_dim_1: int = 300,
        hidden_dim_2: int = 400
    ):
        """
        Critic network for DDPG algorithm. The critic takes the state and action as input and outputs the Q-value.
        """
        # Note: The critic network is a multi-input network, so we need to concatenate the state and action vectors.
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.fc1_dim = hidden_dim_1
        self.fc2_dim = hidden_dim_2

        self.fc1 = nn.Linear(self.observation_space, self.fc1_dim)
        self.bn1 = nn.LayerNorm(self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim + self.action_space, self.fc2_dim)
        self.bn2 = nn.LayerNorm(self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, 1)

        # Initialize weights (all of this comming from the paper "Continuous Control with Deep Reinforcement Learning" - Part 7 of the paper)
        # https://arxiv.org/pdf/1509.02971.pdf
        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.fc3, last_layer=True)

    def forward(self, state, action):
        state_value = F.relu(self.bn1(self.fc1(state)))
        action_value = F.relu(self.bn2(self.fc2(torch.cat([state_value, action], dim=1))))
        q_value = self.fc3(action_value)

        return q_value

class CriticCNN(nn.Module):
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
            nn.Conv2d(frames, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_output_size = self._get_conv_output_size((frames, height, width))

        self.fc1 = nn.Linear(conv_output_size, hidden_fc_dim)
        self.bn1 = nn.LayerNorm(hidden_fc_dim)
        self.fc2 = nn.Linear(hidden_fc_dim + action_space, hidden_fc_dim)
        self.bn2 = nn.LayerNorm(hidden_fc_dim)
        self.fc3 = nn.Linear(hidden_fc_dim, 1)

        # Initialize weights for convolutional layers
        init_weights(self.conv_layers[0])
        init_weights(self.conv_layers[2])
        init_weights(self.conv_layers[4])

        # Initialize weights for fully connected layers
        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.fc3, last_layer=True)

    def _get_conv_output_size(self, shape):
        """Calculate the size of the flattened features after conv layers"""
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)

        return int(np.prod(o.shape))
    

    def forward(self, state, action):
        # Input expected in shape (batch, height, width, frames)
        # Convert to PyTorch's expected format (batch, channels, height, width)
        if state.dim() == 4 and state.shape[3] == self.n_frames:
            state = state.permute(0, 3, 1, 2)

        conv_output = self.conv_layers(state)
        state_value = F.relu(self.bn1(self.fc1(conv_output)))
        action_value = F.relu(self.bn2(self.fc2(torch.cat([state_value, action], dim=1))))
        q_value = self.fc3(action_value)
        return q_value