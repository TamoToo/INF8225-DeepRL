import numpy as np
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(
            self,
            action_space: int,
            observation_space: int,
            hidden_dim_1: int = 128,
            hidden_dim_2: int = 128
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_space, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, action_space)
        )

    def forward(self, x):
        return self.network(x)

class DQN_CNN(nn.Module):
    def __init__(self, action_space: int, observation_space: int):
        super().__init__()
        width, height, frames = observation_space
        self.n_frames = frames

        # Note: We don't use max pooling here, as it can lead to information loss. (we already have information about spatial location by stacking 4 images as the input)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(frames, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_output_size = self._get_conv_output_size((frames, height, width))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )
    
    def _get_conv_output_size(self, shape):
        """Calculate the size of the flattened features after conv layers"""
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)

        return int(np.prod(o.shape))

    def forward(self, x):
        # Input expected in shape (batch, height, width, frames)
        # Convert to PyTorch's expected format (batch, channels, height, width)
        if x.dim() == 4 and x.shape[3] == self.n_frames:  # If in NHWC format with 1 channel
            x = x.permute(0, 3, 1, 2)

        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x