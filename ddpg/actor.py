import numpy as np
import torch.nn as nn


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
        f1 = 1 / np.sqrt(self.network[0].in_features)
        nn.init.uniform_(self.network[0].weight.data, -f1, f1)
        nn.init.uniform_(self.network[0].bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.network[3].in_features)
        nn.init.uniform_(self.network[3].weight.data, -f2, f2)
        nn.init.uniform_(self.network[3].bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.network[6].weight.data, -f3, f3)
        nn.init.uniform_(self.network[6].bias.data, -f3, f3)

        
    def forward(self, x):
        return self.network(x)
