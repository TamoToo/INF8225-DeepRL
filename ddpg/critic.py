import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.bn2 = nn.LayerNorm(self.fc2_dim)
        self.fc1_action = nn.Linear(self.action_space, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, 1)

        # Initialize weights (all of this comming from the paper "Continuous Control with Deep Reinforcement Learning" - Part 7 of the paper)
        # https://arxiv.org/pdf/1509.02971.pdf
        f1 = 1 / np.sqrt(self.fc1.in_features)
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.in_features)
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)


    def forward(self, state, action):
        state_value = F.relu(self.bn1(self.fc1(state)))
        state_value = F.relu(self.bn2(self.fc2(state_value)))
        action_value = F.relu(self.fc1_action(action))

        x = F.relu(state_value + action_value)
        x = self.fc3(x)

        return x
