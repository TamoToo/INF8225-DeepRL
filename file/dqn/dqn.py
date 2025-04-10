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
