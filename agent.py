import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN, DQN_CNN
from replay_buffer import ReplayBuffer


# Create the DQN Agent
class Agent:
    def __init__(
            self,
            batch_size: int,
            gamma: float,
            epsilon_start: float,
            epsilon_min: float,
            epsilon_decay: float,
            tau: float,
            lr: float,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def select_action(self, state):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError


class DQNAgent(Agent):
    def __init__(
            self,
            model: str,
            batch_size: int,
            gamma: float,
            epsilon_start: float,
            epsilon_min: float,
            epsilon_decay: float,
            tau: float,
            lr: float,
            memory_capacity: int,
            action_space: int,
            observation_space: int
    ):
        super().__init__(batch_size, gamma, epsilon_start, epsilon_min, epsilon_decay, tau, lr)
        self.action_dim = action_space
        self.obs_dim = observation_space

        # Initialize the DQN networks
        if not model:
            raise ValueError("Model name must be provided")
        if model == 'DQN':
            if isinstance(observation_space, tuple):
                observation_space = observation_space[0]
            self.q_network = DQN(action_space, observation_space).to(self.device)
            self.target_network = DQN(action_space, observation_space).to(self.device)
        elif model == 'DQN_CNN':
            self.q_network = DQN_CNN(action_space, observation_space).to(self.device)
            self.target_network = DQN_CNN(action_space, observation_space).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize the optimizer, loss function, and replay buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.memory = ReplayBuffer(memory_capacity, self.obs_dim, self.action_dim)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return torch.tensor(np.random.choice(self.action_dim))
        state = state.unsqueeze(0).to(self.device)
        return torch.argmax(self.q_network(state))
    
    def store_transition(self, state, action, next_state, reward, done):
        self.memory.store(state, action, next_state, reward, done)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size, self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_q[dones] = 0.0
            next_q = next_q.max(dim=1)[0]
            target_q = rewards + self.gamma * next_q

        loss = self.loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_epsilon()

        # Soft update of the target network weights
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))