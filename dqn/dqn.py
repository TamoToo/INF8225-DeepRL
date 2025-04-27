from dqn.network import DQN, DQN_CNN
from interface.agent import AgentAbstract
from utils.replay_buffer import ReplayBuffer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os

class Agent(AgentAbstract):
    def __init__(
            self,
            name: str,
            device: str,
            batch_size: int,
            gamma: float,
            epsilon_start: float,
            epsilon_min: float,
            epsilon_decay: float,
            tau: float,
            lr: float,
            memory_capacity: int,
            action_space: int,
            observation_space: int,
            model_type: str = "DQN"
    ):
        super().__init__(name, device, batch_size, gamma, tau)
        self.action_dim = action_space
        self.obs_dim = observation_space
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.memory_capacity = memory_capacity
        self.lr = lr

        if model_type == "DQN":
            if isinstance(observation_space, tuple):
                observation_space = observation_space[0]
            self.q_network = DQN(action_space, observation_space).to(self.device)
            self.target_network = DQN(action_space, observation_space).to(self.device)

        elif model_type == "DQN_CNN":
            self.q_network = DQN_CNN(action_space, observation_space).to(self.device)
            self.target_network = DQN_CNN(action_space, observation_space).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize the optimizer, loss function, and replay buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.memory = ReplayBuffer(self.memory_capacity, self.obs_dim, self.action_dim)

    def select_action(self, state, eval_mode=False):
        if not eval_mode and np.random.rand() < self.epsilon:
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
            next_q = next_q.max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones.float())

        loss = self.loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_epsilon()
        self.update_network(self.q_network, self.target_network)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))

    def save_models(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        torch.save(self.q_network.state_dict(), f'{self.path}_dqn.pth')
        torch.save(self.target_network.state_dict(), f'{self.path}_dqn_target.pth')

        config = self.save_config()
        config.update({
            'model_type': 'DQN_CNN' if isinstance(self.q_network, DQN_CNN) else 'DQN',
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'lr': self.lr,
            'memory_capacity': self.memory_capacity,
            'action_space': self.action_dim,
            'observation_space': self.obs_dim
        })
        torch.save(config, f'{self.path}_config.pth')

    def load_models(self):
        self.q_network.load_state_dict(torch.load(f'{self.path}_dqn.pth'))
        self.target_network.load_state_dict(torch.load(f'{self.path}_dqn_target.pth'))

    def eval_mode(self):
        self.q_network.eval()
        self.target_network.eval()

    def train_mode(self):
        self.q_network.train()
        self.target_network.train()

    def print_parameters(self):
        print(f"Agent Parameters:")
        print(f"  Device: {self.device}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Epsilon Start: {self.epsilon_start}")
        print(f"  Epsilon Min: {self.epsilon_min}")
        print(f"  Epsilon Decay: {self.epsilon_decay}")
        print(f"  Tau: {self.tau}")
        print(f"  Learning Rate: {self.lr}")
        print(f"  Memory Capacity: {self.memory_capacity}")
        print(f"  Action Space: {self.action_dim}")
        print(f"  Observation Space: {self.obs_dim}")
        print("-" * 30)

    @staticmethod
    def load_agent(name, device, eval_mode=True):
        """Load agent from saved configuration"""
        path = f'models/{name}'
        config = torch.load(f'{path}_config.pth', weights_only=False)
        
        agent = Agent(
            name=config['name'],
            device=device,
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            tau=config['tau'],
            epsilon_start=config['epsilon_start'],
            epsilon_min=config['epsilon_min'],
            epsilon_decay=config['epsilon_decay'],
            lr=config['lr'],
            memory_capacity=config['memory_capacity'],
            action_space=config['action_space'],
            observation_space=config['observation_space'],
            model_type=config['model_type']
        )
        
        agent.load_models()
        if eval_mode:
            agent.eval_mode()
        
        return agent