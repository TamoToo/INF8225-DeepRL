import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN, DQN_CNN
from ddpg import DDPGActor, DDPGCritic
from utils import OrnsteinUhlenbeckActionNoise
from replay_buffer import ReplayBuffer


# Create the DQN Agent
class Agent:
    def __init__(
            self,
            device: str,
            batch_size: int,
            gamma: float,
            tau: float,
            lr: float,
    ):
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

    def select_action(self, state):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError


class DQNAgent(Agent):
    def __init__(
            self,
            model: str,
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
            observation_space: int
    ):
        super().__init__(device, batch_size, gamma, tau, lr)
        self.action_dim = action_space
        self.obs_dim = observation_space
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay

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
            next_q[dones] = 0.0 # Set Q value to 0 for terminal states
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



class DDPGAgent(Agent):
    def __init__(
            self,
            model: str,
            device: str,
            batch_size: int,
            gamma: float,
            tau: float,
            lr: float,
            memory_capacity: int,
            action_space: int,
            observation_space: int
    ):
        super().__init__(device, batch_size, gamma, tau, lr)
        self.action_dim = action_space
        self.obs_dim = observation_space

        # Initialize the DDPG networks
        if model == 'DDPG':
            if isinstance(observation_space, tuple):
                observation_space = observation_space[0]
            if isinstance(action_space, tuple):
                action_space = action_space[0]
            self.actor = DDPGActor(action_space, observation_space).to(self.device)
            self.actor_target = DDPGActor(action_space, observation_space).to(self.device)
            self.critic = DDPGCritic(action_space, observation_space).to(self.device)
            self.critic_target = DDPGCritic(action_space, observation_space).to(self.device)
        # elif model == 'DDPG_CNN':
        #     self.actor = DDPGActor(action_space, observation_space).to(self.device)
        #     self.actor_target = DDPGActor(action_space, observation_space).to(self.device)
        #     self.critic = DDPGCritic(action_space, observation_space).to(self.device)
        #     self.critic_target = DDPGCritic(action_space, observation_space).to(self.device)

        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space))

        # Initialize the optimizer and loss function
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.memory = ReplayBuffer(memory_capacity, self.obs_dim, self.action_dim, action_type='continuous')


    def select_action(self, state):
        self.actor.eval()
        state = state.unsqueeze(0).to(self.device)
        # state = state.to(self.device)
        action = self.actor(state)
        action += torch.tensor(self.action_noise.sample(), dtype=torch.float32).to(self.device)

        self.actor.train()
        return action
    
    def store_transition(self, state, action, next_state, reward, done):
        self.memory.store(state, action, next_state, reward, done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size, self.device)
        
        # Target networks should always be in eval mode
        self.actor_target.eval()
        self.critic_target.eval()

        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            # print(next_states.shape, target_actions.shape)
            target_q = self.critic_target(next_states, target_actions)
            target_q[dones] = 0.0 # Set Q value to 0 for terminal states
            target_q = rewards.unsqueeze(1) + self.gamma * target_q

        # Critic update
        self.critic.train() # Switch to train mode for critic update
        # print(states.shape, actions.unsqueeze(1).shape)
        current_q = self.critic(states, actions.unsqueeze(1)) 
        critic_loss = self.loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        self.actor.train() # Switch to train mode for actor update
        self.critic.eval() # Ensure critic is in eval mode for actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of the target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)