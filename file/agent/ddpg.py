from ddpg.actor import Actor
from ddpg.critic import Critic
from interface.agent import AgentAbstract
from utils.ornstein_uhlenbeck_action_noise import OrnsteinUhlenbeckActionNoise
from utils.replay_buffer import ReplayBuffer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Agent(AgentAbstract):
    def __init__(
        self,
        name: str,
        device: str,
        batch_size: int,
        gamma: float,
        tau: float,
        lr_actor: float,
        lr_critic: float,
        memory_capacity: int,
        action_space: int,
        observation_space: int
    ):
        super().__init__(name, device, batch_size, gamma, tau)
        self.action_dim = action_space
        self.obs_dim = observation_space
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        if isinstance(observation_space, tuple):
            observation_space = observation_space[0]
        if isinstance(action_space, tuple):
            action_space = action_space[0]

        self.actor = Actor(action_space, observation_space).to(self.device)
        self.actor_target = Actor(action_space, observation_space).to(self.device)
        self.critic = Critic(action_space, observation_space).to(self.device)
        self.critic_target = Critic(action_space, observation_space).to(self.device)

        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space))

        # Initialize the optimizer and loss function
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.loss = nn.MSELoss()
        self.memory = ReplayBuffer(memory_capacity, self.obs_dim, self.action_dim, action_type='continuous')


    def select_action(self, state):
        self.actor.eval()
        state = state.unsqueeze(0).to(self.device)
        # state = state.to(self.device)
        action = self.actor(state)
        action += torch.tensor(self.action_noise.sample(), dtype=torch.float32).to(self.device)

        action = torch.clamp(action, -1.0, 1.0)
        self.actor.train()
        return action
    
    def store_transition(self, state, action, next_state, reward, done):
        self.memory.store(state, action, next_state, reward, done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size, self.device)
        # print(states.shape, actions.shape, next_states.shape, rewards.shape, dones.shape)
        # Target networks should always be in eval mode
        self.actor_target.eval()
        self.critic_target.eval()

        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            # print(target_actions.shape)
            target_q = self.critic_target(next_states, target_actions)
            # target_q[dones] = 0.0 # Set Q value to 0 for terminal states
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1).float())

        # Critic update
        self.critic.train() # Switch to train mode for critic update
        # print(states.shape, actions.unsqueeze(1).shape)
        current_q = self.critic(states, actions) 
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

    def save_models(self):
        torch.save(self.actor.state_dict(), f'{self.path}_ddpg_actor.pth')
        torch.save(self.actor_target.state_dict(), f'{self.path}_ddpg_target_actor.pth')
        torch.save(self.critic.state_dict(), f'{self.path}_ddpg_critic.pth')
        torch.save(self.critic_target.state_dict(), f'{self.path}_ddpg_target_critic.pth')

    def load_models(self):
        self.actor.load_state_dict(torch.load(f'{self.path}_ddpg_actor.pth'))
        self.actor_target.load_state_dict(torch.load(f'{self.path}_ddpg_target_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{self.path}_ddpg_critic.pth'))
        self.critic_target.load_state_dict(torch.load(f'{self.path}_ddpg_target_critic.pth'))

    def eval_mode(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train_mode(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()
