from ddpg.actor import Actor, ActorCNN
from ddpg.critic import Critic, CriticCNN
from interface.agent import AgentAbstract
from utils.ornstein_uhlenbeck_action_noise import OrnsteinUhlenbeckActionNoise
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
        tau: float,
        lr_actor: float,
        lr_critic: float,
        memory_capacity: int,
        action_space: int,
        observation_space: int,
        model_type: str = "DDPG"
    ):
        super().__init__(name, device, batch_size, gamma, tau)
        self.action_dim = action_space
        self.obs_dim = observation_space
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.memory_capacity = memory_capacity

        if isinstance(action_space, tuple):
            action_space = action_space[0]

        if model_type == "DDPG":
            if isinstance(observation_space, tuple):
                observation_space = observation_space[0]

            self.actor = Actor(action_space, observation_space).to(self.device)
            self.actor_target = Actor(action_space, observation_space).to(self.device)
            self.critic = Critic(action_space, observation_space).to(self.device)
            self.critic_target = Critic(action_space, observation_space).to(self.device)

        elif model_type == "DDPG_CNN":
            self.actor = ActorCNN(action_space, observation_space).to(self.device)
            self.actor_target = ActorCNN(action_space, observation_space).to(self.device)
            self.critic = CriticCNN(action_space, observation_space).to(self.device)
            self.critic_target = CriticCNN(action_space, observation_space).to(self.device)


        # FORGOT THIS, MIGHT BE THE PROBLEM: Initialize the target networks with the same weights as the main networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_space))
        # Initialize the optimizer and loss function
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.loss = nn.MSELoss()
        self.memory = ReplayBuffer(self.memory_capacity, self.obs_dim, self.action_dim, action_type='continuous')


    def select_action(self, state):
        self.actor.eval()
        state = state.unsqueeze(0).to(self.device)
        # state = state.to(self.device)
        action = self.actor(state)
        action += torch.tensor(self.action_noise.sample(), dtype=torch.float32).to(self.device)

        action = torch.clamp(action, -1.0, 1.0)
        self.actor.train()
        return action.squeeze(0).cpu().detach()
    
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
            target_q = self.critic_target(next_states, target_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1).float())

        # Critic update
        self.critic.train() # Switch to train mode for critic update
        current_q = self.critic(states, actions) 
        critic_loss = self.loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1) # Gradient clipping
        self.critic_optimizer.step()

        # Actor update
        self.actor.train() # Switch to train mode for actor update
        self.critic.eval() # Ensure critic is in eval mode for actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1) # Gradient clipping
        self.actor_optimizer.step()

        # Soft update of the target networks
        self.update_network(self.actor, self.actor_target)
        self.update_network(self.critic, self.critic_target)

    def save_models(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        torch.save(self.actor.state_dict(), f'{self.path}_ddpg_actor.pth')
        torch.save(self.actor_target.state_dict(), f'{self.path}_ddpg_target_actor.pth')
        torch.save(self.critic.state_dict(), f'{self.path}_ddpg_critic.pth')
        torch.save(self.critic_target.state_dict(), f'{self.path}_ddpg_target_critic.pth')

        config = self.save_config()
        config.update({
            'model_type': 'DDPG',
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'memory_capacity': self.memory_capacity,
            'action_space': self.action_dim,
            'observation_space': self.obs_dim
        })
        torch.save(config, f'{self.path}_config.pth')

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

    def print_parameters(self):
        """Prints the hyperparameters of the agent."""
        print("Agent Parameters:")
        print(f"  Device: {self.device}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Gamma (Discount Factor): {self.gamma}")
        print(f"  Tau (Soft Update): {self.tau}")
        print(f"  Actor Learning Rate: {self.lr_actor}")
        print(f"  Critic Learning Rate: {self.lr_critic}")
        print(f"  Memory Capacity: {self.memory_capacity}")
        print(f"  Action Space: {self.action_dim}")
        print(f"  Observation Space: {self.obs_dim}")
        print("-" * 30)

    
    @staticmethod
    def load_agent(name, device, eval_mode=True):
        """Load agent from saved configuration"""
        path = f'models/{name}'
        config = torch.load(f'{path}_config.pth')
        
        agent = Agent(
            model_type=config['model_type'],
            name=config['name'],
            device=device,
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            tau=config['tau'],
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            memory_capacity=config['memory_capacity'],
            action_space=config['action_space'],
            observation_space=config['observation_space']
        )
        
        agent.load_models()
        if eval_mode:
            agent.eval_mode()
        
        return agent