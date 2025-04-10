import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(
            self,
            capacity: int,
            input_shape: tuple,
            action_space: int,
            action_type: str = 'discrete'
    ):
        self.size = capacity
        self.states = torch.zeros((self.size, *input_shape), dtype=torch.float32)

        if action_type == 'discrete':
            self.actions = torch.zeros(self.size, dtype=torch.int64)
        elif action_type == 'continuous':
            self.actions = torch.zeros((self.size, *action_space), dtype=torch.float32)

        self.next_states = torch.zeros((self.size, *input_shape), dtype=torch.float32)
        self.rewards = torch.zeros(self.size, dtype=torch.float32)
        self.dones = torch.zeros(self.size, dtype=torch.bool)

        # Circular buffer
        self.ptr = 0
        self.current_size = 0

    def store(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        # Update the pointer
        self.ptr = (self.ptr + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size, device=None):
        indices = np.random.choice(self.current_size, batch_size, replace=False)

        states = self.states[indices]
        actions = self.actions[indices]
        next_states = self.next_states[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        if device is not None:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return self.current_size
