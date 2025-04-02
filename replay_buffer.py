import numpy as np

# Create the Replay Buffer
class ReplayBuffer(object):
    def __init__(
            self,
            capacity: int,
            input_shape: tuple,
            action_space: int
    ):
        self.size = capacity
        self.states = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.next_states = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=bool)

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

    def sample(self, batch_size):
        indices = np.random.choice(self.current_size, batch_size, replace=False)

        states = self.states[indices]
        actions = self.actions[indices]
        next_states = self.next_states[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return self.current_size