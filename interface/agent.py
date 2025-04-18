from abc import abstractmethod
import torch

class AgentAbstract(object):
    def __init__(
            self,
            name: str,
            device: str,
            batch_size: int,
            gamma: float,
            tau: float,
    ):
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.path = f'models/{name}'

    @abstractmethod
    def select_action(self, state):
        """Selects an action based on the current state."""
        pass
    
    @abstractmethod
    def store_transition(self, state, action, next_state, reward, done):
        """Stores a transition in the agent's memory."""
        pass
    
    def update_network(self, network: torch.nn.Module, target_network: torch.nn.Module):
        """Performs a soft update of the target network parameters using tau parameter."""
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    @abstractmethod
    def train(self):
        """Trains the agent's models."""
        pass
    
    def save_config(self) -> dict:
        """Saves the agent's configuration parameters."""
        config = {
            'name': self.path.split('/')[-1],
            'device': self.device,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau
        }
        return config
    
    @abstractmethod
    def save_models(self):
        """Saves the agent's model weights."""
        pass
    
    @abstractmethod
    def load_models(self):
        """Loads the agent's model weights."""
        pass