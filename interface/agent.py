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

    def select_action(self, state):
        raise NotImplementedError
    
    def store_transition(self, state, action, next_state, reward, done):
        raise NotImplementedError
    
    def update_network(self, network, target_network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self):
        raise NotImplementedError
    
    def save_config(self):
        config = {
            'name': self.path.split('/')[-1],
            'device': self.device,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau
        }
        return config
    
    def save_models(self):
        raise NotImplementedError
    
    def load_models(self):
        raise NotImplementedError