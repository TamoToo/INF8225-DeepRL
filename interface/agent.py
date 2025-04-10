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
    
    def train(self):
        raise NotImplementedError
    
    def save_models(self):
        raise NotImplementedError
    
    def load_models(self):
        raise NotImplementedError
