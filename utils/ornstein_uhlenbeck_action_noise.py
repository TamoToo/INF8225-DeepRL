from interface.action_noise import ActionNoise
import numpy as np

class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, theta=.15, sigma=.2, dt = 1e-2, x0 = None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0

        self.reset()


    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x
