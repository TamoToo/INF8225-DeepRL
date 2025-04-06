from matplotlib import pyplot as plt
import numpy as np
import torch

class PlotResults:
    def __init__(
        self,
    ):
        self.rewards = []

    def add(self, reward):
        self.rewards.append(reward)

    def get_rewards(self):
        return self.rewards

    def plot_rewards(self, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.plot(self.rewards)
        plt.title("Rewards over episodes")
        plt.xlabel("Episode #")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.show()
    
    def plot_rewards_smooth(self, window_length=50, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        moving_average = np.convolve(self.rewards, np.ones(window_length) / window_length, mode="valid")
        plt.plot(moving_average)
        plt.title(f'Running Average Reward ({window_length} episodes)')
        plt.xlabel("Episode #")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.show()


class OrnsteinUhlenbeckActionNoise(object):
    def __init__(
            self, 
            # action_dim,
            mu = 0,
            theta = 0.15,
            sigma = 0.2,
            dt = 1e-2,
            x0 = None
    ):
        # self.action_dim = action_dim
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


def show_epsilon_decay(epsilon_start, epsilon_end, epsilon_decay):
    epsilons = [epsilon_start]
    epsilon = epsilon_start
    while epsilons[-1] > epsilon_end:
        epsilon = max(epsilon_end, epsilons[-1] * (1 - epsilon_decay))
        epsilons.append(epsilon)
        # print(epsilon)

    # print(len(epsilons))
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    plt.show()


def preprocess_state(state):
    """Preprocess the state for the CNN model"""
    # Input shape: (4, 96, 96, 3) - 4 stacked RGB frames

    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)

    n_frames, height, width, channels = state.shape
    
    grayscale_frames = torch.zeros((n_frames, height, width), dtype=torch.float32)
    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
    for i in range(n_frames):
        # Dot product along the RGB channels
        grayscale_frames[i] = torch.sum(state[i, :, :, :3] * rgb_weights, dim=2)
    
    # Transpose from (4, 96, 96) to (96, 96, 4)
    transposed = grayscale_frames.permute(1, 2, 0)
    normalized = transposed / 255.0
    return normalized


def show_frame(state):
    """Display the first frame of the state"""
    plt.figure(figsize=(5, 5))
    
    # Check if state is a stacked frame
    if len(state.shape) == 3 and state.shape[2] > 1:
        # Display first frame from stack
        plt.imshow(state[:, :, 0], cmap='gray')
    else:
        # Display single frame
        plt.imshow(state, cmap='gray')

    plt.axis('off')
    plt.show()