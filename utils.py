from matplotlib import pyplot as plt
import numpy as np

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
    
    # Convert each frame to grayscale
    # Apply RGB->grayscale conversion to the RGB channel (last dimension)
    grayscale_frames = np.zeros((state.shape[0], state.shape[1], state.shape[2]))
    for i in range(state.shape[0]):
        grayscale_frames[i] = np.dot(state[i, :, :, :3], [0.2989, 0.5870, 0.1140])
    
    # Transpose from (4, 96, 96) to (96, 96, 4)
    transposed = np.transpose(grayscale_frames, (1, 2, 0))
    
    normalized = transposed / 255.0
    
    return normalized.astype(np.float32)


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