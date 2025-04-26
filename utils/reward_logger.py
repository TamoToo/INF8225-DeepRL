import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class RewardLogger(object):
    def __init__(self):
        self.rewards = []


    def add(self, reward):
        self.rewards.append(reward)

    def get_rewards(self):
        return self.rewards

    def plot_rewards(self, file_name, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.plot(self.rewards)
        plt.title("Rewards over episodes")
        plt.xlabel("Episode #")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(file_name)
    
    def plot_rewards_smooth(self, file_name, window_length=50, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        moving_average = np.convolve(self.rewards, np.ones(window_length) / window_length, mode="valid")
        plt.plot(moving_average)
        plt.title(f'Running Average Reward ({window_length} episodes)')
        plt.xlabel("Episode #")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(file_name)

    def plot_multiple_rewards_smooth(self, rewards_lists, file_name, window_length=50, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        for rewards in rewards_lists:
            moving_average = np.convolve(rewards, np.ones(window_length) / window_length, mode="valid")
            plt.plot(moving_average)
        plt.title(f'Running Average Reward ({window_length} episodes)')
        plt.xlabel("Episode #")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(file_name)
