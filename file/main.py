from agent.ddpg import Agent as DDPGAgent
from gymnasium.wrappers import RecordVideo, NumpyToTorch

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    testMountainCarWithDDPG(device)


def testMountainCarWithDDPG(device: torch.device):
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
    env = RecordVideo(
        env,
        video_folder='videos/ddpg/',
        episode_trigger=lambda episode_id: episode_id % 50 == 0,
        name_prefix="mountain-car"
    )
    
    env = NumpyToTorch(env, device)

    print(env.action_space.shape)
    print(env.observation_space.shape)

    agent = DDPGAgent(
        name='DDPG-CartPole',
        device=device,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        lr_actor=2.5e-3,
        lr_critic=1e-4,
        memory_capacity=100000,
        action_space=env.action_space.shape,
        observation_space=env.observation_space.shape
    )

    n_episodes = 200
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            action = action[0].detach()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated:
                print("GOAL REACHED at episode", episode)

            agent.store_transition(state, action, next_state, reward, done)
            agent.train()
            state = next_state
            total_reward += reward

        if episode % 5 == 0:
            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")


if __name__ == "__main__":
    main()
