from ddpg.ddpg import Agent as DDPGAgent
from dqn.dqn import Agent as DQNAgent
from utils.reward_logger import RewardLogger
from gymnasium.wrappers import RecordVideo, NumpyToTorch

import gymnasium as gym
import numpy as np
import torch

DQN_FIGURES_DIR = "output/figures/dqn/"
DDPG_FIGURES_DIR = "output/figures/ddpg/"
DQN_VIDEOS_DIR = "output/videos/dqn/"
DDPG_VIDEOS_DIR = "output/videos/ddpg/"

EPISODES_INTERVAL_OF_PRINT = 10
EPISODES_INTERVAL_OF_VIDEO = 50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    testCartPole(device, n_episodes=200)
    testMountainCar(device, n_episodes=200)


def testCartPole(device: torch.device, n_episodes: int = 200):
    testEnvironmentWithDQN("CartPole-v1", device, n_episodes=n_episodes)

def testMountainCar(device: torch.device, n_episodes: int = 200):
    testEnvironmentWithDQN("MountainCar-v0", device, n_episodes=n_episodes)
    testEnvironmentWithDDPG("MountainCarContinuous-v0", device, n_episodes=n_episodes)


def testEnvironmentWithDQN(env_name: str, device: torch.device, n_episodes: int = 200, model_type: str = "DQN"):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=DQN_VIDEOS_DIR,
        episode_trigger=lambda episode_id: episode_id % EPISODES_INTERVAL_OF_VIDEO == 0,
        name_prefix=env_name
    )
    
    env = NumpyToTorch(env, device)

    agent = DQNAgent(
        name=f"DQN-{env_name}",
        device=device,
        batch_size=64,
        gamma=0.99,
        epsilon_start=0.95,
        epsilon_min=0.01,
        epsilon_decay=1e-4,
        tau=0.005,
        lr=1e-3,
        memory_capacity=10000,
        action_space=env.action_space.n,
        observation_space=env.observation_space.shape,
        model_type=model_type
    )

    reward_logger = RewardLogger()
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, next_state, reward, done)
            agent.train()
            state = next_state
            total_reward += reward

        reward_logger.add(total_reward)

        if episode % EPISODES_INTERVAL_OF_PRINT == 0:
            print(f"Episode {episode + 1}/{n_episodes}, Average Reward: {np.mean(reward_logger.get_rewards()[-EPISODES_INTERVAL_OF_PRINT:]):.2f}, Epsilon: {agent.epsilon:.2f}")

    plot_reward_name = f"{model_type}-{env_name}-Reward"
    reward_logger.plot_rewards(f"{DQN_FIGURES_DIR}{plot_reward_name}.png")
    reward_logger.plot_rewards_smooth(f"{DQN_FIGURES_DIR}{plot_reward_name}Smooth.png")


def testEnvironmentWithDDPG(env_name: str, device: torch.device, n_episodes: int = 200, model_type: str = "DDPG"):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=DDPG_VIDEOS_DIR,
        episode_trigger=lambda episode_id: episode_id % EPISODES_INTERVAL_OF_VIDEO == 0,
        name_prefix=env_name
    )
    
    env = NumpyToTorch(env, device)

    agent = DDPGAgent(
        name=f"DDPG-{env_name}",
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

    reward_logger = RewardLogger()
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            action = action[0].detach()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, next_state, reward, done)
            agent.train()
            state = next_state
            total_reward += reward

        reward_logger.add(total_reward)

        if episode % EPISODES_INTERVAL_OF_PRINT == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(reward_logger.get_rewards()[-EPISODES_INTERVAL_OF_PRINT:]):.2f}")


    plot_reward_name = f"{model_type}-{env_name}-Reward"
    reward_logger.plot_rewards(f"{DDPG_FIGURES_DIR}{plot_reward_name}.png")
    reward_logger.plot_rewards_smooth(f"{DDPG_FIGURES_DIR}{plot_reward_name}Smooth.png")


if __name__ == "__main__":
    main()
