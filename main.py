from ddpg.ddpg import Agent as DDPGAgent
from dqn.dqn import Agent as DQNAgent
from utils.reward_logger import RewardLogger
from gymnasium.wrappers import RecordVideo, NumpyToTorch

import gymnasium as gym
import numpy as np
import torch

import os

DQN_FIGURES_DIR = "output/figures/dqn/"
DDPG_FIGURES_DIR = "output/figures/ddpg/"
DQN_VIDEOS_DIR = "output/videos/dqn/"
DDPG_VIDEOS_DIR = "output/videos/ddpg/"

EPISODES_INTERVAL_OF_PRINT = 10
EPISODES_INTERVAL_OF_VIDEO = 50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # testCartPole(device, max_steps=100000)
    testMountainCar(device, max_steps=100000)
    # testLunarLander(device, max_steps=100000)


def testCartPole(device: torch.device, max_steps: int = 100000):
    testEnvironmentWithDQN("CartPole-v1", device, max_steps=max_steps)

def testMountainCar(device: torch.device, max_steps: int = 100000):
    testEnvironmentWithDQN("MountainCar-v0", device, max_steps=max_steps)
    testEnvironmentWithDDPG("MountainCarContinuous-v0", device, max_steps=max_steps)

def testLunarLander(device: torch.device, max_steps: int = 100000):
    testEnvironmentWithDQN("LunarLander-v2", device, max_steps=max_steps)
    testEnvironmentWithDDPG("LunarLanderContinuous-v2", device, max_steps=max_steps)


def testEnvironmentWithDQN(env_name: str, device: torch.device, max_steps: int = 100000, model_type: str = "DQN"):
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
        epsilon_decay=5e-5,
        tau=0.005,
        lr=1e-3,
        memory_capacity=10000,
        action_space=env.action_space.n,
        observation_space=env.observation_space.shape,
        model_type=model_type
    )

    total_steps = 0
    episode = 0

    reward_logger = RewardLogger()
    while total_steps < max_steps:
        state, _ = env.reset()

        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            if total_steps >= max_steps:
                break
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, next_state, reward, done)
            agent.train()
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

        if episode_steps > 0:
            reward_logger.add(episode_reward)
            episode += 1 # Increment completed episode count

            if episode % EPISODES_INTERVAL_OF_PRINT == 0:
                print(f"Episode {episode}, Steps: {total_steps}/{max_steps}, Average Reward: {np.mean(reward_logger.get_rewards()[-EPISODES_INTERVAL_OF_PRINT:]):.2f}, Epsilon: {agent.epsilon:.2f}")

    print(f"\nTraining finished after {total_steps} steps across {episode} episodes.")
    os.makedirs(DQN_FIGURES_DIR, exist_ok=True)
    plot_reward_name = f"{model_type}-{env_name}-Reward"
    reward_logger.plot_rewards(f"{DQN_FIGURES_DIR}{plot_reward_name}.png")
    reward_logger.plot_rewards_smooth(f"{DQN_FIGURES_DIR}{plot_reward_name}Smooth.png")

    agent.save_models()
    agent.save_config()


def testEnvironmentWithDDPG(env_name: str, device: torch.device, max_steps: int = 100000, model_type: str = "DDPG"):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=DDPG_VIDEOS_DIR,
        episode_trigger=lambda episode_id: episode_id % EPISODES_INTERVAL_OF_VIDEO == 0,
        name_prefix=env_name
    )
    env = NumpyToTorch(env, device)

    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    action_shape = env.action_space.shape if is_continuous else env.action_space.n # Handle discrete vs continuous
    obs_shape = env.observation_space.shape
    print(f"Action shape: {action_shape}, Observation shape: {obs_shape}")
    agent = DDPGAgent(
        name=f"DDPG-{env_name}",
        device=device,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        lr_actor=2.5e-3,
        lr_critic=1e-4,
        memory_capacity=100000,
        action_space=action_shape,
        observation_space=obs_shape
    )

    reward_logger = RewardLogger()
    total_steps = 0
    episode = 0

    while total_steps < max_steps:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            if total_steps >= max_steps:
                break
            action = agent.select_action(state)
            # TODO: Add a way to make sure the action is in the valid range for multiple actions dimension env
            # print(f"Action: {action}")
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, next_state, reward, done)
            agent.train()
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

        if episode_steps > 0:
            reward_logger.add(episode_reward)
            episode += 1

            if episode % EPISODES_INTERVAL_OF_PRINT == 0:
                print(f"Episode {episode}, Steps: {total_steps}/{max_steps}, Average Reward: {np.mean(reward_logger.get_rewards()[-EPISODES_INTERVAL_OF_PRINT:]):.2f}")


    print(f"\nTraining finished after {total_steps} steps across {episode} episodes.")
    os.makedirs(DDPG_FIGURES_DIR, exist_ok=True)
    plot_reward_name = f"{model_type}-{env_name}-Reward"
    reward_logger.plot_rewards(f"{DDPG_FIGURES_DIR}{plot_reward_name}.png")
    reward_logger.plot_rewards_smooth(f"{DDPG_FIGURES_DIR}{plot_reward_name}Smooth.png")

    agent.save_models()
    agent.save_config()


if __name__ == "__main__":
    main()
