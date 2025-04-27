from ddpg.ddpg import Agent as DDPGAgent
from dqn.dqn import Agent as DQNAgent
from utils.reward_logger import RewardLogger
from utils.utils import preprocess_state
from gymnasium.wrappers import RecordVideo, NumpyToTorch, FrameStackObservation

import gymnasium as gym
import numpy as np
import torch

import os

COMPARE_FIGURES_DIR = "output/figures/compare/"
DQN_FIGURES_DIR = "output/figures/dqn/"
DDPG_FIGURES_DIR = "output/figures/ddpg/"
DQN_VIDEOS_DIR = "output/videos/dqn/"
DDPG_VIDEOS_DIR = "output/videos/ddpg/"

EPISODES_INTERVAL_OF_PRINT = 10
EPISODES_INTERVAL_OF_VIDEO = 50

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    testCartPole(device, max_steps=10000)
    testMountainCar(device, max_steps=250000)
    testLunarLander(device, max_steps=1000000)
    testRacingCar(device, max_steps=250000)


def testCartPole(device: torch.device, max_steps: int = 100000):
    testEnvironmentWithDQN("CartPole-v1", device, max_steps=max_steps)

def testMountainCar(device: torch.device, max_steps: int = 100000):
    rewards_dqn = testEnvironmentWithDQN("MountainCar-v0", device, max_steps=max_steps)
    rewards_ddpg = testEnvironmentWithDDPG("MountainCarContinuous-v0", device, max_steps=max_steps)
    
    os.makedirs(COMPARE_FIGURES_DIR, exist_ok=True)
    RewardLogger().plot_multiple_rewards_smooth(
        [rewards_dqn, rewards_ddpg],
        file_name=f"{COMPARE_FIGURES_DIR}MountainCar.png",
        window_length=50,
        figsize=(10, 5)
    )

def testLunarLander(device: torch.device, max_steps: int = 100000):
    rewards_dqn = testEnvironmentWithDQN("LunarLander-v3", device, max_steps=max_steps, continuous=False)
    rewards_ddpg = testEnvironmentWithDDPG("LunarLander-v3", device, max_steps=max_steps, continuous=True)

    os.makedirs(COMPARE_FIGURES_DIR, exist_ok=True)
    RewardLogger().plot_multiple_rewards_smooth(
        [rewards_dqn, rewards_ddpg],
        file_name=f"{COMPARE_FIGURES_DIR}LunarLander.png",
        window_length=50,
        figsize=(10, 5)
    )

def testRacingCar(device: torch.device, max_steps: int = 100000):
    rewards_dqn = testEnvironmentWithDQN("CarRacing-v3", device, max_steps=max_steps, model_type="DQN_CNN", start_skip=15, continuous=False)
    rewards_ddpg = testEnvironmentWithDDPG("CarRacing-v3", device, max_steps=max_steps, model_type="DDPG_CNN", start_skip=15, continuous=True)
    
    os.makedirs(COMPARE_FIGURES_DIR, exist_ok=True)
    RewardLogger().plot_multiple_rewards_smooth(
        [rewards_dqn, rewards_ddpg],
        file_name=f"{COMPARE_FIGURES_DIR}CarRacing.png",
        window_length=50,
        figsize=(10, 5)
    )

def testEnvironmentWithDQN(env_name: str, device: torch.device, max_steps: int = 100000, model_type: str = "DQN", start_skip: int = 0, **kwargs):
    env = gym.make(env_name, render_mode="rgb_array", **kwargs)
    env = RecordVideo(
        env,
        video_folder=DQN_VIDEOS_DIR + env_name,
        episode_trigger=lambda episode_id: episode_id % EPISODES_INTERVAL_OF_VIDEO == 0,
        name_prefix=env_name
    )

    if "CNN" in model_type:
        env = FrameStackObservation(env, 4) # Stack 4 frames for CNN input
    
    n_actions, n_observations = env.action_space.n, env.observation_space.shape

    if "CNN" in model_type:
        n_observations = (n_observations[1], n_observations[2], n_observations[0]) # Change the order of dimensions for CNN input (height, width, frames)
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
        lr=1e-4,
        memory_capacity=10000,
        action_space=n_actions,
        observation_space=n_observations,
        model_type=model_type
    )

    reward_logger = RewardLogger()
    total_steps = 0
    episode = 0

    while total_steps < max_steps:
        state, _ = env.reset()

        for _ in range(start_skip):
            # Take random actions during the skip phase
            random_action = np.random.randint(n_actions)
            state, _, _, _, _ = env.step(random_action)

        if "CNN" in model_type:
            state = preprocess_state(state, device)

        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            if total_steps >= max_steps:
                break
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if "CNN" in model_type:
                next_state = preprocess_state(next_state, device)

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
    
    return reward_logger.rewards

def testEnvironmentWithDDPG(env_name: str, device: torch.device, max_steps: int = 100000, model_type: str = "DDPG", start_skip: int = 0, **kwargs):
    env = gym.make(env_name, render_mode="rgb_array", **kwargs)
    env = RecordVideo(
        env,
        video_folder=DDPG_VIDEOS_DIR + env_name,
        episode_trigger=lambda episode_id: episode_id % EPISODES_INTERVAL_OF_VIDEO == 0,
        name_prefix=env_name
    )

    if "CNN" in model_type:
        env = FrameStackObservation(env, 4) # Stack 4 frames for CNN input
    
    n_actions, n_observations = env.action_space.shape, env.observation_space.shape

    if "CNN" in model_type:
        n_observations = (n_observations[1], n_observations[2], n_observations[0]) # Change the order of dimensions for CNN input (height, width, frames)
    env = NumpyToTorch(env, device)

    agent = DDPGAgent(
        name=f"DDPG-{env_name}",
        device=device,
        batch_size=256,
        gamma=0.99,
        tau=0.001,
        lr_actor=1e-5,
        lr_critic=1e-5,
        memory_capacity=100000,
        action_space=n_actions,
        observation_space=n_observations,
        model_type=model_type
    )

    reward_logger = RewardLogger()
    total_steps = 0
    episode = 0

    while total_steps < max_steps:
        state, _ = env.reset()
        agent.action_noise.reset()  # Reset action noise for each episode

        for _ in range(start_skip):
            # Take random actions during the skip phase
            random_action = torch.tensor(np.random.uniform(0, 1, n_actions), dtype=torch.float32).to(device) # Random action for car racing env
            state, _, _, _, _ = env.step(random_action) 

        if "CNN" in model_type:
            state = preprocess_state(state, device)

        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            if total_steps >= max_steps:
                break
            raw_action = agent.select_action(state)
            action_to_env = raw_action.clone()
            if "CarRacing" in env_name:
                # Make sure the action is in the correct range (0, 1) for gas and brake in car racing env
                action_to_env[1] = (action_to_env[1] + 1.0) / 2.0  # Rescale gas
                action_to_env[2] = (action_to_env[2] + 1.0) / 2.0  # Rescale brake

            next_state, reward, terminated, truncated, _ = env.step(action_to_env)
            done = terminated or truncated

            if "CNN" in model_type:
                next_state = preprocess_state(next_state, device)

            agent.store_transition(state, raw_action, next_state, reward, done)
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

    return reward_logger.rewards


if __name__ == "__main__":
    main()