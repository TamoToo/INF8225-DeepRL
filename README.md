# Project Overview
This project provides implementations of DQN and DDPG agents for gymnasium environments.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TamoToo/INF8225-DeepRL.git
   cd INF8225-DeepRL
   ```

2. **Set up a Python virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   All required Python packages are listed in `requirements.txt`. To install them, run:
   ```bash
   pip install -r requirements.txt
   ```

## Note: Using CUDA (Optional)

If you want to use CUDA, please install it on your own.

## Quick Start

<!-- 1. **Train a DQN agent**
   ```bash
   python train_dqn.py --env CartPole-v1
   ```

2. **Train a DDPG agent**
   ```bash
   python train_ddpg.py --env Pendulum-v0
   ```

3. **Evaluate a trained model**
   ```bash
   python evaluate.py --model_path runs/dqn_cartpole/model.pt
   ``` -->

## Project Structure

- `dqn/`: implementations DQN agent and network
- `ddpg/`: implementations DDPG agent and networks
- `utils/`: implementations of the replay buffer, OU noise...
- `models/`: neural network architectures
- `output/`: videos and figures from training

<!-- ## License -->
