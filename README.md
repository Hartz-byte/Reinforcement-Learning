[![RL](https://img.shields.io/badge/Reinforcement%20Learning-DQN-blueviolet)](#)
[![RL Agent](https://img.shields.io/badge/RL-Agent--based-brightgreen)](#)
[![DQN](https://img.shields.io/badge/Algorithm-DQN-orange)](#)
[![Environment: Gym](https://img.shields.io/badge/Gym-LunarLander--v2-orange)](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)


# Lunar Lander Simulation (DQN)

This project implements a Deep Q-Network (DQN) agent to solve the `LunarLander-v2` environment from OpenAI Gym. The agent learns to control a lunar lander to achieve a smooth landing between designated flags using reinforcement learning techniques.

The project demonstrates:
- Experience replay
- Epsilon-greedy exploration strategy
- Target network and soft updates
- Gradient clipping
- Reward clipping
- Smooth loss functions (Huber Loss)

---

## Environment

The [LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment is a classic control problem where the goal is to land a spaceship softly on the landing pad. It is part of the Box2D environments in OpenAI Gym.

- **State space (8 values)**: Position, velocity, angle, and leg contact flags.
- **Action space (4 actions)**: 
  - 0: Do nothing  
  - 1: Fire left orientation engine  
  - 2: Fire main engine  
  - 3: Fire the right orientation engine
 
---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- TQDM
- Gym (classic and Box2D environments)

Install dependencies:
```bash
pip install torch gym[box2d] numpy matplotlib tqdm
```

---

## Algorithm
The agent uses Deep Q-Learning with the following features:

## Q-Network
A feedforward neural network with:
- 2 hidden layers of 64 units each
- ReLU activation
- Outputs Q-values for each of the 4 actions

Replay Buffer
- Stores past experiences (state, action, reward, next_state, done)
- Random sampling to break the correlation between sequential experiences

Epsilon-Greedy Strategy
- Starts with high exploration (epsilon = 1.0)
- Decays gradually (eps_decay = 0.999) toward eps_end = 0.01

Training Details
- Loss Function: Huber Loss (SmoothL1Loss)
- Optimizer: Adam
- Discount factor (gamma): 0.99
- Soft update of target network with factor (tau = 1e-3)
- Gradient clipping to prevent exploding gradients

---

| Hyperparameter     | Value   |
| ------------------ | ------- |
| Episodes           | 2400    |
| Batch Size         | 64      |
| Learning Rate      | 1e-4    |
| Gamma              | 0.99    |
| Epsilon Start      | 1.0     |
| Epsilon End        | 0.01    |
| Epsilon Decay      | 0.999   |
| Replay Buffer Size | 100,000 |
| Hidden Units       | 64      |

---

## Output:
```bash
Episode 2400, Avg Reward: 129.88, Epsilon: 0.09

Episode 1 Reward: 240.10743291296916
Episode 2 Reward: 246.03374685696483
Episode 3 Reward: 120.2279850943359
Episode 4 Reward: 281.4993702231069
Episode 5 Reward: 147.37552667774162
```

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---
