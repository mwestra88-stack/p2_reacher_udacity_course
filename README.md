# Continuous Control with DDPG – Unity Reacher Environment

This repository contains my solution to the **Continuous Control** project from the Udacity Deep Reinforcement Learning Nanodegree. The agent is trained using **Deep Deterministic Policy Gradient (DDPG)** to solve the Unity **Reacher (1 agent)** environment. The environment was successfully solved in **375 episodes**, achieving an average score of **30.15 over 100 consecutive episodes**.

---

## 1. Project Details

### Environment: Unity Reacher

The environment is based on the Unity ML-Agents **Reacher** task.

- A double-jointed arm moves in a 3D environment.
- The environment solved consist of **1 agent** (note there is also a version containing 20 agents)
- The goal is to keep the end effector inside a moving target location.
- A reward of **+0.1** is provided for each timestep the agent's hand is in the goal location.
- Episodes last up to **1000 timesteps**.

### State Space

- The state space has **33 dimensions**.
- It contains:
  - Position
  - Rotation
  - Velocity
  - Angular velocities of the arm

### Action Space

- The action space has **4 dimensions**.
- Each action corresponds to torque applied to two joints.
- Actions are continuous in the range **[-1, 1]**.

### Solving Criteria

The environment is considered solved when:

> The agent achieves an average score of **+30 over 100 consecutive episodes**.

This implementation achieved:

- **Average Score: 30.15**
- **Episodes to Solve: 375**

---


## 2. Installation & Environment Setup
This project uses Python 3.6 and the legacy `unityagents` package (v0.4), as required by the original Udacity Banana environment. First, I have cloned the github repository:

```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

### 2.1 Create a Python 3.6 Environment
Using conda (recommended):
```bash
py -3.6 -m venv unityagents36
unityagents36\Scripts\activate
```

### 2.2 Install the required packages in this environment

```bash
pip install numpy==1.18.5
pip install tensorflow==1.7.1
pip install unityagents==0.4.0
pip install torch
pip install numpy
pip install matplotlib
pip install jupyter
```

Note that newer versions of Python and ML-Agents are not compatible with this legacy environment.

### 2.3 Register Jupyter kernel
To use this environment inside Jupyter Notebook:
```bash
python -m ipykernel install --user --name unityagents36 --display-name "Python 3.6 (unityagents)"
```
Then select **Python 3.6 (unityagents)** as the kernel in Jupyter.

### 2.4 Download the unity environment
For this project, you will not need to install Unity - this is because Udacity has already built the environment for you, and you can download it from the link below (I used windows, it is also available for other operating systems)

* Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip
  
Then, place the file in the `p2_continuous_control/` folder in the course GitHub repository, and unzip (or decompress) the file.

## 3. Instructions to Train the Agent

The training logic is implemented using:

* `ActorNetwork`
* `CriticNetwork`
* `DDPG_Agent`
* `ReplayBuffer`
* `RunningNorm` (state normalization)
* `ddpg()` training loop

### 3.1 Initialize Environment

In the notebook:
```python
from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="Reacher.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

state_size = len(env_info.vector_observations[0])
action_size = brain.vector_action_space_size
```

### 3.2 Initialize Agent
```python
from ddpg_agent import DDPG_Agent

agent = DDPG_Agent(state_size=state_size,
                   action_size=action_size,
                   seed=1234)
```

### 3.3 Train the Agent
```python
scores = ddpg(
    n_episodes=1000,
    max_t=1000,
    noise_scale_start=1.0,
    noise_scale_end=0.1,
    noise_decay=0.999
)
```

The environment will stop automatically once the average score over 100 episodes reaches 30. When solved, the following files are saved:

``
checkpoint_actor.pth
checkpoint_critic.pth
``

## 4. Hyperparameters
```python
BUFFER_SIZE = 1e5          # Replay buffer size
BATCH_SIZE = 256           # Sample size of batch taken from replay buffer
GAMMA = 0.99               # Discount factor
TAU = 1e-3                 # For soft update of target parameters
LRA = 2e-4                 # Learning rate of actor
LRC = 5e-4                 # Learning rate of critic
UPDATE_EVERY = 10          # Update actor and critic network parameters every 10 time steps
HIDDEN_ACTOR = [400, 300]  # Size and dimension of hidden layers of actor network
HIDDEN_CRITIC = [400, 300] # Size and dimension of hidden layers of critic network
NOISE_SIGMA = 0.2          # Standard deviation of random standard normal gaussian noise added to actions for exploration purposes 
```

Additional model parameters/assumptions:
* 10 learning updates every 10 environment steps
* Replay buffer warm-up of 10,000 experiences
* Gradient clipping on critic
* Running state normalization

## 5. State normalisation
To stabilize training, a running mean and variance are maintained:

* Raw states are stored in replay buffer
* Running statistics are updated every step
* States are normalized before being fed to networks
* This improved learning stability and speed.

## 6. Results
* Training Episodes: 375
* Final 100-Episode Average: 30.15
* Environment Successfully Solved

The learning curve shows gradual improvement followed by convergence beyond the required threshold. Nevertheless, the scores development does appear volatile. There is quite some room for improvement.

## 7. Future Improvements

Possible extensions:
* Implement TD3 for improved stability
* Better hyperparameter tuning
* Prioritized Experience Replay
* SAC model
* Multi-agent version of the environment

## 8. Acknowledgements

This project is part of the Udacity Deep Reinforcement Learning Nanodegree Program. Unity ML-Agents and the Reacher environment are provided by Udacity.
