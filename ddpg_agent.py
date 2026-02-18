import numpy as np
import random
from collections import namedtuple, deque

from critic_model import CriticNetwork
from actor_model import ActorNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LRA = 2e-4             # learning rate actor
LRC = 5e-4             # learning rate critic
UPDATE_EVERY = 10        # how often to update the network
HIDDEN_ACTOR = [400, 300] # hidden layers actor
HIDDEN_CRITIC = [400, 300] # hidden layers critic
NOISE_SIGMA = 0.2       # standard deviation of random standard normal gaussian noise added to actions for exploration purposes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Local and Target Critic-Network
        self.critic_local = CriticNetwork(state_size, action_size, seed, HIDDEN_CRITIC).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, seed, HIDDEN_CRITIC).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LRC)

        # Local and Target Actor-Network
        self.actor_local = ActorNetwork(state_size, action_size, seed, HIDDEN_ACTOR).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, seed, HIDDEN_ACTOR).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LRA)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Normalizing states
        self.state_norm = RunningNorm(state_size)
        
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.state_norm.update(state)
        self.state_norm.update(next_state)
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps and execute 10 learning steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            if len(self.memory) > 10000: # warm-up buffer of 10K observations
                for _ in range(10):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)#, indices, priorities, weights, GAMMA)
                
    def act(self, state, noise_scale):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            noise_scale (float): decay of random noise added to actions
        """
        state = self.state_norm.normalize(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state).cpu().numpy()
        self.actor_local.train()

        action_values = action_values.squeeze()
        action_values += np.random.normal(0, NOISE_SIGMA, size=self.action_size) * noise_scale
        action_values = np.clip(action_values, -1, 1) 

        return action_values

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        states = self.state_norm.normalize(states.cpu().numpy())
        states = torch.from_numpy(states).float().to(device)

        next_states = self.state_norm.normalize(next_states.cpu().numpy())
        next_states = torch.from_numpy(next_states).float().to(device)
        
        with torch.no_grad():
            next_actor_actions = self.actor_target(next_states)
            critic_value_target = self.critic_target(next_states, next_actor_actions).detach()
            y = rewards + (gamma * critic_value_target * (1 - dones))
        
        critic_value_local = self.critic_local(states, actions)
        
        loss = F.mse_loss(y, critic_value_local, reduction='mean')
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        Q_values = self.critic_local(states, actions_pred)
        
        #print("Q mean:", Q_values.mean().item(),end='\r')
        
        actor_loss = -Q_values.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store prioritized experience tuples."""
    
    def __init__(self, buffer_size, batch_size, seed, alpha, beta_start, beta_frames, eps):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        
        self.frame = 1
        self.max_priority = 1.0
        
    # -------------------------
    # Beta annealing
    # -------------------------
    def beta(self):
        return min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames
        )
    
    # -------------------------
    # Add transition
    # -------------------------
    def add(self, state, action, reward, next_state, done):
        priority = max(self.max_priority, 1e-6) # New samples get maximum priority to prevent they are never sampled
        self.tree.add(priority, state, action, reward, next_state, done)

    # -------------------------
    # Sample batch
    # -------------------------    
    def sample(self):
        """Sample a batch of prioritized experiences from memory.
        When working with segments, every batch element will come from another interval. 
        This reduces variance compared to pure random samples
        
        """
        experiences = []
        idxs = []
        priorities = []

        total = self.tree.total()
        segment = total / self.batch_size

        beta = self.beta()
        self.frame += 1

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            experiences.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        probs = (np.array(priorities) / total) + 1e-8
        weights = (self.tree.size * probs) ** (-beta) #Importance sampling weights. Prevents sampling bias
        #print('\rframe {}\tbeta: {:.2f}'.format(self.frame, beta), end="")
        weights /= (weights.max()+1e-8)  # normalize. Prevents exploding gradients
        
        #uniform_sample = np.random.uniform(0, self.total(), self.batch_size)
        #results = [self.get(s) for s in uniform_sample]
        
        #indices = [r[0] for r in results]
        #priorities = [r[1] for r in results]
        #experiences = [r[2] for r in results]
        
        # DEBUG
        count=0
        for e in experiences:
            count += 1
            if isinstance(e, int):
                print("exp:{}".format(e))
                print("count:{}".format(count))
                print("Tree: {}".format(self.tree.tree))
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        experiences_2 = (states, actions, rewards, next_states, dones)
        
        return experiences_2, idxs, priorities, weights
    
    # -------------------------
    # Update priorities
    # -------------------------
    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            td_error = td_error.detach().cpu().item()
            priority = (abs(td_error) + self.eps) ** self.alpha
            priority = max(priority, 1e-6)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    
class SumTree:
    def __init__(self, buffer_size):
        """
        buffer_size = number of leaf nodes = max replay buffer size
        """
        self.buffer_size = buffer_size
        self.tree = np.zeros(2 * buffer_size - 1)
        self.data = np.zeros(buffer_size, dtype=object)
        self.write = 0
        self.size = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    # -------------------------
    # Private helpers
    # -------------------------
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # -------------------------
    # Public API
    # -------------------------
    def total(self):
        return self.tree[0]

    def add(self, priority, state, action, reward, next_state, done):
        idx = self.write + self.buffer_size - 1
        
        data = self.experience(state, action, reward, next_state, done)

        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.buffer_size + 1

        return idx, self.tree[idx], self.data[data_idx]
    
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        # Calculate the squared differences
        squared_diff = (predictions - targets) ** 2
        # Apply weights
        weighted_loss = self.weights * squared_diff
        # Return the mean loss
        return weighted_loss.mean()

class RunningNorm:
    def __init__(self, size, eps=1e-5):
        self.size = size
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(size)
        self.var = np.ones(size)

    def update(self, x):
        x = np.array(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)
