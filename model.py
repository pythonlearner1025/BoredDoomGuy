import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import threading
import time

OBS_HEIGHT = 80
OBS_WIDTH = 60
OBS_CHANNELS = 4
OBS_SIZE = OBS_CHANNELS * OBS_HEIGHT * OBS_WIDTH

# ============================================================================
# Statistics Tracking
# ============================================================================
class RunningStats():
    def __init__(self, eps=1e-4):
        self.mean = 0
        self.count = eps
        self.M2 = 0.0
        self.lock = threading.Lock()

    # input is vectorized adv, rwd, obs
    def update(self, x):
        with self.lock:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def get(self):
        with self.lock:
            if self.count < 2:
                return self.count, self.mean, float('nan')
            var = self.M2 / (self.count - 1)  # sample variance
            return self.count, self.mean, math.sqrt(var)

class EpisodicNoveltyBuffer:
    """Lightweight episodic novelty tracker using cosine similarity in embedding space."""
    def __init__(self, embedding_dim, similarity_threshold=0.9, device='cpu'):
        self.similarity_threshold = similarity_threshold
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.states = torch.empty((0, embedding_dim), device=self.device)

    def reset(self):
        self.states = torch.empty((0, self.embedding_dim), device=self.device)

    def compute_novelty(self, phi):
        phi_vec = self._normalize(phi)
        if phi_vec.numel() == 0:
            return 0.0
        if self.states.numel() == 0:
            return 1.0
        sims = torch.matmul(self.states, phi_vec)
        visit_count = (sims > self.similarity_threshold).sum().item()
        return 1.0 / math.sqrt(visit_count + 1)

    def add(self, phi):
        phi_vec = self._normalize(phi)
        if phi_vec.numel() == 0:
            return
        self.states = torch.cat([self.states, phi_vec.unsqueeze(0)], dim=0)

    def _normalize(self, phi):
        if isinstance(phi, torch.Tensor):
            tensor = phi.detach()
        else:
            tensor = torch.tensor(phi, dtype=torch.float32, device=self.device)
        tensor = tensor.view(-1).to(self.device, dtype=torch.float32)
        norm = torch.linalg.norm(tensor, ord=2)
        if norm > 0:
            tensor = tensor / norm
        else:
            tensor = torch.zeros_like(tensor)
        return tensor


# ============================================================================
# Neural Network Components
# ============================================================================
def get_cnn(in_ch, out_dim):
    layers = [
        nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten(),
    ]
    conv = nn.Sequential(*layers)
    with torch.no_grad():
        dummy = torch.zeros(1, in_ch, OBS_HEIGHT, OBS_WIDTH)
        flat_dim = conv(dummy).shape[1]
    return nn.Sequential(
        conv,
        nn.Linear(flat_dim, out_dim)
    )

class Agent(nn.Module):
    def __init__(self, n_acts):
        super().__init__()
        self.enc = get_cnn(OBS_CHANNELS, 512)  # 4 channels for frame stack
        self.pnet = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_acts)
        )
        self.vnet = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.device = None  # will be set after .to(device)

    def forward(self, x):
        x = x / 255.0  # normalize
        z = self.enc(x)
        act_logits = self.pnet(z)
        v = self.vnet(z).squeeze(-1)
        return act_logits, v

    def to(self, device):
        """Override to track device."""
        self.device = device
        return super().to(device)

    def get_action(self, obs):
        """Sample action from policy."""
        if self.device is None:
            self.device = next(self.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action.item()

    def get_action_logprob_value(self, obs):
        """Get action, log probability, and value for training."""
        if self.device is None:
            self.device = next(self.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action.item(), logp.item(), v.item()

    def evaluate_actions(self, obs, actions):
        """Evaluate log probs, entropy, and values for given obs and actions."""
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

# LPM-specific dynamics and error models
class ConvEncoder(nn.Module):
    """Shared CNN feature extractor used by the dynamics and error models."""
    def __init__(self, in_ch=OBS_CHANNELS, feat_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, OBS_HEIGHT, OBS_WIDTH)
            flat_dim = self.conv(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x / 255.0
        z = self.conv(x)
        return self.head(z)


class LPMDynamicsModel(nn.Module):
    """Predicts next observation given current observation and action."""
    def __init__(self, n_actions, feat_dim=512, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim + n_actions, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, feat_dim)

    def forward(self, phi_obs, act_onehot):
        if act_onehot.dim() == 1:
            act_onehot = act_onehot.unsqueeze(0)
        elif act_onehot.dim() > 2:
            act_onehot = act_onehot.view(act_onehot.size(0), -1)
        z = torch.cat([phi_obs, act_onehot], dim=-1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.fc_out(z)

class LPMErrorModel(nn.Module):
    """Predicts expected previous log-MSE error for (obs, action)."""
    def __init__(self, n_actions, feat_dim=512, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, 1)
        )

    def forward(self, phi_obs, act_onehot):
        if act_onehot.dim() == 1:
            act_onehot = act_onehot.unsqueeze(0)
        elif act_onehot.dim() > 2:
            act_onehot = act_onehot.view(act_onehot.size(0), -1)
        z = torch.cat([phi_obs, act_onehot], dim=-1)
        return self.net(z).squeeze(-1)

# Random encoder sufficient for encoding states, from https://arxiv.org/pdf/1808.04355
# Use RandomEncoder when computing phi_st_next, phi_st
class RandomEncoder(nn.Module):
    def __init__(self, in_ch=4, out_dim=512):  # 4 channels for frame stack
        super().__init__()
        # Use local RNG fork to avoid clobbering global seed
        with torch.random.fork_rng():
            torch.manual_seed(42)
            self.net = get_cnn(in_ch, out_dim)

        # Add batch norm for feature stability (Burda et al. Sec 2.2)
        self.bn = nn.BatchNorm1d(out_dim, affine=False)
        self.bn.eval()  # Keep in eval mode (running stats only, no learning)

        # freeze random net
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = x / 255.0
        features = self.net(x)
        features = self.bn(features)  # Normalize features for stable curiosity scale
        return features

# Trainable CNN encoder for Inverse Dynamics Model (ICM - Pathak et al. 2017)
# Learns features that are relevant for predicting actions
class TrainableEncoder(nn.Module):
    def __init__(self, in_ch=4, out_dim=512):  # 4 channels for frame stack
        super().__init__()
        self.net = get_cnn(in_ch, out_dim)
        # Add batch norm for feature stability
        self.bn = nn.BatchNorm1d(out_dim, affine=False)
        self.bn.eval()  # Keep in eval mode (running stats only)

    def forward(self, x):
        x = x / 255.0
        features = self.net(x)
        features = self.bn(features)
        return features

# Inverse Dynamics Model: predicts action from (phi(s_t), phi(s_t+1))
# Used to train the encoder to extract action-relevant features
class InverseDynamicsModel(nn.Module):
    def __init__(self, feat_dim, n_actions, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, phi_t, phi_t_next):
        z = torch.cat([phi_t, phi_t_next], dim=-1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.fc3(z)  # Logits for action prediction

class DynamicsReplayBuffer:
    """Fixed-size replay buffer storing transitions for dynamics training."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.obs = []
        self.next_obs = []
        self.actions = []
    
    def __len__(self): return len(self.obs)
    
    def extend(self, obs_batch, action_batch, next_obs_batch):
        for o, a, n in zip(obs_batch, action_batch, next_obs_batch):
            if len(self.obs) >= self.capacity:
                # Remove from left (guaranteed uniform random due to shuffle in sample)
                self.obs.pop(0)
                self.actions.pop(0)
                self.next_obs.pop(0)
            self.obs.append(o.astype(np.uint8))
            self.actions.append(int(a))
            self.next_obs.append(n.astype(np.uint8))
    
    def sample(self, batch_size):
        # Shuffle all lists in-place to guarantee uniform random eviction
        indices = list(range(len(self.obs)))
        random.shuffle(indices)
        self.obs = [self.obs[i] for i in indices]
        self.actions = [self.actions[i] for i in indices]
        self.next_obs = [self.next_obs[i] for i in indices]
        
        # Select first batch_size elements
        obs_batch = torch.tensor(np.stack(self.obs[:batch_size]), dtype=torch.float32)
        act_batch = torch.tensor(self.actions[:batch_size], dtype=torch.long)
        next_obs_batch = torch.tensor(np.stack(self.next_obs[:batch_size]), dtype=torch.float32)
        return obs_batch, act_batch, next_obs_batch

class ErrorReplayBuffer:

    """Queue storing (obs, action, epsilon) tuples for error model training."""

    def __init__(self, capacity):
        from collections import deque
        self.capacity = capacity
        self.ready = deque(maxlen=capacity)
        self.pending = deque()

    def __len__(self):
        return len(self.ready)

    def extend(self, obs_batch, action_batch, epsilon_batch):
        for o, a, e in zip(obs_batch, action_batch, epsilon_batch):
            sample = (o.astype(np.uint8), int(a), float(e))
            self.pending.append(sample)
            if len(self.pending) > self.capacity:
                self.pending.popleft()

    def commit_pending(self):
        while self.pending:
            self.ready.append(self.pending.popleft())

    def sample(self, batch_size):
        if len(self.ready) < batch_size:
            raise ValueError("Not enough committed samples for error buffer")

        random.shuffle(self.ready)
        obs_batch = torch.tensor(np.stack([self.ready[i][0] for i in range(batch_size)]), dtype=torch.float32)
        act_batch = torch.tensor([self.ready[i][1] for i in range(batch_size)], dtype=torch.long)
        eps_batch = torch.tensor([self.ready[i][2] for i in range(batch_size)], dtype=torch.float32)
        return obs_batch, act_batch, eps_batch