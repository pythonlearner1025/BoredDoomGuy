#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learning Progress Monitoring (LPM) exploration for Doom E1M1 training with PPO.
Derived from the IDM implementation but replaces curiosity with LPM dynamics/error models.
"""
import os
import math
import time
import torch
import random
import threading
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from PIL import Image

from queue import Queue
from vizdoom import DoomGame
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
import vizdoom as vzd

os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP conflicts
os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL conflicts


def get_env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

# ============================================================================
# Hyperparameters
# ============================================================================
ITERS = get_env_int("LPM_ITERS", 1000)
EPOCHS = 4  # INCREASED: More training iterations to learn bullet/wall dynamics thoroughly
MAX_ROLLOUT_FRAMES = get_env_int("LPM_MAX_ROLLOUT_FRAMES", int(2e4))  # Reduced to avoid OOM
MAX_FRAMES_PER_EPISODE = get_env_int("LPM_MAX_FRAMES_PER_EPISODE", int(37.5*60*1))  # Per-thread 1 minute episode limit
GAMMA_INT = 0.99
GAMMA_EXT = 0.999
GAE_LAMBDA = 0.95
CLIP_COEF = 0.1
ENT_COEF = 0.01   
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
MINIBATCH_SIZE = 2048
LEARNING_RATE = 1e-4

INTRINSIC_ONLY = True  # If True, train with only intrinsic reward (no extrinsic)
INTRINSIC_COEF = 1.0 if INTRINSIC_ONLY else 0.5
EXTRINSIC_COEF = 0.0 if INTRINSIC_ONLY else 0.5  # Set to 0.0 for pure curiosity (Burda et al.)
IGNORE_DONES = True  # "Death is not the end" - infinite horizon bootstrapping
FRAME_SKIP = 2
FRAME_STACK = 4  # Stack last 4 frames as in the paper
ENCODER_FEATS = 512

# LPM specific replay/buffer parameters
ERROR_BUFFER_SIZE = MAX_ROLLOUT_FRAMES * 3
DYNAMICS_REPLAY_SIZE = MAX_ROLLOUT_FRAMES * 3

ERROR_WARMUP = ERROR_BUFFER_SIZE // 2

DYNAMICS_LEARNING_RATE = 2e-4
ERROR_LEARNING_RATE = 3e-4
DYNAMICS_GRAD_STEPS = 10
ERROR_GRAD_STEPS = 10

LPM_HIDDEN = 512
DYNAMICS_BATCH_SIZE = 512
ERROR_BATCH_SIZE = 512

EPSILON_CLAMP = 1e-6
OBS_HEIGHT = 60
OBS_WIDTH = 80
OBS_CHANNELS = FRAME_STACK
OBS_SIZE = OBS_CHANNELS * OBS_HEIGHT * OBS_WIDTH

# Macro actions for Doom E1M1 (no analog delta buttons)
MACRO_ACTIONS = [
    [],                                          # 0: NOOP
    [vzd.Button.MOVE_FORWARD],                   # 1: Forward
    [vzd.Button.MOVE_BACKWARD],                  # 2: Backward
    [vzd.Button.TURN_LEFT],                      # 3: Turn left
    [vzd.Button.TURN_RIGHT],                     # 4: Turn right
    [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT],   # 5: Forward + left
    [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_RIGHT],  # 6: Forward + right
    [vzd.Button.ATTACK],                         # 7: Attack
    [vzd.Button.USE],                            # 8: Use
    [vzd.Button.MOVE_FORWARD, vzd.Button.ATTACK],      # 9: Forward + attack
    [vzd.Button.MOVE_LEFT],                      # 10: Strafe left
    [vzd.Button.MOVE_RIGHT],                     # 11: Strafe right
]

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
        self.capacity = capacity
        self.storage = []
    
    def __len__(self): return len(self.storage)
    
    def extend(self, obs_batch, action_batch, epsilon_batch):
        for o, a, e in zip(obs_batch, action_batch, epsilon_batch):
            sample = (o.astype(np.uint8), int(a), float(e))
            self.storage.append(sample)
    
    def sample(self, batch_size):
        
        # Shuffle list in-place to guarantee uniform random eviction
        random.shuffle(self.storage)
        
        # Select first batch_size elements
        obs_batch = torch.tensor(np.stack([self.storage[i][0] for i in range(batch_size)]), dtype=torch.float32)
        act_batch = torch.tensor([self.storage[i][1] for i in range(batch_size)], dtype=torch.long)
        eps_batch = torch.tensor([self.storage[i][2] for i in range(batch_size)], dtype=torch.float32)
        return obs_batch, act_batch, eps_batch
# ============================================================================
# GAE Computation
# ============================================================================
def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda, ignore_dones=False):
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: (T, N) tensor of rewards
        values: (T, N) tensor of value estimates
        dones: (T, N) tensor of done flags
        next_value: (N,) tensor of bootstrap value
        gamma: discount factor
        gae_lambda: GAE lambda parameter
        ignore_dones: If True, use infinite-horizon (death is not the end)

    Returns:
        advantages: (T, N) tensor of advantages
        returns: (T, N) tensor of returns
    """
    T, N = rewards.shape
    device = rewards.device
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=device)

    for t in reversed(range(T)):
        if t == T - 1:
            nextvalues = next_value
        else:
            nextvalues = values[t+1]

        # FIX: Use dones[t] not dones[t+1], and honor IGNORE_DONES
        if ignore_dones:
            nextnonterminal = torch.ones(N, device=device)  # Infinite horizon
        else:
            nextnonterminal = (~dones[t]).float()

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values
    return advantages, returns

# ============================================================================
# Episode Rollout
# ============================================================================
def rgb_to_grayscale(rgb_frame):
    """Convert RGB frame to grayscale using standard weights."""
    # ViZDoom screen_buffer shape: (3, H, W) or (H, W, 3) depending on format
    # Check shape and convert accordingly
    if rgb_frame.shape[0] == 3:
        # CHW format: (3, H, W)
        gray = (0.299 * rgb_frame[0] +
                0.587 * rgb_frame[1] +
                0.114 * rgb_frame[2])
    else:
        # HWC format: (H, W, 3)
        gray = (0.299 * rgb_frame[:, :, 0] +
                0.587 * rgb_frame[:, :, 1] +
                0.114 * rgb_frame[:, :, 2])
    return gray.astype(np.uint8)

def resize_gray_frame(gray_frame):
    """Resize grayscale frame to (OBS_HEIGHT, OBS_WIDTH)."""
    if gray_frame.shape[0] == OBS_HEIGHT and gray_frame.shape[1] == OBS_WIDTH:
        return gray_frame
    img = Image.fromarray(gray_frame)
    resized = img.resize((OBS_WIDTH, OBS_HEIGHT), resample=Image.BILINEAR)
    return np.array(resized, dtype=np.uint8)

def stack_frames(frame_buffer, new_frame):
    """Stack last 4 grayscale frames. Returns stacked array (4, H, W)."""
    gray_frame = resize_gray_frame(rgb_to_grayscale(new_frame))
    if len(frame_buffer) == 0:
        # Initialize with 4 copies of first frame
        return [gray_frame] * FRAME_STACK
    else:
        frame_buffer = frame_buffer[1:] + [gray_frame]  # Shift and append
        return frame_buffer
       
def run_episode(thread_id, agent, dynamics_model, error_model, encoder_model, buffer, stats, frame_counter,
                error_ready, save_rgb_debug, device='cpu'):
    game = vzd.DoomGame()
    # Get WAD path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wad_path = os.path.join(script_dir, "Doom1.WAD")
    if not os.path.exists(wad_path):
        raise FileNotFoundError(f"Doom1.WAD not found at {wad_path}. Please place it in the repo root.")
    game.set_doom_game_path(wad_path)
    game.set_doom_map("E1M1")
    game.set_render_hud(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args("+freelook 1")

    # Set available buttons from macro actions
    all_buttons = set()
    for action in MACRO_ACTIONS:
        all_buttons.update(action)
    game.set_available_buttons(list(all_buttons))

    game.init()
    game.new_episode()

    # Storage as uint8 to save memory
    obss = []  # Stacked frames (4, H, W) uint8 grayscale
    obss_next = []
    # Only save RGB if this thread is selected for debug (HUGE memory savings)
    obss_rgb = [] if save_rgb_debug else None
    obss_next_rgb = [] if save_rgb_debug else None
    acts = []
    logps = []
    vals = []
    rwds_e = []
    rwds_i = []
    dones = []
    epsilons = []

    run_rwd_i, run_rwd_e = 0, 0
    n_actions = len(MACRO_ACTIONS)
    frames_collected = 0
    frame_buffer = []  # For frame stacking

    # "Death is not the end" - continue across episode boundaries
    while frames_collected < MAX_FRAMES_PER_EPISODE:
        # Restart episode if finished (infinite horizon)
        if game.is_episode_finished():
            game.new_episode()
            frame_buffer = []  # Reset frame stack on new episode
            continue

        state = game.get_state()
        if state is None:
            continue

        # Get raw frame, convert to grayscale, and stack
        raw_frame = state.screen_buffer  # (3, H, W) RGB uint8 CHW format
        raw_frame_hwc = raw_frame.transpose(1, 2, 0)  # Convert to (H, W, 3) for saving
        frame_buffer = stack_frames(frame_buffer, raw_frame)
        obs_stacked = np.stack(frame_buffer, axis=0)  # (4, H, W) grayscale uint8

        # Get action from agent
        obs_t = torch.tensor(obs_stacked, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, val = agent.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            act_idx = dist.sample()
            logp = dist.log_prob(act_idx)

        act_idx_item = act_idx.item()
        act_onehot = F.one_hot(act_idx, n_actions).float().unsqueeze(0)
        with torch.no_grad():
            phi_t = encoder_model(obs_t)
            pred_next = dynamics_model(phi_t, act_onehot)
            expected_error_log = error_model(phi_t, act_onehot).item()
            expected_error = math.exp(expected_error_log)

        # Take action using macro action
        action_buttons = MACRO_ACTIONS[act_idx_item]
        action_vector = [btn in action_buttons for btn in list(all_buttons)]
        rwd_e = game.make_action(action_vector, FRAME_SKIP)

        # Get next state
        state_next = game.get_state()
        done_flag = game.is_episode_finished()

        if state_next is not None:
            raw_frame_next = state_next.screen_buffer
            raw_frame_next_hwc = raw_frame_next.transpose(1, 2, 0)  # (H, W, 3)
            frame_buffer_next = stack_frames(frame_buffer, raw_frame_next)
            obs_next_stacked = np.stack(frame_buffer_next, axis=0)  # (4, H, W) grayscale

            # Compute intrinsic reward using full frame stack
            with torch.no_grad():
                obs_next_t = torch.tensor(obs_next_stacked, dtype=torch.float32, device=device).unsqueeze(0)
                phi_next_t = encoder_model(obs_next_t)

                mse = F.mse_loss(pred_next, phi_next_t, reduction='none')
            mse_flat = mse.view(mse.shape[0], -1).mean(dim=1)
            epsilon = mse_flat.item()
        else:
            obs_next_stacked = np.zeros_like(obs_stacked)  # Placeholder
            raw_frame_next_hwc = np.zeros_like(raw_frame_hwc)
            epsilon = 0.0

        intrinsic_reward = (expected_error - epsilon) if error_ready else 0.0

        # Store transition as uint8 to save memory
        obss.append(obs_stacked.astype(np.uint8))
        obss_next.append(obs_next_stacked.astype(np.uint8))

        # Only store RGB if this thread is saving debug frames (saves ~1.7GB)
        if save_rgb_debug:
            obss_rgb.append(raw_frame_hwc)
            obss_next_rgb.append(raw_frame_next_hwc)

        acts.append(act_idx_item)
        logps.append(logp.item())
        vals.append(val.item())
        rwds_e.append(rwd_e)
        rwds_i.append(intrinsic_reward)
        dones.append(done_flag)
        epsilons.append(epsilon)

        frames_collected += FRAME_SKIP

        # Update running rewards for stats
        run_rwd_i = run_rwd_i * GAMMA_INT + intrinsic_reward
        run_rwd_e = run_rwd_e * GAMMA_EXT + rwd_e

    # Update stats
    rwd_i_rms, rwd_e_rms = stats
    rwd_i_rms.update(run_rwd_i)
    rwd_e_rms.update(run_rwd_e)

    # Update frame counter atomically
    with frame_counter:
        frame_counter.count += len(obss)

    game.close()

    # Put episode data in buffer (include RGB frames for debug)
    buffer.put((thread_id, obss, obss_next, acts, logps, vals, rwds_e, rwds_i, epsilons, dones, obss_rgb, obss_next_rgb))

class FrameCounter:
    """Thread-safe frame counter."""
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

    def get(self):
        with self.lock:
            return self.count

    def reset(self):
        with self.lock:
            self.count = 0

def run_parallel_episodes(n_threads, agent, dynamics_model, error_model, encoder_model, buffer, stats, frame_counter,
                         error_ready, debug_thread_id, device='cpu'):
    """
    Args:
        debug_thread_id: Only this thread will save RGB frames for debug
    """
    threads = []

    for tid in range(n_threads):
        save_rgb = (tid == debug_thread_id)  # Only one thread saves RGB
        t = threading.Thread(
            target=run_episode,
            args=(tid, agent, dynamics_model, error_model, encoder_model, buffer, stats, frame_counter, error_ready, save_rgb, device)
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

# ============================================================================
# PPO Update
# ============================================================================
def ppo_update(agent, optimizer, obs_batch, act_batch, old_logp_batch,
               adv_batch, ret_batch, clip_coef=CLIP_COEF, ent_coef=ENT_COEF,
               vf_coef=VF_COEF):
    """
    Perform one PPO update step.
    """
    # Normalize advantages
    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

    # Get new predictions
    new_logp, entropy, values = agent.evaluate_actions(obs_batch, act_batch)

    # Policy loss with clipping
    ratio = (new_logp - old_logp_batch).exp()
    unclipped = ratio * adv_batch
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_batch
    pi_loss = -torch.min(unclipped, clipped).mean()

    # Value loss
    v_loss = 0.5 * F.mse_loss(values, ret_batch)

    # Entropy bonus
    ent_loss = -entropy.mean()

    # Total loss
    loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return pi_loss.item(), v_loss.item(), ent_loss.item()

# Forward dynamics update is now inlined in the training loop to properly handle terminals

# ============================================================================
# Main Training Loop
# ============================================================================
def init_wandb(config):
    """Initialize wandb with support for disabled/offline modes."""
    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower()
    disable = wandb_mode in {"disabled", "offline"} or wandb_disabled in {"true", "1", "yes"}
    if disable:
        return wandb.init(mode="disabled", config=config, project="doom-idm-curiosity")
    return wandb.init(project="doom-idm-curiosity", config=config)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    os.makedirs('debug', exist_ok=True)

    wandb_config = {
            "iters": ITERS,
            "epochs": EPOCHS,
            "max_rollout_frames": MAX_ROLLOUT_FRAMES,
            "max_frames_per_episode": MAX_FRAMES_PER_EPISODE,
            "gamma_int": GAMMA_INT,
            "gamma_ext": GAMMA_EXT,
            "gae_lambda": GAE_LAMBDA,
            "clip_coef": CLIP_COEF,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
            "minibatch_size": MINIBATCH_SIZE,
            "policy_lr": LEARNING_RATE,
            "dynamics_lr": DYNAMICS_LEARNING_RATE,
            "error_lr": ERROR_LEARNING_RATE,
            "intrinsic_coef": INTRINSIC_COEF,
            "extrinsic_coef": EXTRINSIC_COEF,
            "intrinsic_only": INTRINSIC_ONLY,
            "ignore_dones": IGNORE_DONES,
            "frame_skip": FRAME_SKIP,
            "frame_stack": FRAME_STACK,
            "error_buffer_size": ERROR_BUFFER_SIZE,
            "error_warmup": ERROR_WARMUP,
            "dynamics_replay_size": DYNAMICS_REPLAY_SIZE,
            "dynamics_batch_size": DYNAMICS_BATCH_SIZE,
            "error_batch_size": ERROR_BATCH_SIZE,
            "dynamics_grad_steps": DYNAMICS_GRAD_STEPS,
            "error_grad_steps": ERROR_GRAD_STEPS,
            "epsilon_clamp": EPSILON_CLAMP,
            "n_actions": len(MACRO_ACTIONS),
            "python_version": "3.14t",
            "gil_disabled": True,
    }
    init_wandb(wandb_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cpus = multiprocessing.cpu_count()
    n_threads_override = os.environ.get("LPM_THREADS")
    if n_threads_override is not None:
        try:
            n_threads = max(1, int(n_threads_override))
        except ValueError:
            n_threads = max(1, cpus - 2)
    else:
        n_threads = max(1, cpus - 2)
    print(f"Using {n_threads} worker threads")

    n_actions = len(MACRO_ACTIONS)
    agent = Agent(n_actions).to(device)
    dynamics_model = LPMDynamicsModel(n_actions, feat_dim=ENCODER_FEATS, hidden=LPM_HIDDEN).to(device)
    error_model = LPMErrorModel(n_actions, feat_dim=ENCODER_FEATS, hidden=LPM_HIDDEN).to(device)
    encoder_model = RandomEncoder(out_dim=ENCODER_FEATS).to(device)

    agent_optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=DYNAMICS_LEARNING_RATE)
    error_optimizer = torch.optim.Adam(error_model.parameters(), lr=ERROR_LEARNING_RATE)

    dynamics_buffer = DynamicsReplayBuffer(DYNAMICS_REPLAY_SIZE)
    error_buffer = ErrorReplayBuffer(ERROR_BUFFER_SIZE)

    rwd_i_rms = RunningStats()
    rwd_e_rms = RunningStats()
    stats = (rwd_i_rms, rwd_e_rms)

    for iter_idx in range(ITERS):
        print(f"\n=== Iteration {iter_idx+1}/{ITERS} ===")
        iter_start = time.time()

        buffer = Queue()
        frame_counter = FrameCounter()
        debug_thread_id = random.randint(0, n_threads - 1)
        error_ready = len(error_buffer) >= ERROR_WARMUP

        all_obs = []
        all_obs_next = []
        all_acts = []
        all_logps = []
        all_advantages = []
        all_returns = []
        all_dones = []
        all_rwds_i_raw = []
        all_rwds_e_raw = []
        all_epsilons = []
        debug_episode_saved = False
        total_episodes = 0

        dynamics_model.eval()
        error_model.eval()
        encoder_model.eval()

        while frame_counter.get() < MAX_ROLLOUT_FRAMES:
            run_parallel_episodes(
                n_threads, agent, dynamics_model, error_model, encoder_model,
                buffer, stats, frame_counter, error_ready, debug_thread_id, device
            )

            while not buffer.empty():
                ep = buffer.get()
                (thread_id, obss, obss_next, acts, logps, vals,
                 rwds_e, rwds_i, epsilons_ep, dones,
                 obss_rgb, obss_next_rgb) = ep
                total_episodes += 1

                if thread_id == debug_thread_id and not debug_episode_saved and obss_rgb is not None:
                    debug_dir = f'debug/iter_{iter_idx+1:04d}'
                    os.makedirs(debug_dir, exist_ok=True)
                    action_names = ['NOOP', 'FWD', 'BACK', 'TURN_L', 'TURN_R', 'FWD_L', 'FWD_R',
                                    'ATTACK', 'USE', 'FWD_ATK', 'STRAFE_L', 'STRAFE_R']
                    for i, (rgb_curr, rgb_next, act_idx, r_i, r_e) in enumerate(
                        zip(obss_rgb, obss_next_rgb, acts, rwds_i, rwds_e)
                    ):
                        Image.fromarray(rgb_curr, mode='RGB').save(f'{debug_dir}/frame_{i:04d}_current.png')
                        Image.fromarray(rgb_next, mode='RGB').save(f'{debug_dir}/frame_{i:04d}_next.png')
                        action_name = action_names[act_idx] if act_idx < len(action_names) else f'ACT_{act_idx}'
                        with open(f'{debug_dir}/frame_{i:04d}_{action_name}.txt', 'w') as f:
                            f.write(f'Action: {action_name}\\n')
                            f.write(f'Intrinsic Reward: {r_i:.6f}\\n')
                            f.write(f'Extrinsic Reward: {r_e:.3f}\\n')

                    print(f"Saved {len(obss_rgb)} RGB frame pairs to {debug_dir}/")
                    debug_episode_saved = True

                obs_ep = torch.tensor(np.array(obss), dtype=torch.uint8)
                obs_next_ep = torch.tensor(np.array(obss_next), dtype=torch.uint8)
                acts_ep = torch.tensor(acts, dtype=torch.long)
                logps_ep = torch.tensor(logps, dtype=torch.float32)
                vals_ep = torch.tensor(vals, dtype=torch.float32)
                rwds_e_ep = torch.tensor(rwds_e, dtype=torch.float32)
                rwds_i_ep = torch.tensor(rwds_i, dtype=torch.float32)
                eps_ep = torch.tensor(epsilons_ep, dtype=torch.float32)
                dones_ep = torch.tensor(dones, dtype=torch.bool)

                _, _, std_i = rwd_i_rms.get()
                _, _, std_e = rwd_e_rms.get()

                rwds_i_norm = rwds_i_ep / max(std_i if not math.isnan(std_i) else 1.0, 1e-3)
                rwds_e_norm = rwds_e_ep / max(std_e if not math.isnan(std_e) else 1.0, 1e-3)

                total_rwds_ep = EXTRINSIC_COEF * rwds_e_norm + INTRINSIC_COEF * rwds_i_norm

                with torch.no_grad():
                    if IGNORE_DONES or not dones_ep[-1]:
                        obs_next_last = obs_next_ep[-1].float().to(device).unsqueeze(0)
                        _, next_val = agent.forward(obs_next_last)
                        next_val = next_val[0].cpu()
                    else:
                        next_val = torch.tensor(0.0)

                advs_ep, rets_ep = compute_gae(
                    total_rwds_ep.unsqueeze(1),
                    vals_ep.unsqueeze(1),
                    dones_ep.unsqueeze(1),
                    next_val.unsqueeze(0),
                    GAMMA_EXT,
                    GAE_LAMBDA,
                    ignore_dones=IGNORE_DONES
                )

                all_obs.append(obs_ep)
                all_obs_next.append(obs_next_ep)
                all_acts.append(acts_ep)
                all_logps.append(logps_ep)
                all_advantages.append(advs_ep.squeeze(1))
                all_returns.append(rets_ep.squeeze(1))
                all_dones.append(dones_ep)
                all_epsilons.append(eps_ep)

                all_rwds_i_raw.extend(rwds_i)
                all_rwds_e_raw.extend(rwds_e)

                del ep, obss, obss_next, obss_rgb, obss_next_rgb

        dynamics_model.train()
        error_model.train()
        print(f"Collected {total_episodes} episodes, {frame_counter.get()} frames")

        if not all_obs:
            print("No data collected this iteration, skipping updates.")
            continue

        obs_t = torch.cat(all_obs, dim=0)
        obs_next_t = torch.cat(all_obs_next, dim=0)
        acts_t = torch.cat(all_acts, dim=0)
        logps_t = torch.cat(all_logps, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        returns = torch.cat(all_returns, dim=0)
        dones_t = torch.cat(all_dones, dim=0)
        eps_t = torch.cat(all_epsilons, dim=0)

        dynamics_buffer.extend(
            obs_t.cpu().numpy(),
            acts_t.cpu().numpy(),
            obs_next_t.cpu().numpy()
        )
        error_buffer.extend(
            obs_t.cpu().numpy(),
            acts_t.cpu().numpy(),
            eps_t.cpu().numpy()
        )

        n_samples = len(obs_t)
        indices = np.arange(n_samples)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_ent_loss = 0.0
        total_dynamics_loss = 0.0
        error_loss_total = 0.0
        n_updates = 0
        dynamics_updates = 0
        error_updates = 0

        eps_pred_total = 0.0
        eps_actual_total = 0.0

        for epoch in range(EPOCHS):
            np.random.shuffle(indices)
            epoch_start = time.time()
            for start in range(0, n_samples, MINIBATCH_SIZE):
                end = min(start + MINIBATCH_SIZE, n_samples)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx].float().to(device)
                pi_loss, v_loss, ent_loss = ppo_update(
                    agent, agent_optimizer,
                    mb_obs, acts_t[mb_idx].to(device), logps_t[mb_idx].to(device),
                    advantages[mb_idx].to(device), returns[mb_idx].to(device)
                )

                total_pi_loss += pi_loss
                total_v_loss += v_loss
                total_ent_loss += ent_loss
                n_updates += 1

            for _ in range(DYNAMICS_GRAD_STEPS):
                if len(dynamics_buffer) < DYNAMICS_BATCH_SIZE:
                    break
                obs_b, act_b, next_obs_b = dynamics_buffer.sample(DYNAMICS_BATCH_SIZE)
                obs_b = obs_b.to(device)
                next_obs_b = next_obs_b.to(device)
                act_onehot = F.one_hot(act_b.to(device), n_actions).float()

                with torch.no_grad():
                    phi_obs = encoder_model(obs_b)
                    phi_next = encoder_model(next_obs_b)

                pred_next = dynamics_model(phi_obs, act_onehot)
                dyn_loss = F.mse_loss(pred_next, phi_next)

                dynamics_optimizer.zero_grad(set_to_none=True)
                dyn_loss.backward()
                nn.utils.clip_grad_norm_(dynamics_model.parameters(), MAX_GRAD_NORM)
                dynamics_optimizer.step()

                total_dynamics_loss += dyn_loss.item()
                dynamics_updates += 1

            for _ in range(ERROR_GRAD_STEPS):
                if len(error_buffer) < ERROR_BATCH_SIZE:
                    break
                obs_b, act_b, eps_b = error_buffer.sample(ERROR_BATCH_SIZE)
                obs_b = obs_b.to(device)
                act_onehot = F.one_hot(act_b.to(device), n_actions).float()
                eps_b = eps_b.to(device)

                with torch.no_grad():
                    phi_obs = encoder_model(obs_b)

                eps_pred = error_model(phi_obs, act_onehot)
                log_targets = torch.log(eps_b + EPSILON_CLAMP)
                err_loss = F.mse_loss(eps_pred, log_targets)

                error_optimizer.zero_grad(set_to_none=True)
                err_loss.backward()
                nn.utils.clip_grad_norm_(error_model.parameters(), MAX_GRAD_NORM)
                error_optimizer.step()

                error_loss_total += err_loss.item()
                error_updates += 1

                eps_pred_total += torch.exp(eps_pred).mean().item()
                eps_actual_total += eps_b.mean().item()

            epoch_end = time.time()
            print(f"EPOCH {epoch} took {(epoch_end - epoch_start):0.2f}s")

        pi_loss = total_pi_loss / max(n_updates, 1)
        v_loss = total_v_loss / max(n_updates, 1)
        ent_loss = total_ent_loss / max(n_updates, 1)
        dynamics_loss = total_dynamics_loss / max(dynamics_updates, 1) if dynamics_updates else 0.0
        # log these to track error prediction sanity
        error_loss_mean = error_loss_total / max(error_updates, 1) if error_updates else 0.0
        error_pred_mean = eps_pred_total / max(error_updates, 1) if error_updates else 0.0
        error_actual_mean = eps_actual_total / max(error_updates, 1) if error_updates else 0.0

        iter_time = time.time() - iter_start
        _, mean_i_running, std_i_running = rwd_i_rms.get()
        _, mean_e_running, std_e_running = rwd_e_rms.get()

        batch_mean_i = np.mean(all_rwds_i_raw) if all_rwds_i_raw else 0.0
        batch_std_i = np.std(all_rwds_i_raw) if all_rwds_i_raw else 0.0
        batch_max_i = np.max(all_rwds_i_raw) if all_rwds_i_raw else 0.0
        batch_min_i = np.min(all_rwds_i_raw) if all_rwds_i_raw else 0.0

        batch_mean_e = np.mean(all_rwds_e_raw) if all_rwds_e_raw else 0.0
        batch_std_e = np.std(all_rwds_e_raw) if all_rwds_e_raw else 0.0
        batch_max_e = np.max(all_rwds_e_raw) if all_rwds_e_raw else 0.0
        batch_sum_e = np.sum(all_rwds_e_raw) if all_rwds_e_raw else 0.0

        total_reward = EXTRINSIC_COEF * mean_e_running + INTRINSIC_COEF * mean_i_running
        batch_total = EXTRINSIC_COEF * batch_mean_e + INTRINSIC_COEF * batch_mean_i

        eps_mean = eps_t.mean().item() if len(eps_t) > 0 else 0.0
        eps_std = eps_t.std().item() if len(eps_t) > 0 else 0.0
        eps_max = eps_t.max().item() if len(eps_t) > 0 else 0.0
        eps_min = eps_t.min().item() if len(eps_t) > 0 else 0.0

        log_dict = {
            "update_step": iter_idx + 1,
            "reward/intrinsic_running": mean_i_running,
            "reward/extrinsic_running": mean_e_running,
            "reward/intrinsic_std_running": std_i_running if not math.isnan(std_i_running) else 0,
            "reward/extrinsic_std_running": std_e_running if not math.isnan(std_e_running) else 0,
            "reward/intrinsic_batch_mean": batch_mean_i,
            "reward/intrinsic_batch_std": batch_std_i,
            "reward/intrinsic_batch_max": batch_max_i,
            "reward/intrinsic_batch_min": batch_min_i,
            "reward/extrinsic_batch_mean": batch_mean_e,
            "reward/extrinsic_batch_std": batch_std_e,
            "reward/extrinsic_batch_max": batch_max_e,
            "reward/extrinsic_batch_sum": batch_sum_e,
            "reward/total_batch": batch_total,
            "loss/policy": pi_loss,
            "loss/value": v_loss,
            "loss/entropy": -ent_loss,
            "loss/dynamics": dynamics_loss,

            # error stuff
            "loss/error_model": error_loss_mean,
            "lpm/error_pred": error_pred_mean,
            "lpm/error_actual": error_actual_mean,

            "time/iteration_time": iter_time,
            "time/fps": frame_counter.get() / iter_time,
            "data/episodes_collected": total_episodes,
            "data/frames_collected": frame_counter.get(),
            "data/dynamics_buffer": len(dynamics_buffer),
            "data/error_buffer": len(error_buffer),
            "data/error_ready": float(error_ready),
            "lpm/epsilon_mean": eps_mean,
            "lpm/epsilon_std": eps_std,
            "lpm/epsilon_max": eps_max,
            "lpm/epsilon_min": eps_min,
        }

        for k,v in log_dict.items():
            print(k, ": ", v)


        wandb.log(log_dict)

        print(f"Timer {iter_time:.1f}s | FPS: {frame_counter.get()/iter_time:.0f}")
        print(f"Policy Loss: {pi_loss:.4f}, Value Loss: {v_loss:.4f}, Entropy: {-ent_loss:.4f}")
        print(f"Dynamics Loss: {dynamics_loss:.6f}, Error Model Loss: {error_loss_mean:.6f}")
        print(f"Intrinsic (batch): μ={batch_mean_i:.6f}, σ={batch_std_i:.6f}, max={batch_max_i:.6f}")
        print(f"Extrinsic (batch): μ={batch_mean_e:.3f}, max={batch_max_e:.3f}, sum={batch_sum_e:.1f}")
        if INTRINSIC_ONLY:
            print("  ^ Note: Training with INTRINSIC_ONLY (extrinsic not used for learning)")

        if (iter_idx + 1) % 25 == 0:
            checkpoint = {
                'iteration': iter_idx + 1,
                'agent_state_dict': agent.state_dict(),
                'dynamics_state_dict': dynamics_model.state_dict(),
                'error_state_dict': error_model.state_dict(),
                'agent_optimizer': agent_optimizer.state_dict(),
                'dynamics_optimizer': dynamics_optimizer.state_dict(),
                'error_optimizer': error_optimizer.state_dict(),
            }
            torch.save(checkpoint, f'idm_lpm_checkpoint_{iter_idx+1}.pth')
            print(f"Saved checkpoint at iteration {iter_idx+1}")

    print("\nTraining complete!")
    wandb.finish()
