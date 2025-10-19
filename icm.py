#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IDM (Intrinsic Curiosity Module) for Doom E1M1 training with PPO.
Running with Python 3.14t (free-threading, no GIL) for true parallelism.
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
# ============================================================================
# Hyperparameters
# ============================================================================
ITERS = 1000
EPOCHS = 3  # INCREASED: More training iterations to learn bullet/wall dynamics thoroughly
MAX_ROLLOUT_FRAMES = int(3e4)  # Reduced to avoid OOM
MAX_FRAMES_PER_EPISODE = int(37.5*60*1)  # Per-thread 1 minute episode limit
GAMMA_INT = 0.99
GAMMA_EXT = 0.999
GAE_LAMBDA = 0.95
CLIP_COEF = 0.1
ENT_COEF = 0.01 # TODO inc this?  
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
MINIBATCH_SIZE = 1024
LEARNING_RATE = 1e-4
FWD_LEARNING_RATE = 1e-4  # Higher LR for forward model to learn dynamics faster
INTRINSIC_ONLY = True  # If True, train with only intrinsic reward (no extrinsic)
INTRINSIC_COEF = 1.0 if INTRINSIC_ONLY else 0.5
EXTRINSIC_COEF = 0.0 if INTRINSIC_ONLY else 0.5  # Set to 0.0 for pure curiosity (Burda et al.)
IGNORE_DONES = True  # "Death is not the end" - infinite horizon bootstrapping
USE_INVERSE_DYNAMICS = True  # If True, use trainable encoder with IDM loss (Pathak et al.)
FWD_ENC_COEF = 0.25
FEAT_DIM = 512
FWD_HIDDEN = 1024  # DONE Larger forward model to memorize bullet/wall dynamics
FWD_LAYERS = 2

IDM_HIDDEN = 512  # Hidden size for inverse dynamics model
FEAT_LEARNING_RATE = 1e-5  # DONE Lower learning rate for stability 
FRAME_SKIP = 4
FRAME_STACK = 4  # Stack last 4 frames as in the paper

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
    return nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(11264, out_dim)
        )

class Agent(nn.Module):
    def __init__(self, n_acts):
        super().__init__()
        self.enc = get_cnn(4, 512)  # 4 channels for frame stack
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

# Random encoder sufficient for encoding states, from https://arxiv.org/pdf/1808.04355
# Use RandomEncoder when computing phi_tt_next, phi_tt
class RandomEncoder(nn.Module):
    def __init__(self, in_ch=4, out_dim=512):  # 4 channels for frame stack
        super().__init__()
        # Use local RNG fork to avoid clobbering global seed
        with torch.random.fork_rng():
            torch.manual_seed(69)
            self.net = get_cnn(in_ch, out_dim)

        # Add batch norm for feature stability (Burda et al. Sec 2.2)
        self.bn = nn.BatchNorm1d(out_dim, affine=False)
        self.bn.eval()  # Keep in eval mode (running stats only, no learning)

        # freeze random net
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Don't use no_grad here - the parameters are frozen but we need
        # the output to be part of the computation graph for fwd_model training
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

'''
# Forward dynamics model: f(phi(s), a) -> phi(s')
# TODO FDM loss steadily increases and tied to intrinsic rwd - maybe inc capacity?
class ForwardDynamicsModel(nn.Module):
    def __init__(self, feat_dim, n_actions, hidden=1024):
        super().__init__()
        # Deeper network to better memorize stochastic dynamics (bullet holes, particles)
        self.fc1 = nn.Linear(feat_dim + n_actions, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(FWD_LAYERS)])
        self.fc_head = nn.Linear(hidden, feat_dim)

    def forward(self, phi_t, a_onehot):
        z = torch.cat([phi_t, a_onehot], dim=-1)
        z = F.relu(self.fc1(z))
        for block in self.blocks:
            z = F.relu(block(z))
        return self.fc_head(z)  # mean of Gaussian with fixed variance

'''
class ForwardDynamicsModel(nn.Module):
    def __init__(self, feat_dim, n_actions, hidden=1024, gru_hidden=512):
        super().__init__()
        # Larger first layer to process state+action
        self.fc1 = nn.Linear(feat_dim + n_actions, hidden)
        
        # GRU to maintain temporal context (memory of recent states/actions)
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=True)
        self.gru_hidden = None  # Will store hidden state across forward passes
        
        # Deeper network with more capacity
        self.fc2 = nn.Linear(hidden + gru_hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, feat_dim)
        
    def forward(self, phi_t, a_onehot):
        batch_size = phi_t.shape[0]
        device = phi_t.device
        
        # Process current state-action pair
        z = torch.cat([phi_t, a_onehot], dim=-1)
        z = F.relu(self.fc1(z))
        
        # Pass through GRU to get temporal context
        z_seq = z.unsqueeze(1)  # Add sequence dimension for GRU
        # Initialize or resize hidden state if needed
        if self.gru_hidden is None or self.gru_hidden.shape[1] != batch_size:
            self.gru_hidden = torch.zeros(1, batch_size, 512).to(device)
        
        gru_out, self.gru_hidden = self.gru(z_seq, self.gru_hidden.detach())
        gru_features = gru_out.squeeze(1)
        
        # Combine current features with temporal context
        combined = torch.cat([z, gru_features], dim=-1)
        
        # Process through deeper layers
        z = F.relu(self.fc2(combined))
        z = F.relu(self.fc3(z))
        return self.fc4(z)
    
    def reset_hidden(self):
        """Call this at episode boundaries to reset GRU memory"""
        self.gru_hidden = None

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

def stack_frames(frame_buffer, new_frame):
    """Stack last 4 grayscale frames. Returns stacked array (4, H, W)."""
    gray_frame = rgb_to_grayscale(new_frame)
    if len(frame_buffer) == 0:
        # Initialize with 4 copies of first frame
        return [gray_frame] * FRAME_STACK
    else:
        frame_buffer = frame_buffer[1:] + [gray_frame]  # Shift and append
        return frame_buffer

def run_episode(thread_id, agent, fwd_model, phi_enc, buffer, stats, frame_counter,
                save_rgb_debug, device='cpu'):
    """
    Args:
        save_rgb_debug: If True, save RGB frames for debug visualization
    """
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
        act_onehot = torch.zeros(n_actions, device=device)
        act_onehot[act_idx_item] = 1.0

        # Compute predicted next state features for intrinsic reward
        # Use full frame stack for phi encoding
        with torch.no_grad():
            phi_t = phi_enc(obs_t)
            phi_hat = fwd_model(phi_t, act_onehot.unsqueeze(0))

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
                phi_t_next = phi_enc(obs_next_t)
                rwd_i = F.mse_loss(phi_hat, phi_t_next, reduction='none').mean().item()
        else:
            obs_next_stacked = np.zeros_like(obs_stacked)  # Placeholder
            raw_frame_next_hwc = np.zeros((obs_stacked.shape[1], obs_stacked.shape[2], 3), dtype=np.uint8)
            rwd_i = 0.0

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
        rwds_i.append(rwd_i)
        dones.append(done_flag)

        frames_collected += FRAME_SKIP

        # Update running rewards for stats
        run_rwd_i = run_rwd_i * GAMMA_INT + rwd_i
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
    buffer.put((thread_id, obss, obss_next, acts, logps, vals, rwds_e, rwds_i, dones, obss_rgb, obss_next_rgb))

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

def run_parallel_episodes(n_threads, agent, fwd_model, phi_enc, buffer, stats, frame_counter,
                         debug_thread_id, device='cpu'):
    """
    Args:
        debug_thread_id: Only this thread will save RGB frames for debug
    """
    threads = []

    for tid in range(n_threads):
        save_rgb = (tid == debug_thread_id)  # Only one thread saves RGB
        t = threading.Thread(
            target=run_episode,
            args=(tid, agent, fwd_model, phi_enc, buffer, stats, frame_counter, save_rgb, device)
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
if __name__ == '__main__':
    # Enable anomaly detection to find in-place operations
    torch.autograd.set_detect_anomaly(True)

    # Create debug directory
    os.makedirs('debug', exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="doom-idm-curiosity",
        config={
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
            "learning_rate": LEARNING_RATE,
            "fwd_learning_rate": FWD_LEARNING_RATE,
            "intrinsic_coef": INTRINSIC_COEF,
            "extrinsic_coef": EXTRINSIC_COEF,
            "intrinsic_only": INTRINSIC_ONLY,
            "ignore_dones": IGNORE_DONES,
            "use_inverse_dynamics": USE_INVERSE_DYNAMICS,
            "feat_dim": FEAT_DIM,
            "fwd_hidden": FWD_HIDDEN,
            "idm_hidden": IDM_HIDDEN,
            "feat_learning_rate": FEAT_LEARNING_RATE if USE_INVERSE_DYNAMICS else None,
            "frame_skip": FRAME_SKIP,
            "frame_stack": FRAME_STACK,
            "n_actions": len(MACRO_ACTIONS),
            "python_version": "3.14t",
            "gil_disabled": True,
        }
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check number of cpu cores
    cpus = 8#int(multiprocessing.cpu_count() * 1)
    n_threads = max(1, cpus - 2)
    print(f"Using {n_threads} worker threads")

    # Initialize models
    n_actions = len(MACRO_ACTIONS)
    agent = Agent(n_actions).to(device)  # Policy uses full 4-frame stack

    # Choose encoder based on USE_INVERSE_DYNAMICS flag
    if USE_INVERSE_DYNAMICS:
        phi_enc = TrainableEncoder(in_ch=4, out_dim=FEAT_DIM).to(device)
        idm_model = InverseDynamicsModel(FEAT_DIM, n_actions, hidden=IDM_HIDDEN).to(device)
        print("Using TRAINABLE encoder with Inverse Dynamics Model (ICM)")
    else:
        phi_enc = RandomEncoder(in_ch=4, out_dim=FEAT_DIM).to(device)
        idm_model = None
        print("Using FROZEN Random encoder")

    fwd_model = ForwardDynamicsModel(FEAT_DIM, n_actions, hidden=FWD_HIDDEN).to(device)

    # Optimizers (higher LR for forward model to learn dynamics faster)
    agent_optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    fwd_optimizer = torch.optim.Adam(fwd_model.parameters(), lr=FWD_LEARNING_RATE)

    # Optimizer for trainable encoder + IDM (if using)
    if USE_INVERSE_DYNAMICS:
        feat_idm_params = list(phi_enc.parameters()) + list(idm_model.parameters())
        feat_idm_optimizer = torch.optim.Adam(feat_idm_params, lr=FEAT_LEARNING_RATE)
    else:
        feat_idm_optimizer = None

    # Statistics trackers
    rwd_i_rms = RunningStats()
    rwd_e_rms = RunningStats()
    stats = (rwd_i_rms, rwd_e_rms)

    # Training loop
    for iter_idx in range(ITERS):
        print(f"\n=== Iteration {iter_idx+1}/{ITERS} ===")
        iter_start = time.time()

        # Collect rollouts with incremental processing
        buffer = Queue()
        frame_counter = FrameCounter()

        # Pick one random thread to save RGB debug frames
        debug_thread_id = random.randint(0, n_threads - 1)

        # Accumulate processed data incrementally to save memory
        all_obs = []
        all_obs_next = []
        all_acts = []
        all_logps = []
        all_advantages = []
        all_returns = []
        all_dones = []
        all_rwds_i_raw = []  # Store raw rewards for batch statistics
        all_rwds_e_raw = []
        debug_episode_saved = False
        total_episodes = 0

        while frame_counter.get() < MAX_ROLLOUT_FRAMES:
            run_parallel_episodes(n_threads, agent, fwd_model, phi_enc, buffer, stats,
                                frame_counter, debug_thread_id, device)

            # Process episodes immediately as they arrive (don't accumulate)
            while not buffer.empty():
                ep = buffer.get()
                thread_id, obss, obss_next, acts, logps, vals, rwds_e, rwds_i, dones, obss_rgb, obss_next_rgb = ep
                total_episodes += 1

                # Save debug frames from the selected thread (only once)
                if thread_id == debug_thread_id and not debug_episode_saved and obss_rgb is not None:
                    debug_dir = f'debug/iter_{iter_idx+1:04d}'
                    os.makedirs(debug_dir, exist_ok=True)
                    action_names = ['NOOP', 'FWD', 'BACK', 'TURN_L', 'TURN_R', 'FWD_L', 'FWD_R',
                                   'ATTACK', 'USE', 'FWD_ATK', 'STRAFE_L', 'STRAFE_R']

                    for i, (rgb_curr, rgb_next, act_idx, r_i, r_e) in enumerate(zip(obss_rgb, obss_next_rgb, acts, rwds_i, rwds_e)):
                        Image.fromarray(rgb_curr, mode='RGB').save(f'{debug_dir}/frame_{i:04d}_current.png')
                        Image.fromarray(rgb_next, mode='RGB').save(f'{debug_dir}/frame_{i:04d}_next.png')
                        action_name = action_names[act_idx] if act_idx < len(action_names) else f'ACT_{act_idx}'
                        with open(f'{debug_dir}/frame_{i:04d}_{action_name}.txt', 'w') as f:
                            f.write(f'Action: {action_name}\n')
                            f.write(f'Intrinsic Reward: {r_i:.6f}\n')
                            f.write(f'Extrinsic Reward: {r_e:.3f}\n')

                    print(f"Saved {len(obss_rgb)} RGB frame pairs to {debug_dir}/")
                    debug_episode_saved = True

                # Convert episode to tensors (uint8 for memory efficiency)
                obs_ep = torch.tensor(np.array(obss), dtype=torch.uint8)
                obs_next_ep = torch.tensor(np.array(obss_next), dtype=torch.uint8)
                acts_ep = torch.tensor(acts, dtype=torch.long)
                logps_ep = torch.tensor(logps, dtype=torch.float32)
                vals_ep = torch.tensor(vals, dtype=torch.float32)
                rwds_e_ep = torch.tensor(rwds_e, dtype=torch.float32)
                rwds_i_ep = torch.tensor(rwds_i, dtype=torch.float32)
                dones_ep = torch.tensor(dones, dtype=torch.bool)

                # Normalize rewards using running stats
                _, _, std_i = rwd_i_rms.get()
                _, _, std_e = rwd_e_rms.get()

                rwds_i_norm = rwds_i_ep / max(std_i if not math.isnan(std_i) else 1.0, 1e-3)
                rwds_e_norm = rwds_e_ep / max(std_e if not math.isnan(std_e) else 1.0, 1e-3)

                total_rwds_ep = EXTRINSIC_COEF * rwds_e_norm + INTRINSIC_COEF * rwds_i_norm

                # Bootstrap value from NEXT state
                with torch.no_grad():
                    if IGNORE_DONES or not dones_ep[-1]:
                        obs_next_last = obs_next_ep[-1].float().to(device).unsqueeze(0)
                        _, next_val = agent.forward(obs_next_last)
                        next_val = next_val[0].cpu()
                    else:
                        next_val = torch.tensor(0.0)

                # Compute GAE with IGNORE_DONES flag
                advs_ep, rets_ep = compute_gae(
                    total_rwds_ep.unsqueeze(1),
                    vals_ep.unsqueeze(1),
                    dones_ep.unsqueeze(1),
                    next_val.unsqueeze(0),
                    GAMMA_EXT,
                    GAE_LAMBDA,
                    ignore_dones=IGNORE_DONES
                )

                # Flatten and accumulate immediately
                all_obs.append(obs_ep)
                all_obs_next.append(obs_next_ep)
                all_acts.append(acts_ep)
                all_logps.append(logps_ep)
                all_advantages.append(advs_ep.squeeze(1))
                all_returns.append(rets_ep.squeeze(1))
                all_dones.append(dones_ep)

                # Store raw rewards for batch statistics
                all_rwds_i_raw.extend(rwds_i)
                all_rwds_e_raw.extend(rwds_e)

                # Free memory immediately
                del ep, obss, obss_next, obss_rgb, obss_next_rgb
        print(f"Collected {total_episodes} episodes, {frame_counter.get()} frames")

        # Concatenate all episodes (keep on CPU, uint8 for obs)
        obs_t = torch.cat(all_obs, dim=0)  # uint8
        obs_next_t = torch.cat(all_obs_next, dim=0)  # uint8
        acts_t = torch.cat(all_acts, dim=0)
        logps_t = torch.cat(all_logps, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        returns = torch.cat(all_returns, dim=0)
        dones_t = torch.cat(all_dones, dim=0)

        # PPO updates
        n_samples = len(obs_t)
        indices = np.arange(n_samples)

        # Track average losses
        total_pi_loss = 0
        total_v_loss = 0
        total_ent_loss = 0
        total_fwd_loss = 0
        total_idm_loss = 0
        n_updates = 0

        for epoch in range(EPOCHS):
            np.random.shuffle(indices)
            e = time.time()
            for start in range(0, n_samples, MINIBATCH_SIZE):
                end = min(start + MINIBATCH_SIZE, n_samples)
                mb_idx = indices[start:end]

                # PPO update (cast uint8 to float32 on GPU)
                mb_obs = obs_t[mb_idx].float().to(device)
                pi_loss, v_loss, ent_loss = ppo_update(
                    agent, agent_optimizer,
                    mb_obs, acts_t[mb_idx].to(device), logps_t[mb_idx].to(device),
                    advantages[mb_idx].to(device), returns[mb_idx].to(device)
                )

                # Forward dynamics update with correct next_obs (no cross-episode pairs)
                # Use full frame stack for phi encoding
                mb_obs_stack = obs_t[mb_idx].float().to(device)  # (B, 4, H, W)
                mb_obs_next_stack = obs_next_t[mb_idx].float().to(device)  # (B, 4, H, W)
                mb_not_terminal = (~dones_t[mb_idx]).float().to(device)
                mb_acts = acts_t[mb_idx].to(device)

                # Encode features
                phi_t = phi_enc(mb_obs_stack)
                phi_t_next = phi_enc(mb_obs_next_stack)

                # Compute both forward and inverse losses first, then update models
                # This prevents double-backpropagation through the same computational graph

                act_onehot = F.one_hot(mb_acts, n_actions).float()

                # Forward dynamics loss (train forward model to predict next features)
                phi_pred = fwd_model(phi_t.detach(), act_onehot)
                per_sample_fwd = F.mse_loss(phi_pred, phi_t_next.detach(), reduction='none').mean(dim=1)
                if IGNORE_DONES:
                    fwd_loss_val = per_sample_fwd.mean()
                else:
                    fwd_loss_val = (per_sample_fwd * mb_not_terminal).sum() / (mb_not_terminal.sum().clamp_min(1))

                # Inverse dynamics loss (train encoder to extract action-relevant features)
                idm_loss = 0.0
                if USE_INVERSE_DYNAMICS:
                    # Recompute features for inverse dynamics to ensure fresh graph
                    phi_idm_t = phi_enc(mb_obs_stack)
                    phi_idm_t_next = phi_enc(mb_obs_next_stack)

                    # Predict action from (phi_t, phi_t_next)
                    action_logits = idm_model(phi_idm_t, phi_idm_t_next)
                    idm_loss_val = F.cross_entropy(action_logits, mb_acts, reduction='none')
                    if IGNORE_DONES:
                        idm_loss_val = idm_loss_val.mean()
                    else:
                        idm_loss_val = (idm_loss_val * mb_not_terminal).sum() / (mb_not_terminal.sum().clamp_min(1))

                    # Forward dynamics loss for encoder (part of joint optimization)
                    phi_pred_enc = fwd_model(phi_idm_t, act_onehot)
                    per_sample_fwd_enc = F.mse_loss(phi_pred_enc, phi_idm_t_next.detach(), reduction='none').mean(dim=1)
                    if IGNORE_DONES:
                        fwd_loss_enc = per_sample_fwd_enc.mean()
                    else:
                        fwd_loss_enc = (per_sample_fwd_enc * mb_not_terminal).sum() / (mb_not_terminal.sum().clamp_min(1))

                    # Pathak's joint optimization: (1-β)L_I + βL_F
                    beta = 0.25
                    joint_loss_val = (1 - beta) * idm_loss_val + beta * fwd_loss_enc

                    # Backward for encoder + IDM (joint optimization)
                    feat_idm_optimizer.zero_grad(set_to_none=True)
                    joint_loss_val.backward()
                    feat_idm_optimizer.step()

                    idm_loss = idm_loss_val.item()

                # Backward for forward model only (separate computation)
                fwd_optimizer.zero_grad(set_to_none=True)
                fwd_loss_val.backward()
                fwd_optimizer.step()

                fwd_loss = fwd_loss_val.item()

                # Accumulate losses
                total_pi_loss += pi_loss
                total_v_loss += v_loss
                total_ent_loss += ent_loss
                total_fwd_loss += fwd_loss
                total_idm_loss += idm_loss
                n_updates += 1
            
            s = time.time()
            print(f"EPOCH {epoch} took {(s-e):0.2f}")
        # Average losses
        pi_loss = total_pi_loss / n_updates
        v_loss = total_v_loss / n_updates
        ent_loss = total_ent_loss / n_updates
        fwd_loss = total_fwd_loss / n_updates
        idm_loss = total_idm_loss / n_updates

        # Logging
        iter_time = time.time() - iter_start
        _, mean_i_running, std_i_running = rwd_i_rms.get()
        _, mean_e_running, std_e_running = rwd_e_rms.get()

        # Intrinsic reward statistics
        batch_mean_i = np.mean(all_rwds_i_raw) if all_rwds_i_raw else 0.0
        batch_std_i = np.std(all_rwds_i_raw) if all_rwds_i_raw else 0.0
        batch_max_i = np.max(all_rwds_i_raw) if all_rwds_i_raw else 0.0
        batch_min_i = np.min(all_rwds_i_raw) if all_rwds_i_raw else 0.0

        # Extrinsic reward statistics (always log even if INTRINSIC_ONLY)
        batch_mean_e = np.mean(all_rwds_e_raw) if all_rwds_e_raw else 0.0
        batch_std_e = np.std(all_rwds_e_raw) if all_rwds_e_raw else 0.0
        batch_max_e = np.max(all_rwds_e_raw) if all_rwds_e_raw else 0.0
        batch_sum_e = np.sum(all_rwds_e_raw) if all_rwds_e_raw else 0.0  # Total game rewards this iteration

        # Calculate total reward (weighted sum, accounting for INTRINSIC_ONLY)
        total_reward = EXTRINSIC_COEF * mean_e_running + INTRINSIC_COEF * mean_i_running
        batch_total = EXTRINSIC_COEF * batch_mean_e + INTRINSIC_COEF * batch_mean_i

        # Log to wandb
        log_dict = {
            "update_step": iter_idx + 1,
            # Running statistics (exponential moving average)
            "reward/intrinsic_running": mean_i_running,
            "reward/extrinsic_running": mean_e_running,
            "reward/intrinsic_std_running": std_i_running if not math.isnan(std_i_running) else 0,
            "reward/extrinsic_std_running": std_e_running if not math.isnan(std_e_running) else 0,
            # CURRENT BATCH intrinsic statistics
            "reward/intrinsic_batch_mean": batch_mean_i,
            "reward/intrinsic_batch_std": batch_std_i,
            "reward/intrinsic_batch_max": batch_max_i,
            "reward/intrinsic_batch_min": batch_min_i,
            # CURRENT BATCH extrinsic statistics (track even in INTRINSIC_ONLY mode!)
            "reward/extrinsic_batch_mean": batch_mean_e,
            "reward/extrinsic_batch_std": batch_std_e,
            "reward/extrinsic_batch_max": batch_max_e,
            "reward/extrinsic_batch_sum": batch_sum_e,  # Total game score this iteration
            # Total reward used for training
            "reward/total_batch": batch_total,
            "loss/policy": pi_loss,
            "loss/value": v_loss,
            "loss/entropy": -ent_loss,
            "loss/forward_dynamics": fwd_loss,
            "time/iteration_time": iter_time,
            "time/fps": frame_counter.get() / iter_time,
            "data/episodes_collected": total_episodes,
            "data/frames_collected": frame_counter.get(),
            }

        # Add IDM loss if using inverse dynamics
        if USE_INVERSE_DYNAMICS:
            log_dict["loss/inverse_dynamics"] = idm_loss

        wandb.log(log_dict)

        print(f"Time: {iter_time:.1f}s | FPS: {frame_counter.get()/iter_time:.0f}")
        print(f"Policy Loss: {pi_loss:.4f}, Value Loss: {v_loss:.4f}, Entropy: {-ent_loss:.4f}")
        print(f"Forward Dynamics Loss: {fwd_loss:.6f}")
        if USE_INVERSE_DYNAMICS:
            print(f"Inverse Dynamics Loss: {idm_loss:.6f}")
        print(f"Intrinsic (batch): μ={batch_mean_i:.6f}, σ={batch_std_i:.6f}, max={batch_max_i:.6f}")
        print(f"Extrinsic (batch): μ={batch_mean_e:.3f}, max={batch_max_e:.3f}, sum={batch_sum_e:.1f}")
        if INTRINSIC_ONLY:
            print(f"  ^ Note: Training with INTRINSIC_ONLY (extrinsic not used for learning)")

        # Save checkpoint
        if (iter_idx + 1) % 25 == 0:
            checkpoint = {
                'iteration': iter_idx + 1,
                'agent_state_dict': agent.state_dict(),
                'fwd_model_state_dict': fwd_model.state_dict(),
                'agent_optimizer': agent_optimizer.state_dict(),
                'fwd_optimizer': fwd_optimizer.state_dict(),
            }
            torch.save(checkpoint, f'idm_checkpoint_{iter_idx+1}.pth')
            print(f"Saved checkpoint at iteration {iter_idx+1}")

    print("\nTraining complete!")
    wandb.finish()
