import os
import math
import time
import torch
import json
import random
import threading
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import vizdoom as vzd
import numpy as np
import wandb

from datetime import datetime
from PIL import Image
from queue import Queue
from model import (
    RNDAgent, 
    RunningStats,
)

ITERS = 1000
EPOCHS = 2  
FRAME_SKIP = 4
FRAME_STACK = 4  # Stack last 4 frames as in the paper
MAX_ROLLOUT_FRAMES = FRAME_SKIP * int(4e4)  # Reduced to avoid OOM
MAX_FRAMES_PER_EPISODE = int(37.5*60*2)  # Per-thread 1 minute episode limit

CLIP_COEF = 0.1
ENT_COEF_START = 0.10
ENT_COEF_END = 0.001
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
MINIBATCH_SIZE = 2048
LEARNING_RATE = 1e-4

GAMMA_INT = 0.99
GAMMA_EXT = 0.999
GAE_LAMBDA = 0.95
INTRINSIC_COEF = 1.0
EXTRINSIC_COEF = 0.0

# "Death is not the end" - treat episode boundaries as non-terminal for GAE
IGNORE_DONES = True

EPSILON_CLAMP = 1e-6
OBS_HEIGHT = 60
OBS_WIDTH = 80
OBS_CHANNELS = FRAME_STACK
OBS_SIZE = OBS_CHANNELS * OBS_HEIGHT * OBS_WIDTH
OBS_RANDOM_WARMUP_STEPS = 10000
OBS_CLIP_MIN = -5.0
OBS_CLIP_MAX = 5.0

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

BUTTONS_ORDER = sorted(
    {btn for action in MACRO_ACTIONS for btn in action},
    key=lambda b: b.value
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

cpus = multiprocessing.cpu_count()
n_threads = max(1, cpus - 2)
print(f"Using {n_threads} worker threads")

n_actions = len(MACRO_ACTIONS)

torch.autograd.set_detect_anomaly(True)
os.makedirs('debug', exist_ok=True)
debug_dir = 'debug/'+datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(debug_dir, exist_ok=True)

wandb_config = {
        "iters": ITERS,
        "epochs": EPOCHS,
        "max_rollout_frames": MAX_ROLLOUT_FRAMES,
        "max_frames_per_episode": MAX_FRAMES_PER_EPISODE,
        "gamma_int": GAMMA_INT,
        "gamma_ext": GAMMA_EXT,
        "gae_lambda": GAE_LAMBDA,
        "clip_coef": CLIP_COEF,
        "ent_coef_start": ENT_COEF_START,
        "ent_coef_end": ENT_COEF_END,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "minibatch_size": MINIBATCH_SIZE,
        "policy_lr": LEARNING_RATE,
        "ignore_dones": IGNORE_DONES,
        "intrinsic_coef": INTRINSIC_COEF,
        "extrinsic_coef": EXTRINSIC_COEF,
        "frame_skip": FRAME_SKIP,
        "frame_stack": FRAME_STACK,
        "n_actions": len(MACRO_ACTIONS),
}

def init_wandb(config):
    """Initialize wandb with support for disabled/offline modes."""
    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower()
    disable = wandb_mode in {"disabled", "offline"} or wandb_disabled in {"true", "1", "yes"}
    if disable:
        return wandb.init(mode="disabled", config=config, project="doom-idm-curiosity")
    return wandb.init(project="doom-idm-curiosity", config=config)

def rgb_to_grayscale(rgb_frame):
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

class RandomAgent:
    """Uniformly samples a macro action index for warmup."""
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def act(self) -> int:
        return random.randint(0, self.n_actions - 1)

def warmup_observation_stats(ob_rms):
    """Collect observations with a random policy to initialize ob_rms."""
    game = vzd.DoomGame()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wad_path = os.path.join(script_dir, "Doom1.WAD")
    game.set_doom_game_path(wad_path)
    game.set_doom_map("E1M1")
    game.set_render_hud(False)
    game.set_death_reward(-0.5)
    game.set_kill_reward(10.0)
    game.set_armor_reward(0.0)
    game.set_health_reward(0.0)
    game.set_map_exit_reward(100.0)
    game.set_secret_reward(50.0)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args("+freelook 1")
    game.set_available_buttons(BUTTONS_ORDER)
    game.init()
    game.new_episode()

    rand_agent = RandomAgent(len(MACRO_ACTIONS))
    steps = 0
    while steps < OBS_RANDOM_WARMUP_STEPS:
        if game.is_episode_finished():
            game.new_episode()
            continue
        state = game.get_state()
        if state is None:
            continue
        raw_frame = state.screen_buffer  # (3, H, W)
        gray = resize_gray_frame(rgb_to_grayscale(raw_frame))  # (H, W)
        gray01 = (gray.astype(np.float32) / 255.0)
        sample = torch.tensor(gray01[..., None], dtype=torch.float32, device=device)
        ob_rms.update(sample)

        act_idx = rand_agent.act()
        action_buttons = MACRO_ACTIONS[act_idx]
        action_vector = [btn in action_buttons for btn in BUTTONS_ORDER]
        game.make_action(action_vector, FRAME_SKIP)
        steps += 1

    game.close()
       
def run_episode(thread_id, agent, buffer, frame_counter,
                save_rgb_debug, device='cpu', ob_rms=None):
    game = vzd.DoomGame()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wad_path = os.path.join(script_dir, "Doom1.WAD")
    game.set_doom_game_path(wad_path)
    game.set_doom_map("E1M1")
    game.set_render_hud(False)
    #game.set_hit_reward(1.0) #hitting barrels also give reward, so disable this
    game.set_death_reward(-0.5)
    game.set_kill_reward(10.0)
    game.set_armor_reward(0.0)
    game.set_health_reward(0.0)
    game.set_map_exit_reward(100.0)
    game.set_secret_reward(50.0)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args("+freelook 1")
    game.set_available_buttons(BUTTONS_ORDER)
    game.init()
    game.new_episode()

    obss = []  # Stacked frames (4, H, W) uint8 grayscale
    obss_rgb = [] if save_rgb_debug else None
    acts, logps, vals_int, vals_ext, rwds_e, dones = [], [], [], [], [], []

    frames_collected = 0
    frame_buffer = []

    while frames_collected < MAX_FRAMES_PER_EPISODE:
        if game.is_episode_finished():
            game.new_episode()
            frame_buffer = []
            continue

        state = game.get_state()
        if state is None:
            continue

        raw_frame = state.screen_buffer  # (3, H, W) RGB uint8 CHW format
        raw_frame_hwc = raw_frame.transpose(1, 2, 0)  # Convert to (H, W, 3) for saving
        frame_buffer = stack_frames(frame_buffer, raw_frame)
        obs_stacked_np = np.stack(frame_buffer, axis=0)  # (4, H, W) grayscale uint8
        # Normalize to [0,1] for policy/value networks (no whitening per paper guidance)
        obs_img = (obs_stacked_np.astype(np.float32) / 255.0)
        obs_t_policy = torch.tensor(obs_img, dtype=torch.float32, device=device).unsqueeze(0)  # (1,C,H,W)

        with torch.no_grad():
            action, _, logp, v_int, v_ext = agent.step(obs_t_policy)
            act_idx_item = int(action)

        action_buttons = MACRO_ACTIONS[act_idx_item]
        action_vector = [btn in action_buttons for btn in BUTTONS_ORDER]
        rwd_e = game.make_action(action_vector, FRAME_SKIP)
        done_flag = game.is_episode_finished()

        obss.append(obs_stacked_np.astype(np.uint8))
        if save_rgb_debug: obss_rgb.append(raw_frame_hwc)

        acts.append(act_idx_item)
        logps.append(float(logp))
        vals_int.append(float(v_int))
        vals_ext.append(float(v_ext))
        rwds_e.append(rwd_e)
        dones.append(done_flag)

        frames_collected += FRAME_SKIP

    with frame_counter:
        frame_counter.count += len(obss) * FRAME_SKIP

    game.close()

    # Compute intrinsic rewards from normalized+clipped image observations for RND
    obss_np = np.array(obss, dtype=np.uint8)  # (T, 4, H, W)
    obs_img = (obss_np.astype(np.float32) / 255.0)
    obs_t_img = torch.as_tensor(obs_img, dtype=torch.float32, device=device)  # (T,C,H,W)
    if ob_rms is not None:
        _, mean_hw1, std_hw1 = ob_rms.get()
        mean_hw1_t = mean_hw1 if isinstance(mean_hw1, torch.Tensor) else torch.tensor(mean_hw1)
        std_hw1_t = std_hw1 if isinstance(std_hw1, torch.Tensor) else torch.tensor(std_hw1)
        mean_hw1_t = mean_hw1_t.to(dtype=torch.float32, device=device)
        std_hw1_t = torch.clamp(torch.as_tensor(std_hw1_t, dtype=torch.float32, device=device), min=1e-6)
        mean_hw = mean_hw1_t[..., 0]  # (H,W)
        std_hw = std_hw1_t[..., 0]
        mean_chw = mean_hw.unsqueeze(0).repeat(OBS_CHANNELS, 1, 1)
        std_chw = std_hw.unsqueeze(0).repeat(OBS_CHANNELS, 1, 1)
        obs_t_img = torch.clamp((obs_t_img - mean_chw) / std_chw, min=OBS_CLIP_MIN, max=OBS_CLIP_MAX)
    rwds_i_t = agent.rnd.intrinsic_reward(obs_t_img)
    rwds_i = rwds_i_t.detach().cpu().numpy().tolist()

    buffer.put((thread_id, obss, acts, logps, vals_int, vals_ext, rwds_e, rwds_i, dones, obss_rgb))

class FrameCounter:
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

def run_parallel_episodes(n_threads, agent, buffer, frame_counter,
                         debug_thread_id, device='cpu', ob_rms=None):
    threads = []
    for tid in range(n_threads):
        save_rgb = (tid == debug_thread_id)  # Only one thread saves RGB
        t = threading.Thread(
            target=run_episode,
            args=(tid, agent, buffer, frame_counter, save_rgb, device, ob_rms)
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def ppo_update(agent, optimizer, obs_policy_mb, obs_rnd_mb , act_mb, old_logp_mb,
               adv_mb, ret_int_mb, ret_ext_mb, clip_coef=CLIP_COEF, 
               ent_coef=0.01, vf_coef=VF_COEF):

    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
    logits, v_int_new, v_ext_new = agent.forward(obs_policy_mb)
    dist = torch.distributions.Categorical(logits=logits)
    # FIXED: Negative log-prob / log-prob sign conventions and ratio
    logp_new_mb = dist.log_prob(act_mb)
    entropy = dist.entropy().mean()
    ratio = (logp_new_mb - old_logp_mb).exp()

    unclipped = ratio * adv_mb
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_mb
    pi_loss = -torch.min(unclipped, clipped).mean()

    v_loss_int = 0.5 * F.mse_loss(v_int_new, ret_int_mb)
    v_loss_ext = 0.5 * F.mse_loss(v_ext_new, ret_ext_mb)
    v_loss = v_loss_int + v_loss_ext

    ent_loss = -entropy

    rnd_loss = agent.rnd.predictor_loss(obs_rnd_mb).mean()

    total_loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss + rnd_loss
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    return pi_loss.item(), v_loss.item(), v_loss_int.item(), v_loss_ext.item(), ent_loss.item(), rnd_loss.item()

def compute_gae_1d(rewards: torch.Tensor,
                   values: torch.Tensor,
                   dones: torch.Tensor,
                   gamma: float,
                   gae_lambda: float,
                   ignore_dones: bool = False) -> (torch.Tensor, torch.Tensor):
    """Compute GAE for a single episode sequence.
    rewards, values, dones: shape (T,)
    Returns: advantages, returns of shape (T,)
    Note: we keep zero-bootstrap at the sequence end. If ignore_dones=True,
    we bypass terminal gating within the sequence only (no extra bootstrap).
    """
    T = rewards.shape[0]
    device = rewards.device
    advantages = torch.zeros(T, dtype=torch.float32, device=device)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nextvalues = torch.tensor(0.0, dtype=torch.float32, device=device)
            if ignore_dones:
                nextnonterminal = 1.0
            else:
                nextnonterminal = 0.0 if (dones[t].float() > 0.5) else 1.0
        else:
            nextvalues = values[t+1]
            if ignore_dones:
                nextnonterminal = 1.0
            else:
                nextnonterminal = 0.0 if (dones[t+1].float() > 0.5) else 1.0
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns

def compute_gae(rewards, values, dones, gamma, gae_lambda, ignore_dones: bool = False):
    T, N = rewards.shape
    device = rewards.device
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=device)
    for t in reversed(range(T)):
        if t == T - 1:
            nextvalues = values[:, t]
            if ignore_dones:
                nextnonterminal = torch.ones(N, device=device)
            else:
                nextnonterminal = 1.0 - (dones[:, t])
        else:
            nextvalues = values[:, t+1]
            if ignore_dones:
                nextnonterminal = torch.ones(N, device=device)
            else:
                nextnonterminal = 1.0 - (dones[:, t+1])
        delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values[:,    t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[:, t] = lastgaelam

    returns = advantages + values
    return advantages, returns

if __name__ == '__main__':
    init_wandb(wandb_config)
    agent = RNDAgent(OBS_SIZE, n_actions, hidden=(256,256), rnd_feat_dim=128, conv_out_dim=128).to(device)
    agent_optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    rff_rms = RunningStats()
    # Observation RunningStats for last-frame normalization: shape (H, W, 1)
    ob_rms = RunningStats(shape=(OBS_HEIGHT, OBS_WIDTH, 1), dtype=torch.float32, device=device)

    # Initialize observation normalization parameters using a random agent
    print("Warming up observation normalization with random policy...")
    warmup_observation_stats(ob_rms)
    print("Warmup complete.")

    for iter_idx in range(ITERS):
        print(f"\n=== Iteration {iter_idx+1}/{ITERS} ===")
        iter_start = time.time()

        buffer = Queue()
        frame_counter = FrameCounter()
        debug_thread_id = random.randint(0, n_threads - 1)

        debug_episode_saved = False
        total_episodes = 0
        rollout_data = {
            "obs": [],
            "acts": [],
            "logps": [],
            "vals_int": [],
            "vals_ext": [],
            "rwds_e": [],
            "rwds_i": [],
            "dones": [],
            "rollout_rwds_i_raw": [],
            "rollout_rwds_e_raw": [],
        }

        while frame_counter.get() < MAX_ROLLOUT_FRAMES:
            run_parallel_episodes(
                n_threads, agent, buffer, frame_counter, debug_thread_id, device, ob_rms
            )

            while not buffer.empty():
                ep = buffer.get()
                (thread_id, obss, acts, logps, vals_int, vals_ext,
                 rwds_e, rwds_i, dones,
                 obss_rgb) = ep
                total_episodes += 1

                if thread_id == debug_thread_id and not debug_episode_saved and obss_rgb is not None:
                    save_frame_dir = f'{debug_dir}/iter_{iter_idx+1:04d}'
                    os.makedirs(save_frame_dir)
                    action_names = ['NOOP', 'FWD', 'BACK', 'TURN_L', 'TURN_R', 'FWD_L', 'FWD_R',
                                    'ATTACK', 'USE', 'FWD_ATK', 'STRAFE_L', 'STRAFE_R']
                    for i, (rgb_curr, act_idx, r_i, r_e) in enumerate(
                        zip(obss_rgb, acts, rwds_i, rwds_e)
                    ):
                        Image.fromarray(rgb_curr, mode='RGB').save(f'{save_frame_dir}/frame_{i:04d}_current.png')
                        action_name = action_names[act_idx] if act_idx < len(action_names) else f'ACT_{act_idx}'
                        with open(f'{save_frame_dir}/frame_{i:04d}_{action_name}.txt', 'w') as f:
                            f.write(f'Action: {action_name}\\n')
                            f.write(f'Intrinsic Reward: {r_i:.6f}\\n')
                            f.write(f'Extrinsic Reward: {r_e:.3f}\n')

                    debug_episode_saved = True

                # FIXED: don't flatline everything 
                rollout_data["obs"].append(obss)
                rollout_data["acts"].append(acts)
                rollout_data["logps"].append(logps)
                rollout_data["vals_int"].append(vals_int)
                rollout_data["vals_ext"].append(vals_ext)
                rollout_data["rwds_e"].append(rwds_e)
                rollout_data["rwds_i"].append(rwds_i)
                rollout_data["dones"].append(dones)
                rollout_data["rollout_rwds_i_raw"].append(rwds_i)
                rollout_data["rollout_rwds_e_raw"].append(rwds_e)
                del ep, obss, obss_rgb

        print(f"Collected {total_episodes} episodes, {frame_counter.get()} frames")

        # Update observation running stats with all last-frames from this rollout
        # This keeps whitening parameters current before building the training dataset
        for ep_idx in range(len(rollout_data["obs"])):
            obss_ep = np.array(rollout_data["obs"][ep_idx], dtype=np.uint8)  # (T, 4, H, W)
            if obss_ep.shape[0] == 0:
                continue
            last_frames = obss_ep[:, -1, :, :]  # (T, H, W) last channel is most recent
            last_frames_norm = (last_frames.astype(np.float32) / 255.0)
            batch = torch.tensor(last_frames_norm[..., None], dtype=torch.float32, device=device)  # (T, H, W, 1)
            ob_rms.update(batch)

        # Build reward forward filter over all episodes (scalar running stats)
        filtered_intrinsic_all = []
        for rw_i_ep in rollout_data["rwds_i"]:
            rff_state_scalar = 0.0
            for r in rw_i_ep:
                rff_state_scalar = rff_state_scalar * GAMMA_INT + float(r)
                filtered_intrinsic_all.append(rff_state_scalar)
        if len(filtered_intrinsic_all) > 0:
            rff_rms.update(torch.tensor(filtered_intrinsic_all, dtype=torch.float32))
        _, mean_i_running, std_i_running = rff_rms.get()
        std_i_running = float(std_i_running) if isinstance(std_i_running, (float, int)) else float(torch.as_tensor(std_i_running).item())
        if not np.isfinite(std_i_running) or std_i_running <= 0.0:
            std_i_running = 1.0

        # Per-episode GAE, then flatten transitions to build dataset
        flat_obs_policy_list = []
        flat_obs_rnd_list = []
        flat_acts_list = []
        flat_oldlogp_list = []
        flat_adv_list = []
        flat_ret_i_list = []
        flat_ret_e_list = []
        flat_rwds_i_norm = []
        flat_rwds_e_raw = []

        for ep_idx in range(len(rollout_data["obs"])):
            # observations
            obss_ep = np.array(rollout_data["obs"][ep_idx], dtype=np.uint8)  # (T, 4, H, W)
            T = obss_ep.shape[0]
            if T == 0:
                continue
            # Build two views of observations:
            # - Policy/value: scaled to [0,1] only (no whitening)
            # - RND: whitened using running stats and clipped to [-5, 5]
            obs_img = (obss_ep.astype(np.float32) / 255.0)
            obs_policy_ep = torch.tensor(obs_img, dtype=torch.float32)

            _, mean_hw1, std_hw1 = ob_rms.get()
            mean_hw1_t = mean_hw1 if isinstance(mean_hw1, torch.Tensor) else torch.tensor(mean_hw1)
            std_hw1_t = std_hw1 if isinstance(std_hw1, torch.Tensor) else torch.tensor(std_hw1)
            mean_hw1_t = mean_hw1_t.to(dtype=torch.float32)
            std_hw1_t = torch.clamp(torch.as_tensor(std_hw1_t, dtype=torch.float32), min=1e-6)
            mean_hw = mean_hw1_t[..., 0]  # (H,W)
            std_hw = std_hw1_t[..., 0]
            mean_chw = mean_hw.unsqueeze(0).repeat(OBS_CHANNELS, 1, 1)
            std_chw = std_hw.unsqueeze(0).repeat(OBS_CHANNELS, 1, 1)
            obs_rnd_ep = torch.clamp((torch.tensor(obs_img, dtype=torch.float32) - mean_chw) / std_chw, min=OBS_CLIP_MIN, max=OBS_CLIP_MAX)
            acts_ep = torch.tensor(rollout_data["acts"][ep_idx], dtype=torch.long)
            logps_ep = torch.tensor(rollout_data["logps"][ep_idx], dtype=torch.float32)
            vals_i_ep = torch.tensor(rollout_data["vals_int"][ep_idx], dtype=torch.float32)
            vals_e_ep = torch.tensor(rollout_data["vals_ext"][ep_idx], dtype=torch.float32)
            dones_ep = torch.tensor(rollout_data["dones"][ep_idx], dtype=torch.bool)
            rwds_e_ep = torch.tensor(rollout_data["rwds_e"][ep_idx], dtype=torch.float32)
            rwds_i_raw_ep = torch.tensor(rollout_data["rwds_i"][ep_idx], dtype=torch.float32)

            # normalize intrinsic rewards by running std of filtered intrinsic
            rwds_i_norm_ep = rwds_i_raw_ep / std_i_running

            adv_e_ep, ret_e_ep = compute_gae_1d(rwds_e_ep, vals_e_ep, dones_ep, GAMMA_EXT, GAE_LAMBDA, ignore_dones=IGNORE_DONES)
            adv_i_ep, ret_i_ep = compute_gae_1d(rwds_i_norm_ep, vals_i_ep, dones_ep, GAMMA_INT, GAE_LAMBDA, ignore_dones=IGNORE_DONES)
            adv_total_ep = EXTRINSIC_COEF * adv_e_ep + INTRINSIC_COEF * adv_i_ep

            flat_obs_policy_list.append(obs_policy_ep)
            flat_obs_rnd_list.append(obs_rnd_ep)
            flat_acts_list.append(acts_ep)
            flat_oldlogp_list.append(logps_ep)
            flat_adv_list.append(adv_total_ep)
            flat_ret_i_list.append(ret_i_ep)
            flat_ret_e_list.append(ret_e_ep)
            flat_rwds_i_norm.extend(rwds_i_norm_ep.tolist())
            flat_rwds_e_raw.extend(rwds_e_ep.tolist())

        if len(flat_obs_policy_list) == 0:
            print("No samples collected; skipping update this iteration.")
            continue

        # Concatenate across episodes
        rollout_obs_policy = torch.cat(flat_obs_policy_list, dim=0).to(device)
        rollout_obs_rnd = torch.cat(flat_obs_rnd_list, dim=0).to(device)
        rollout_acts = torch.cat(flat_acts_list, dim=0).to(device)
        rollout_logps = torch.cat(flat_oldlogp_list, dim=0).to(device)
        rollout_advs = torch.cat(flat_adv_list, dim=0).to(device)
        rollout_rets_i = torch.cat(flat_ret_i_list, dim=0).to(device)
        rollout_rets_e = torch.cat(flat_ret_e_list, dim=0).to(device)

        # normalize advantages
        rollout_advs = (rollout_advs - rollout_advs.mean()) / (rollout_advs.std() + 1e-8)

        n_samples = rollout_obs_policy.shape[0]
        indices = np.arange(n_samples)

        total_pi_loss, total_v_loss, total_v_loss_i, total_v_loss_e, total_ent_loss, total_rnd_loss, n_updates = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for epoch in range(EPOCHS):
            np.random.shuffle(indices)
            epoch_start = time.time()
            # exponential entropy decay schedule
            ENT_COEF = ENT_COEF_END + (ENT_COEF_START - ENT_COEF_END) * math.exp(-3.0 * (iter_idx / max(ITERS - 1, 1)))
            for start in range(0, n_samples, MINIBATCH_SIZE):
                end = min(start + MINIBATCH_SIZE, n_samples)
                mb_idx = indices[start:end]
                mb_obs_policy = rollout_obs_policy[mb_idx]
                mb_obs_rnd = rollout_obs_rnd[mb_idx]
                mb_acts = rollout_acts[mb_idx]
                mb_logps = rollout_logps[mb_idx]
                mb_advs = rollout_advs[mb_idx]
                mb_rets_i = rollout_rets_i[mb_idx]
                mb_rets_e = rollout_rets_e[mb_idx]

                pi_loss, v_loss, v_loss_i, v_loss_e, ent_loss, rnd_loss = ppo_update(
                    agent, 
                    agent_optimizer, 
                    mb_obs_policy,
                    mb_obs_rnd, 
                    mb_acts, 
                    mb_logps, 
                    mb_advs, 
                    mb_rets_i, 
                    mb_rets_e, 
                    ent_coef=ENT_COEF
                )

                total_pi_loss += pi_loss
                total_v_loss += v_loss
                total_v_loss_i += v_loss_i  
                total_v_loss_e += v_loss_e
                total_ent_loss += ent_loss
                total_rnd_loss += rnd_loss
                n_updates += 1

            epoch_end = time.time()
            print(f"EPOCH {epoch} took {(epoch_end - epoch_start):0.2f}s")

        pi_loss = total_pi_loss / max(n_updates, 1)
        v_loss = total_v_loss / max(n_updates, 1)
        v_loss_i = total_v_loss_i / max(n_updates, 1)
        v_loss_e = total_v_loss_e / max(n_updates, 1)
        ent_loss = total_ent_loss / max(n_updates, 1)
        rnd_loss = total_rnd_loss / max(n_updates, 1)

        iter_time = time.time() - iter_start
        _, mean_i_running, std_i_running = rff_rms.get()
        batch_mean_i = float(np.mean(flat_rwds_i_norm)) if len(flat_rwds_i_norm) > 0 else 0.0
        batch_std_i = float(np.std(flat_rwds_i_norm)) if len(flat_rwds_i_norm) > 0 else 0.0
        batch_max_i = float(np.max(flat_rwds_i_norm)) if len(flat_rwds_i_norm) > 0 else 0.0
        batch_min_i = float(np.min(flat_rwds_i_norm)) if len(flat_rwds_i_norm) > 0 else 0.0

        batch_mean_e = float(np.mean(flat_rwds_e_raw)) if len(flat_rwds_e_raw) > 0 else 0.0
        batch_std_e = float(np.std(flat_rwds_e_raw)) if len(flat_rwds_e_raw) > 0 else 0.0

        total_reward = EXTRINSIC_COEF * batch_mean_e + INTRINSIC_COEF * batch_mean_i
        batch_total = total_reward

        log_dict = {
            "update_step": iter_idx + 1,
            "reward/intrinsic_batch_mean": batch_mean_i,
            "reward/extrinsic_batch_mean": batch_mean_e,
            "loss/policy": pi_loss,
            "loss/rnd": rnd_loss,
            "loss/value": v_loss,
            "loss/value_i": v_loss_i,
            "loss/value_e": v_loss_e,
            "loss/entropy": -ent_loss,
            "reward/intrinsic_running": float(mean_i_running),
            "reward/extrinsic_running": batch_mean_e,
            "reward/intrinsic_std_running": float(std_i_running) if not math.isnan(float(std_i_running)) else 0.0,
            "reward/extrinsic_std_running": batch_std_e if 'batch_std_e' in locals() else 0.0,
            "reward/intrinsic_batch_std": batch_std_i,
            "reward/intrinsic_batch_max": batch_max_i,
            "reward/intrinsic_batch_min": batch_min_i,
            "reward/total_batch": batch_total,
            "time/iteration_time": iter_time,
            "time/fps": frame_counter.get() / iter_time,
            "data/episodes_collected": total_episodes,
            "data/frames_collected": frame_counter.get(),
        }

        with open(f'{debug_dir}/log_{iter_idx:04d}.json', 'w') as f:
            json.dump(log_dict, f)

        for k,v in log_dict.items():
            print(k, ": ", v)
        
        wandb.log(log_dict)
        print(f"Timer {iter_time:.1f}s | FPS: {frame_counter.get()/iter_time:.0f}")
        print(f"Policy Loss: {pi_loss:.4f}, Value Loss: {v_loss:.4f}, Entropy: {-ent_loss:.4f}")
        print(f"RND Loss: {rnd_loss:.4f}")
        print("--------------------------------")
        print("Iteration Rollout Rewards:")
        # Intrinsic normalized stats (across all transitions)
        rwds_i_norm_all = [ri / (std_i_running if std_i_running > 0 else 1.0) for ri in flat_rwds_i_norm]
        if len(rwds_i_norm_all) > 0:
            print(f"Intrinsic scaled: μ={float(np.mean(rwds_i_norm_all)):.3f}, max={float(np.max(rwds_i_norm_all)):.3f}, sum={float(np.sum(rwds_i_norm_all)):.1f}")
        print(f"Extrinsic raw: μ={batch_mean_e}")
        os.makedirs(f'{debug_dir}/checkpoints', exist_ok=True)
        if (iter_idx + 1) % 25 == 0:
            checkpoint = {
                'iteration': iter_idx + 1,
                'agent_state_dict': agent.state_dict(),
                'agent_optimizer': agent_optimizer.state_dict(),
            }
            torch.save(checkpoint, f'{debug_dir}/checkpoints/icm_rnd_checkpoint_{iter_idx+1}.pth')
    wandb.finish()
