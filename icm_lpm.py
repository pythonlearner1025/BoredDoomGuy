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

from PIL import Image
from queue import Queue
from model import (
    Agent, 
    LPMDynamicsModel, 
    LPMErrorModel, 
    EpisodicNoveltyBuffer,
    RandomEncoder, 
    DynamicsReplayBuffer, 
    ErrorReplayBuffer, 
    RunningStats,
)

ITERS = 1000
EPOCHS = 3  
FRAME_SKIP = 4
FRAME_STACK = 4  # Stack last 4 frames as in the paper
MAX_ROLLOUT_FRAMES = FRAME_SKIP * int(4e4)  # Reduced to avoid OOM
MAX_FRAMES_PER_EPISODE = int(37.5*60*3)  # Per-thread 1 minute episode limit
GAMMA_INT = 0.99
GAMMA_EXT = 0.999
GAE_LAMBDA = 0.95
CLIP_COEF = 0.1
ENT_COEF_START = 0.03
ENT_COEF_END = 0.001
ENT_COEF = ENT_COEF_START
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
MINIBATCH_SIZE = 2048
LEARNING_RATE = 1e-4

INTRINSIC_ONLY = False  # If True, train with only intrinsic reward (no extrinsic)
INTRINSIC_COEF = 1.0 if INTRINSIC_ONLY else 0.1
EXTRINSIC_COEF = 0.0 if INTRINSIC_ONLY else 0.9  # Allow sparse extrinsic reward to break ties
IGNORE_DONES = INTRINSIC_ONLY  # "Death is not the end" - infinite horizon bootstrapping
ENCODER_FEATS = 512

# Novelty
NOVELTY_SIMILARITY_THRESH = 0.9 # lower => ADHD 
LP_WEIGHT = 1.0
EPISODIC_WEIGHT = 0.0
SKIP_FRAMES_NOVELTY = 15 if EPISODIC_WEIGHT > 0 else 1e6 # every 0.5 seconds 

# LPM specific replay/buffer parameters
ERROR_BUFFER_SIZE = (MAX_ROLLOUT_FRAMES * 5) // FRAME_SKIP
DYNAMICS_REPLAY_SIZE = (MAX_ROLLOUT_FRAMES * 5) // FRAME_SKIP
ERROR_WARMUP = ERROR_BUFFER_SIZE // 2

DYNAMICS_LEARNING_RATE = 1e-4
ERROR_LEARNING_RATE = 1e-4
DYNAMICS_GRAD_STEPS = 10
ERROR_GRAD_STEPS = 15

LPM_HIDDEN = 512
DYNAMICS_BATCH_SIZE = 512
ERROR_BATCH_SIZE = 512

EPSILON_CLAMP = 1e-8
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

BUTTONS_ORDER = sorted(
    {btn for action in MACRO_ACTIONS for btn in action},
    key=lambda b: b.value
)

def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    T, N = rewards.shape
    device = rewards.device
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=device)
    for t in reversed(range(T)):
        if t == T - 1:
            nextvalues = next_value
        else:
            nextvalues = values[t+1]

        if IGNORE_DONES:
            nextnonterminal = torch.ones(N, device=device)
        else:
            nextnonterminal = (~dones[t]).float()

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values
    return advantages, returns

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
       
def run_episode(thread_id, agent, dynamics_model, error_model, encoder_model, buffer, stats, frame_counter,
                error_ready, save_rgb_debug, device='cpu'):
    game = vzd.DoomGame()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wad_path = os.path.join(script_dir, "Doom1.WAD")
    game.set_doom_game_path(wad_path)
    game.set_doom_map("E1M1")
    game.set_render_hud(False)
    #game.set_hit_reward(1.0) #hitting barrels also give reward, so disable this
    game.set_death_reward(-1.0)
    game.set_kill_reward(1.0)
    game.set_armor_reward(0.0)
    game.set_health_reward(0.0)
    game.set_map_exit_reward(10.0)
    game.set_secret_reward(5.0)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args("+freelook 1")
    game.set_available_buttons(BUTTONS_ORDER)
    game.init()
    game.new_episode()
    novelty_buffer = EpisodicNoveltyBuffer(embedding_dim=ENCODER_FEATS, similarity_threshold=NOVELTY_SIMILARITY_THRESH, device=device)
    novelty_buffer.reset()

    obss, obss_next = [], []  # Stacked frames (4, H, W) uint8 grayscale
    obss_rgb = [] if save_rgb_debug else None
    obss_next_rgb = [] if save_rgb_debug else None
    acts, logps, vals, rwds_e, rwds_i, novelties, learning_progresses, dones, epsilons = [], [], [], [], [], [], [], [], []

    run_rwd_i, run_rwd_e = 0, 0
    n_actions = len(MACRO_ACTIONS)
    frames_collected = 0
    frame_buffer = []
    novelty_counter = SKIP_FRAMES_NOVELTY - 1  
    last_novelty = 1.0

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
        obs_t = torch.tensor(obs_stacked_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            logits, val = agent.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            act_idx = dist.sample()
            logp = dist.log_prob(act_idx)

        act_idx_item = act_idx.item()
        act_onehot = F.one_hot(act_idx, n_actions).float().unsqueeze(0)

        with torch.no_grad():
            obs_stack_tensor = torch.tensor(obs_stacked_np, dtype=torch.float32, device=device).unsqueeze(0)
            phi_t = encoder_model(obs_stack_tensor)

        novelty_counter += 1
        if novelty_counter >= SKIP_FRAMES_NOVELTY:
            phi_features = phi_t.squeeze(0)
            episodic_novelty = novelty_buffer.compute_novelty(phi_features)
            novelty_buffer.add(phi_features)
            phi_cpu = phi_features.detach().cpu().numpy()
            norm = np.linalg.norm(phi_cpu)
            if norm > 0:
                phi_cpu = phi_cpu / norm
            novelty_counter = 0
            last_novelty = episodic_novelty
        else:
            episodic_novelty = last_novelty

        with torch.no_grad():
            pred_next = dynamics_model(phi_t, act_onehot)
            expected_error_log = error_model(phi_t, act_onehot).item()

        action_buttons = MACRO_ACTIONS[act_idx_item]
        action_vector = [btn in action_buttons for btn in BUTTONS_ORDER]
        rwd_e = game.make_action(action_vector, FRAME_SKIP)

        state_next = game.get_state()
        done_flag = game.is_episode_finished()

        if state_next is not None:
            raw_frame_next = state_next.screen_buffer
            raw_frame_next_hwc = raw_frame_next.transpose(1, 2, 0)  # (H, W, 3)
            frame_buffer_next = stack_frames(frame_buffer, raw_frame_next)
            obs_next_stacked_np = np.stack(frame_buffer_next, axis=0)  # (4, H, W) grayscale

            with torch.no_grad():
                obs_next_t = torch.tensor(obs_next_stacked_np, dtype=torch.float32, device=device).unsqueeze(0)
                phi_next_t = encoder_model(obs_next_t)

                mse = F.mse_loss(pred_next, phi_next_t, reduction='none')
            mse_flat = mse.view(mse.shape[0], -1).mean(dim=1)
            epsilon = math.log(max(mse_flat.item(), EPSILON_CLAMP))
        else:
            continue

        learning_progress = max(expected_error_log - epsilon, 0.0) if error_ready else 0.0
        rwd_i = (
            LP_WEIGHT * learning_progress +
            EPISODIC_WEIGHT * episodic_novelty
        )
        obss.append(obs_stacked_np.astype(np.uint8))
        obss_next.append(obs_next_stacked_np.astype(np.uint8))

        if save_rgb_debug:
            obss_rgb.append(raw_frame_hwc)
            obss_next_rgb.append(raw_frame_next_hwc)

        acts.append(act_idx_item)
        logps.append(logp.item())
        vals.append(val.item())
        rwds_e.append(rwd_e)
        rwds_i.append(rwd_i)
        novelties.append(episodic_novelty)
        learning_progresses.append(learning_progress)
        dones.append(done_flag)
        epsilons.append(epsilon)

        frames_collected += FRAME_SKIP

        # Update running rewards for stats
        run_rwd_i = run_rwd_i * GAMMA_INT + rwd_i
        run_rwd_e = run_rwd_e * GAMMA_EXT + rwd_e

    intrinsic_stats, extrinsic_stats = stats
    intrinsic_stats.update(run_rwd_i)
    extrinsic_stats.update(run_rwd_e)

    with frame_counter:
        frame_counter.count += len(obss) * FRAME_SKIP

    game.close()

    buffer.put((thread_id, obss, obss_next, acts, logps, vals, rwds_e, rwds_i, epsilons, novelties, learning_progresses, dones, obss_rgb, obss_next_rgb))

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

def run_parallel_episodes(n_threads, agent, dynamics_model, error_model, encoder_model, buffer, stats, frame_counter,
                         error_ready, debug_thread_id, device='cpu'):
    threads = []
    for tid in range(n_threads):
        save_rgb = (tid == debug_thread_id)  # Only one thread saves RGB
        t = threading.Thread(
            target=run_episode,
            args=(tid, agent, dynamics_model, error_model, encoder_model, buffer, stats, frame_counter,
                  error_ready, save_rgb, device)
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def ppo_update(agent, optimizer, obs_batch, act_batch, old_logp_batch,
               adv_batch, ret_batch, clip_coef=CLIP_COEF, ent_coef=0.01,
               vf_coef=VF_COEF):
    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
    new_logp, entropy, values = agent.evaluate_actions(obs_batch, act_batch)
    ratio = (new_logp - old_logp_batch).exp()
    unclipped = ratio * adv_batch
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_batch
    pi_loss = -torch.min(unclipped, clipped).mean()
    v_loss = 0.5 * F.mse_loss(values, ret_batch)
    ent_loss = -entropy.mean()
    loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    return pi_loss.item(), v_loss.item(), ent_loss.item()

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
    from datetime import datetime
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

    cpus = multiprocessing.cpu_count()-4
    n_threads = max(1, cpus - 2)
    print(f"Using {n_threads} worker threads")

    n_actions = len(MACRO_ACTIONS)
    agent = Agent(n_actions).to(device)
    dynamics_model = LPMDynamicsModel(n_actions, feat_dim=ENCODER_FEATS, hidden=LPM_HIDDEN).to(device)
    error_model = LPMErrorModel(n_actions, feat_dim=ENCODER_FEATS, hidden=LPM_HIDDEN).to(device)
    encoder_model = RandomEncoder(out_dim=ENCODER_FEATS, in_ch=FRAME_STACK).to(device)

    agent_optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=DYNAMICS_LEARNING_RATE)
    error_optimizer = torch.optim.Adam(error_model.parameters(), lr=ERROR_LEARNING_RATE)

    dynamics_buffer = DynamicsReplayBuffer(DYNAMICS_REPLAY_SIZE)
    error_buffer = ErrorReplayBuffer(ERROR_BUFFER_SIZE)
    # Running RMS normalizers for rewards (as in Burda et al.)
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

        all_obs, all_obs_next, all_acts, all_logps, all_advantages, all_returns, all_dones, all_rwds_i_raw, all_rwds_e_raw, all_epsilons, all_novelties, all_learning_progress = [], [], [], [], [], [], [], [], [], [], [], []
        debug_episode_saved = False
        total_episodes = 0
        episodes_data = []

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
                 rwds_e, rwds_i, epsilons_ep, novelties_ep, learning_progresses_ep, dones,
                 obss_rgb, obss_next_rgb) = ep
                total_episodes += 1

                if thread_id == debug_thread_id and not debug_episode_saved and obss_rgb is not None:
                    save_frame_dir = f'{debug_dir}/iter_{iter_idx+1:04d}'
                    os.makedirs(save_frame_dir)
                    action_names = ['NOOP', 'FWD', 'BACK', 'TURN_L', 'TURN_R', 'FWD_L', 'FWD_R',
                                    'ATTACK', 'USE', 'FWD_ATK', 'STRAFE_L', 'STRAFE_R']
                    for i, (rgb_curr, rgb_next, act_idx, r_i, r_e, nov, lp) in enumerate(
                        zip(obss_rgb, obss_next_rgb, acts, rwds_i, rwds_e, novelties_ep, learning_progresses_ep)
                    ):
                        Image.fromarray(rgb_curr, mode='RGB').save(f'{save_frame_dir}/frame_{i:04d}_current.png')
                        Image.fromarray(rgb_next, mode='RGB').save(f'{save_frame_dir}/frame_{i:04d}_next.png')
                        action_name = action_names[act_idx] if act_idx < len(action_names) else f'ACT_{act_idx}'
                        with open(f'{save_frame_dir}/frame_{i:04d}_{action_name}.txt', 'w') as f:
                            f.write(f'Action: {action_name}\\n')
                            f.write(f'Intrinsic Reward: {r_i:.6f}\\n')
                            f.write(f'Extrinsic Reward: {r_e:.3f}\n')
                            f.write(f'Episodic Novelty: {nov:.6f}\n')
                            f.write(f'Learning Progress: {lp:.6f}\n')

                    debug_episode_saved = True

                obs_ep = torch.tensor(np.array(obss), dtype=torch.uint8)
                obs_next_ep = torch.tensor(np.array(obss_next), dtype=torch.uint8)
                acts_ep = torch.tensor(acts, dtype=torch.long)
                logps_ep = torch.tensor(logps, dtype=torch.float32)
                vals_ep = torch.tensor(vals, dtype=torch.float32)
                rwds_e_ep = torch.tensor(rwds_e, dtype=torch.float32)
                rwds_i_ep = torch.tensor(rwds_i, dtype=torch.float32)
                eps_ep = torch.tensor(epsilons_ep, dtype=torch.float32)
                novelty_ep = torch.tensor(novelties_ep, dtype=torch.float32)
                learning_progress_ep = torch.tensor(learning_progresses_ep, dtype=torch.float32)
                dones_ep = torch.tensor(dones, dtype=torch.bool)

                with torch.no_grad():
                    if IGNORE_DONES or not dones_ep[-1]:
                        obs_next_last = obs_next_ep[-1].float().to(device).unsqueeze(0)
                        _, next_val = agent.forward(obs_next_last)
                        next_val = next_val[0].cpu()
                    else:
                        next_val = torch.tensor(0.0)

                episodes_data.append({
                    "obs": obs_ep,
                    "obs_next": obs_next_ep,
                    "acts": acts_ep,
                    "logps": logps_ep,
                    "vals": vals_ep,
                    "rwds_e": rwds_e_ep,
                    "rwds_i": rwds_i_ep,
                    "eps": eps_ep,
                    "novelty": novelty_ep,
                    "learning_progress": learning_progress_ep,
                    "dones": dones_ep,
                    "next_val": next_val,
                })
                all_rwds_i_raw.extend(rwds_i)
                all_rwds_e_raw.extend(rwds_e)

                del ep, obss, obss_next, obss_rgb, obss_next_rgb

        dynamics_model.train()
        error_model.train()
        print(f"Collected {total_episodes} episodes, {frame_counter.get()} frames")

        _, _, rwd_i_std = rwd_i_rms.get()
        _, _, rwd_e_std = rwd_e_rms.get()

        intrinsic_scale = max(float(rwd_i_std), 1e-3) if all_rwds_i_raw else 1.0
        extrinsic_scale = max(float(rwd_e_std), 1e-3) if all_rwds_e_raw else 1.0

        rwds_e_norm_scaled = []
        rwds_i_norm_scaled = []

        for episode in episodes_data:
            rwds_i_norm = episode["rwds_i"] / intrinsic_scale
            rwds_e_norm = episode["rwds_e"] / extrinsic_scale
            total_rwds_ep = EXTRINSIC_COEF * rwds_e_norm + INTRINSIC_COEF * rwds_i_norm

            advs_ep, rets_ep = compute_gae(
                total_rwds_ep.unsqueeze(1),
                episode["vals"].unsqueeze(1),
                episode["dones"].unsqueeze(1),
                episode["next_val"].unsqueeze(0),
                GAMMA_EXT,  
                GAE_LAMBDA
            )

            rwds_e_norm_scaled.append(rwds_e_norm)
            rwds_i_norm_scaled.append(rwds_i_norm)

            all_obs.append(episode["obs"])
            all_obs_next.append(episode["obs_next"])
            all_acts.append(episode["acts"])
            all_logps.append(episode["logps"])
            all_advantages.append(advs_ep.squeeze(1))
            all_returns.append(rets_ep.squeeze(1))
            all_dones.append(episode["dones"])
            all_epsilons.append(episode["eps"])
            all_novelties.append(episode["novelty"])
            all_learning_progress.append(episode["learning_progress"])

        obs_t = torch.cat(all_obs, dim=0)
        obs_next_t = torch.cat(all_obs_next, dim=0)
        acts_t = torch.cat(all_acts, dim=0)
        logps_t = torch.cat(all_logps, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        returns = torch.cat(all_returns, dim=0)
        dones_t = torch.cat(all_dones, dim=0)
        eps_t = torch.cat(all_epsilons, dim=0)
        novelty_t = torch.cat(all_novelties, dim=0) if all_novelties else torch.empty(0, dtype=torch.float32)
        learning_progress_t = torch.cat(all_learning_progress, dim=0) if all_learning_progress else torch.empty(0, dtype=torch.float32)

        dynamics_buffer.extend(obs_t.cpu().numpy(),acts_t.cpu().numpy(),obs_next_t.cpu().numpy())
        error_buffer.extend(obs_t.cpu().numpy(), acts_t.cpu().numpy(), eps_t.cpu().numpy())

        n_samples = len(obs_t)
        indices = np.arange(n_samples)

        total_pi_loss, total_v_loss, total_ent_loss, total_dynamics_loss, error_loss_total, n_updates, dynamics_updates, error_updates = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        eps_pred_total, eps_actual_total = 0.0, 0.0

        for epoch in range(EPOCHS):
            np.random.shuffle(indices)
            epoch_start = time.time()
            ENT_COEF = (1/math.e**(iter_idx/ITERS * math.log(ENT_COEF_START/ENT_COEF_END))) * ENT_COEF_START

            for start in range(0, n_samples, MINIBATCH_SIZE):
                end = min(start + MINIBATCH_SIZE, n_samples)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx].float().to(device)
                pi_loss, v_loss, ent_loss = ppo_update(
                    agent, agent_optimizer,
                    mb_obs, acts_t[mb_idx].to(device), logps_t[mb_idx].to(device),
                    advantages[mb_idx].to(device), returns[mb_idx].to(device), ent_coef=ENT_COEF
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
                obs_b, act_b, eps_b_log = error_buffer.sample(ERROR_BATCH_SIZE)
                obs_b = obs_b.to(device)
                act_onehot = F.one_hot(act_b.to(device), n_actions).float()
                # eps_b is already logged in rollout
                eps_b_log = eps_b_log.to(device)

                with torch.no_grad():
                    phi_obs = encoder_model(obs_b)

                eps_pred = error_model(phi_obs, act_onehot)
                err_loss = F.mse_loss(eps_pred, eps_b_log)

                error_optimizer.zero_grad(set_to_none=True)
                err_loss.backward()
                nn.utils.clip_grad_norm_(error_model.parameters(), MAX_GRAD_NORM)
                error_optimizer.step()

                error_loss_total += err_loss.item()
                error_updates += 1

                eps_pred_total += eps_pred.mean().item()
                eps_actual_total += eps_b_log.mean().item()

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
        novelty_mean = novelty_t.mean().item() if novelty_t.numel() > 0 else 0.0
        novelty_std = novelty_t.std().item() if novelty_t.numel() > 0 else 0.0
        novelty_max = novelty_t.max().item() if novelty_t.numel() > 0 else 0.0
        learning_progress_mean = learning_progress_t.mean().item() if learning_progress_t.numel() > 0 else 0.0
        learning_progress_std = learning_progress_t.std().item() if learning_progress_t.numel() > 0 else 0.0
        learning_progress_max = learning_progress_t.max().item() if learning_progress_t.numel() > 0 else 0.0

        error_buffer.commit_pending()

        log_dict = {
            "update_step": iter_idx + 1,
            "lpm/error_pred_mean": error_pred_mean,
            "lpm/error_actual": error_actual_mean,
            "reward/learning_progress_mean": learning_progress_mean,
            "reward/intrinsic_batch_mean": batch_mean_i,
            "reward/extrinsic_batch_mean": batch_mean_e,
            "reward/extrinsic_batch_norm_mean": float(np.mean(rwds_e_norm_scaled)),
            "reward/intrinsic_batch_norm_mean": float(np.mean(rwds_i_norm_scaled)),
            "loss/policy": pi_loss,
            "loss/value": v_loss,
            "loss/entropy": -ent_loss,
            "loss/dynamics": dynamics_loss,
            "loss/error_model": error_loss_mean,
            "reward/intrinsic_running": mean_i_running,
            "reward/extrinsic_running": mean_e_running,
            "reward/learning_progress_std": learning_progress_std,
            "reward/learning_progress_max": learning_progress_max,
            "reward/intrinsic_std_running": std_i_running if not math.isnan(std_i_running) else 0,
            "reward/extrinsic_std_running": std_e_running if not math.isnan(std_e_running) else 0,
            "reward/intrinsic_std_rms": rwd_i_std,
            "reward/extrinsic_std_rms": rwd_e_std,
            "reward/intrinsic_batch_std": batch_std_i,
            "reward/intrinsic_batch_max": batch_max_i,
            "reward/intrinsic_batch_min": batch_min_i,
            "reward/extrinsic_batch_std": batch_std_e,
            "reward/extrinsic_batch_max": batch_max_e,
            "reward/extrinsic_batch_sum": batch_sum_e,
            "reward/total_batch": batch_total,
            "reward/novelty_mean": novelty_mean,
            "reward/novelty_std": novelty_std,
            "reward/novelty_max": novelty_max,
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

        with open(f'{debug_dir}/log_{iter_idx:04d}.json', 'w') as f:
            json.dump(log_dict, f)

        for k,v in log_dict.items():
            print(k, ": ", v)
        
        wandb.log(log_dict)
        print(f"Timer {iter_time:.1f}s | FPS: {frame_counter.get()/iter_time:.0f}")
        print(f"Policy Loss: {pi_loss:.4f}, Value Loss: {v_loss:.4f}, Entropy: {-ent_loss:.4f}")
        print(f"Dynamics Loss: {dynamics_loss:.6f}, Error Model Loss: {error_loss_mean:.6f}")
        print("--------------------------------")
        print("Iteration Rollout Rewards:")
        print(f"Intrinsic raw: μ={batch_mean_i:.6f}, σ={batch_std_i:.6f}, max={batch_max_i:.6f}")
        print(f"Extrinsic raw: μ={batch_mean_e:.3f}, max={batch_max_e:.3f}, sum={batch_sum_e:.1f}")
        print(f"Extrinsic scaled: μ={float(np.mean(rwds_e_norm_scaled)):.3f}, max={float(np.max(rwds_e_norm_scaled)):.3f}, sum={float(np.sum(rwds_e_norm_scaled)):.1f}")
        print(f"Intrinsic scaled: μ={float(np.mean(rwds_i_norm_scaled)):.3f}, max={float(np.max(rwds_i_norm_scaled)):.3f}, sum={float(np.sum(rwds_i_norm_scaled)):.1f}")
        if INTRINSIC_ONLY: print("Note: Training with INTRINSIC_ONLY (extrinsic not used for learning)")
        os.makedirs(f'{debug_dir}/checkpoints', exist_ok=True)
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
            torch.save(checkpoint, f'{debug_dir}/checkpoints/idm_lpm_checkpoint_{iter_idx+1}.pth')
    wandb.finish()
