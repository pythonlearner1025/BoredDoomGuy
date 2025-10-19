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
    Agent, 
    RunningStats,
)

ITERS = 1000
EPOCHS = 3  
FRAME_SKIP = 4
FRAME_STACK = 4  # Stack last 4 frames as in the paper
MAX_ROLLOUT_FRAMES = FRAME_SKIP * int(1e4)  # Reduced to avoid OOM
MAX_FRAMES_PER_EPISODE = int(37.5*60*1)  # Per-thread 1 minute episode limit

CLIP_COEF = 0.1
ENT_COEF_START = 0.03
ENT_COEF_END = 0.001
ENT_COEF = ENT_COEF_START
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
MINIBATCH_SIZE = 2048
LEARNING_RATE = 1e-4

GAMMA_INT = 0.99
GAMMA_EXT = 0.999
GAE_LAMBDA = 0.95
INTRINSIC_COEF = 0.5
EXTRINSIC_COEF = 0.5

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

cpus = multiprocessing.cpu_count()-4
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
        "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "minibatch_size": MINIBATCH_SIZE,
        "policy_lr": LEARNING_RATE,
        "intrinsic_coef": INTRINSIC_COEF,
        "extrinsic_coef": EXTRINSIC_COEF,
        "frame_skip": FRAME_SKIP,
        "frame_stack": FRAME_STACK,
        "n_actions": len(MACRO_ACTIONS),
}


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
       
def run_episode(thread_id, agent, buffer, rwd_i_rms, frame_counter,
                save_rgb_debug, device='cpu'):
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

    obss, obss_next = [], []  # Stacked frames (4, H, W) uint8 grayscale
    obss_rgb = [] if save_rgb_debug else None
    obss_next_rgb = [] if save_rgb_debug else None
    acts, logps, vals_int, vals_ext, rwds_e, rwds_i, dones = [], [], [], [], [], [], []

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
        obs_t = torch.tensor(obs_stacked_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            acts, logp, v_int, v_ext, entropy = agent.step(obs_t)
            dist = torch.distributions.Categorical(logits=logp)
            act_idx = dist.sample()
            act_idx_item = act_idx.item()

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
        else:
            continue

        obss.append(obs_stacked_np.astype(np.uint8))
        obss_next.append(obs_next_stacked_np.astype(np.uint8))

        if save_rgb_debug:
            obss_rgb.append(raw_frame_hwc)
            obss_next_rgb.append(raw_frame_next_hwc)

        acts.append(act_idx_item)
        logps.append(logp.item())
        vals_int.append(v_int.item())
        vals_ext.append(v_ext.item())
        rwds_e.append(rwd_e)
        dones.append(done_flag)

        frames_collected += FRAME_SKIP

    with frame_counter:
        frame_counter.count += len(obss) * FRAME_SKIP

    game.close()

    buffer.put((thread_id, obss, obss_next, acts, logps, vals_int, vals_ext, rwds_e, dones, obss_rgb, obss_next_rgb))

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

def run_parallel_episodes(n_threads, agent, buffer, rwd_i_rms, frame_counter,
                         debug_thread_id, device='cpu'):
    threads = []
    for tid in range(n_threads):
        save_rgb = (tid == debug_thread_id)  # Only one thread saves RGB
        t = threading.Thread(
            target=run_episode,
            args=(tid, agent, buffer, rwd_i_rms, frame_counter,
                  save_rgb, device)
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
    init_wandb(wandb_config)
    agent = Agent(n_actions).to(device)
    agent_optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    rwd_i_rms = RunningStats()

    for iter_idx in range(ITERS):
        print(f"\n=== Iteration {iter_idx+1}/{ITERS} ===")
        iter_start = time.time()

        buffer = Queue()
        frame_counter = FrameCounter()
        debug_thread_id = random.randint(0, n_threads - 1)

        all_obs, all_obs_next, all_acts, all_logps, all_advantages, all_returns, all_dones, all_rwds_i_raw, all_rwds_e_raw = [], [], [], [], [], [], [], [], [], []
        debug_episode_saved = False
        total_episodes = 0
        episodes_data = []

        while frame_counter.get() < MAX_ROLLOUT_FRAMES:
            run_parallel_episodes(
                n_threads, agent, buffer, rwd_i_rms, frame_counter, debug_thread_id, device
            )

            while not buffer.empty():
                ep = buffer.get()
                (thread_id, obss, obss_next, acts, logps, vals_int, vals_ext,
                 rwds_e, rwds_i, dones,
                 obss_rgb, obss_next_rgb) = ep
                total_episodes += 1

                if thread_id == debug_thread_id and not debug_episode_saved and obss_rgb is not None:
                    save_frame_dir = f'{debug_dir}/iter_{iter_idx+1:04d}'
                    os.makedirs(save_frame_dir)
                    action_names = ['NOOP', 'FWD', 'BACK', 'TURN_L', 'TURN_R', 'FWD_L', 'FWD_R',
                                    'ATTACK', 'USE', 'FWD_ATK', 'STRAFE_L', 'STRAFE_R']
                    for i, (rgb_curr, rgb_next, act_idx, r_i, r_e) in enumerate(
                        zip(obss_rgb, obss_next_rgb, acts, rwds_i, rwds_e)
                    ):
                        Image.fromarray(rgb_curr, mode='RGB').save(f'{save_frame_dir}/frame_{i:04d}_current.png')
                        Image.fromarray(rgb_next, mode='RGB').save(f'{save_frame_dir}/frame_{i:04d}_next.png')
                        action_name = action_names[act_idx] if act_idx < len(action_names) else f'ACT_{act_idx}'
                        with open(f'{save_frame_dir}/frame_{i:04d}_{action_name}.txt', 'w') as f:
                            f.write(f'Action: {action_name}\\n')
                            f.write(f'Intrinsic Reward: {r_i:.6f}\\n')
                            f.write(f'Extrinsic Reward: {r_e:.3f}\n')

                    debug_episode_saved = True

                obs_ep = torch.tensor(np.array(obss), dtype=torch.uint8)
                obs_next_ep = torch.tensor(np.array(obss_next), dtype=torch.uint8)
                acts_ep = torch.tensor(acts, dtype=torch.long)
                logps_ep = torch.tensor(logps, dtype=torch.float32)
                vals_int_ep = torch.tensor(vals_int, dtype=torch.float32)
                vals_ext_ep = torch.tensor(vals_ext, dtype=torch.float32)
                rwds_e_ep = torch.tensor(rwds_e, dtype=torch.float32)
                rwds_i_ep = torch.tensor(rwds_i, dtype=torch.float32)
                dones_ep = torch.tensor(dones, dtype=torch.bool)

                # TODO don't understand bootstrapping, check back later
                with torch.no_grad():
                    if not dones_ep[-1]:
                        obs_next_last = obs_next_ep[-1].float().to(device).unsqueeze(0)
                        acts, logp, v_int, v_ext, entropy = agent.step(obs_next_last)
                        next_val_int = v_int[0].cpu()
                        next_val_ext = v_ext[0].cpu()
                    else:
                        next_val_int = torch.tensor(0.0)
                        next_val_ext = torch.tensor(0.0)

                episodes_data.append({
                    "obs": obs_ep,
                    "obs_next": obs_next_ep,
                    "acts": acts_ep,
                    "logps": logps_ep,
                    "vals_int": vals_int_ep,
                    "vals_ext": vals_ext_ep,
                    "rwds_e": rwds_e_ep,
                    "rwds_i": rwds_i_ep,
                    "dones": dones_ep,
                    "next_val_int": next_val_int,
                    "next_val_ext": next_val_ext,
                })

                rwd_i_rms.update(sum(rwds_i)/len(rwds_i))
                rwd_e_rms.update(sum(rwds_e)/len(rwds_e))

                all_rwds_i_raw.extend(rwds_i)
                all_rwds_e_raw.extend(rwds_e)

                del ep, obss, obss_next, obss_rgb, obss_next_rgb

        print(f"Collected {total_episodes} episodes, {frame_counter.get()} frames")

        _, _, rwd_i_std = rwd_i_rms.get()
        _, _, rwd_e_std = rwd_e_rms.get()

        intrinsic_scale = max(float(rwd_i_std), 1e-3) if all_rwds_i_raw else 1.0
        extrinsic_scale = max(float(rwd_e_std), 1e-3) if all_rwds_e_raw else 1.0

        rwds_e_norm_scaled = []
        rwds_i_norm_scaled = []

        for episode in episodes_data:
            rwds_e_norm = episode["rwds_e"] / extrinsic_scale
            rwds_i_norm = torch.clamp(episode["rwds_i"] / intrinsic_scale, max=1.0)
            total_rwds_ep = EXTRINSIC_COEF * rwds_e_norm + INTRINSIC_COEF * rwds_i_norm

            advs_ep, rets_ep = compute_gae(
                total_rwds_ep.unsqueeze(1),
                episode["vals"].unsqueeze(1),
                episode["dones"].unsqueeze(1),
                episode["next_val"].unsqueeze(0),
                GAMMA_EXT,  
                GAE_LAMBDA
            )

            rwds_e_norm_scaled.append(rwds_e_norm * EXTRINSIC_COEF)
            rwds_i_norm_scaled.append(rwds_i_norm * INTRINSIC_COEF)

            all_obs.append(episode["obs"])
            all_obs_next.append(episode["obs_next"])
            all_acts.append(episode["acts"])
            all_logps.append(episode["logps"])
            all_advantages.append(advs_ep.squeeze(1))
            all_returns.append(rets_ep.squeeze(1))
            all_dones.append(episode["dones"])

        obs_t = torch.cat(all_obs, dim=0)
        obs_t_next = torch.cat(all_obs_next, dim=0)
        acts_t = torch.cat(all_acts, dim=0)
        logps_t = torch.cat(all_logps, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        returns = torch.cat(all_returns, dim=0)
        dones_t = torch.cat(all_dones, dim=0)

        n_samples = len(obs_t)
        indices = np.arange(n_samples)

        total_pi_loss, total_v_loss, total_ent_loss, total_dynamics_loss, error_loss_total, n_updates, dynamics_updates, error_updates = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        eps_pred_total, eps_actual_total = 0.0, 0.0
        idm_loss_total, joint_loss_total = 0.0, 0.0

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

            epoch_end = time.time()
            print(f"EPOCH {epoch} took {(epoch_end - epoch_start):0.2f}s")

        pi_loss = total_pi_loss / max(n_updates, 1)
        v_loss = total_v_loss / max(n_updates, 1)
        ent_loss = total_ent_loss / max(n_updates, 1)

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

        log_dict = {
            "update_step": iter_idx + 1,
            "reward/intrinsic_batch_mean": batch_mean_i,
            "reward/extrinsic_batch_mean": batch_mean_e,
            "reward/extrinsic_batch_norm_mean": float(np.mean(rwds_e_norm_scaled)),
            "reward/intrinsic_batch_norm_mean": float(np.mean(rwds_i_norm_scaled)),
            "loss/policy": pi_loss,
            "loss/value": v_loss,
            "loss/entropy": -ent_loss,
            "reward/intrinsic_running": mean_i_running,
            "reward/extrinsic_running": mean_e_running,
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
        print("--------------------------------")
        print("Iteration Rollout Rewards:")
        print(f"Intrinsic raw: μ={batch_mean_i:.6f}, σ={batch_std_i:.6f}, max={batch_max_i:.6f}")
        print(f"Extrinsic raw: μ={batch_mean_e:.3f}, max={batch_max_e:.3f}, sum={batch_sum_e:.1f}")
        print(f"Intrinsic scaled: μ={float(np.mean(rwds_i_norm_scaled)):.3f}, max={float(np.max(rwds_i_norm_scaled)):.3f}, sum={float(np.sum(rwds_i_norm_scaled)):.1f}")
        print(f"Extrinsic scaled: μ={float(np.mean(rwds_e_norm_scaled)):.3f}, max={float(np.max(rwds_e_norm_scaled)):.3f}, sum={float(np.sum(rwds_e_norm_scaled)):.1f}")
        os.makedirs(f'{debug_dir}/checkpoints', exist_ok=True)
        if (iter_idx + 1) % 25 == 0:
            checkpoint = {
                'iteration': iter_idx + 1,
                'agent_state_dict': agent.state_dict(),
                'agent_optimizer': agent_optimizer.state_dict(),
            }
            torch.save(checkpoint, f'{debug_dir}/checkpoints/icm_rnd_checkpoint_{iter_idx+1}.pth')
    wandb.finish()
