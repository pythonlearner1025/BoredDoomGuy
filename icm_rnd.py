import warnings
warnings.filterwarnings('ignore')

import os, math, time, torch, json, random, multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import vizdoom as vzd
import numpy as np
import wandb
from datetime import datetime
from PIL import Image
from multiprocessing import Process
from model import RNDAgent, RunningStats

ITERS, EPOCHS = 1000, 2
FRAME_SKIP, FRAME_STACK = 4, 4
MAX_ROLLOUT_FRAMES = FRAME_SKIP * int(1e4)
MAX_FRAMES_PER_EPISODE = int(37.5*60*1)

CLIP_COEF, VF_COEF, MAX_GRAD_NORM = 0.1, 0.5, 1.0
ENT_COEF_START, ENT_COEF_END = 0.10, 0.001
MINIBATCH_SIZE, LEARNING_RATE = 2048, 1e-4
GAMMA_INT, GAMMA_EXT, GAE_LAMBDA = 0.99, 0.999, 0.95
INTRINSIC_COEF, EXTRINSIC_COEF = 0.8, 0.2

OBS_HEIGHT, OBS_WIDTH = 60, 80
OBS_CHANNELS = FRAME_STACK
OBS_SIZE = OBS_CHANNELS * OBS_HEIGHT * OBS_WIDTH
OBS_RANDOM_WARMUP_STEPS = 10000
OBS_CLIP_MIN = -5.0
OBS_CLIP_MAX = 5.0

MACRO_ACTIONS = [
    [], [vzd.Button.MOVE_FORWARD], [vzd.Button.MOVE_BACKWARD],
    [vzd.Button.TURN_LEFT], [vzd.Button.TURN_RIGHT],
    [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT],
    [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_RIGHT],
    [vzd.Button.ATTACK], [vzd.Button.USE],
    [vzd.Button.MOVE_FORWARD, vzd.Button.ATTACK],
    [vzd.Button.MOVE_LEFT], [vzd.Button.MOVE_RIGHT],
]

BUTTONS_ORDER = sorted({btn for action in MACRO_ACTIONS for btn in action}, key=lambda b: b.value)
n_actions = len(MACRO_ACTIONS)

wandb_config = {
    k: v for k, v in locals().items() 
    if k.isupper() and not k.startswith('_') and isinstance(v, (int, float, str, bool))
}

def init_wandb(config):
    mode = "disabled" if os.environ.get("WANDB_MODE", "").lower() in {"disabled", "offline"} else "online"
    return wandb.init(mode=mode, config=config, project="doom-rnd-curiosity")

def rgb_to_grayscale(rgb_frame):
    weights = [0.299, 0.587, 0.114]
    if rgb_frame.shape[0] == 3:
        gray = sum(w * rgb_frame[i] for i, w in enumerate(weights))
    else:
        gray = sum(w * rgb_frame[:, :, i] for i, w in enumerate(weights))
    return gray.astype(np.uint8)

def resize_gray_frame(gray_frame):
    if gray_frame.shape[:2] == (OBS_HEIGHT, OBS_WIDTH):
        return gray_frame
    return np.array(Image.fromarray(gray_frame).resize((OBS_WIDTH, OBS_HEIGHT), Image.BILINEAR), dtype=np.uint8)

def stack_frames(frame_buffer, new_frame):
    gray_frame = resize_gray_frame(rgb_to_grayscale(new_frame))
    return [gray_frame] * FRAME_STACK if not frame_buffer else frame_buffer[1:] + [gray_frame]
       
def setup_game():
    game = vzd.DoomGame()
    wad_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Doom1.WAD")
    game.set_doom_game_path(wad_path)
    game.set_doom_map("E1M1")
    game.set_render_hud(False)
    for reward, value in [("death", -0.5), ("kill", 10), ("armor", 0), ("health", 0), ("map_exit", 100), ("secret", 50)]:
        getattr(game, f"set_{reward}_reward")(value)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args("+freelook 1")
    game.set_available_buttons(BUTTONS_ORDER)
    game.init()
    return game

def normalize_obs(obs, ob_rms, device):
    obs_t = torch.tensor(obs.astype(np.float32) / 255.0, dtype=torch.float32, device=device).unsqueeze(0)
    if ob_rms is None:
        return obs_t
    _, mean_hw1, std_hw1 = ob_rms.get()
    mean = (mean_hw1 if isinstance(mean_hw1, torch.Tensor) else torch.tensor(mean_hw1, device=device)).to(dtype=torch.float32, device=device)
    std = torch.clamp(torch.as_tensor(std_hw1, dtype=torch.float32, device=device), min=1e-6)
    mean_chw, std_chw = mean[..., 0].unsqueeze(0).repeat(OBS_CHANNELS, 1, 1), std[..., 0].unsqueeze(0).repeat(OBS_CHANNELS, 1, 1)
    return ((obs_t.squeeze(0) - mean_chw) / std_chw).unsqueeze(0)

def run_episode(process_id, agent_state_dict, buffer, frame_counter, save_rgb_debug, device_str='cpu', ob_rms_data=None):
    device = torch.device(device_str)
    agent = RNDAgent(OBS_SIZE, n_actions, hidden=(256,256), rnd_feat_dim=128, conv_out_dim=128).to(device)
    agent.load_state_dict(agent_state_dict)
    agent.eval()
    
    ob_rms = None
    if ob_rms_data:
        ob_rms = RunningStats(shape=(OBS_HEIGHT, OBS_WIDTH, 1), dtype=torch.float32, device=device)
        ob_rms.count, ob_rms.mean, ob_rms.var = ob_rms_data
    
    game = setup_game()
    game.new_episode()

    obss, acts, logps, vals_int, vals_ext, rwds_e, dones = [], [], [], [], [], [], []
    obss_rgb = [] if save_rgb_debug else None
    frame_buffer, frames_collected = [], 0
    timings = {k: [] for k in ['state_get', 'frame_stack', 'obs_norm', 'agent_fwd', 'action_prep', 'env_step', 'data_collect']}

    while frames_collected < MAX_FRAMES_PER_EPISODE:
        if game.is_episode_finished():
            game.new_episode()
            frame_buffer = []
            continue

        t0, state = time.time(), game.get_state()
        if state is None:
            continue
        timings['state_get'].append(time.time() - t0)

        t0 = time.time()
        raw_frame = state.screen_buffer
        frame_buffer = stack_frames(frame_buffer, raw_frame)
        obs_stacked_np = np.stack(frame_buffer, axis=0)
        timings['frame_stack'].append(time.time() - t0)
        
        t0, obs_t = time.time(), normalize_obs(obs_stacked_np, ob_rms, device)
        timings['obs_norm'].append(time.time() - t0)

        with torch.no_grad():
            t0 = time.time()
            action, _, logp, v_int, v_ext = agent.step(obs_t)
            timings['agent_fwd'].append(time.time() - t0)

        t0, act_idx = time.time(), int(action)
        action_vector = [btn in MACRO_ACTIONS[act_idx] for btn in BUTTONS_ORDER]
        timings['action_prep'].append(time.time() - t0)
        
        t0, rwd_e = time.time(), game.make_action(action_vector, FRAME_SKIP)
        timings['env_step'].append(time.time() - t0)

        t0 = time.time()
        obss.append(obs_stacked_np.astype(np.uint8))
        if save_rgb_debug:
            obss_rgb.append(raw_frame.transpose(1, 2, 0))
        acts.extend([act_idx])
        logps.extend([float(logp)])
        vals_int.extend([float(v_int)])
        vals_ext.extend([float(v_ext)])
        rwds_e.extend([rwd_e])
        dones.extend([game.is_episode_finished()])
        timings['data_collect'].append(time.time() - t0)
        frames_collected += FRAME_SKIP

    frame_counter.add(len(obss) * FRAME_SKIP)
    game.close()

    t0, obss_batch = time.time(), torch.as_tensor(np.array(obss, dtype=np.uint8).astype(np.float32) / 255.0, dtype=torch.float32, device=device)
    if ob_rms:
        _, mean, std = ob_rms.get()
        mean = (mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, device=device)).to(dtype=torch.float32, device=device)
        std = torch.clamp(torch.as_tensor(std, dtype=torch.float32, device=device), min=1e-6)
        mean_chw, std_chw = mean[..., 0].unsqueeze(0).repeat(OBS_CHANNELS, 1, 1), std[..., 0].unsqueeze(0).repeat(OBS_CHANNELS, 1, 1)
        obss_batch = (obss_batch - mean_chw) / std_chw
    with torch.no_grad():
        rwds_i = agent.rnd.intrinsic_reward(obss_batch).detach().cpu().numpy().tolist()
    timings['intrinsic'], timings['n_steps'] = time.time() - t0, len(obss)

    buffer.put((process_id, obss, acts, logps, vals_int, vals_ext, rwds_e, rwds_i, dones, obss_rgb, timings))

class FrameCounter:
    def __init__(self, manager):
        self.count, self.lock = manager.Value('i', 0), manager.Lock()
    def get(self):
        with self.lock:
            return self.count.value
    def add(self, value):
        with self.lock:
            self.count.value += value

def run_parallel_episodes(n_procs, agent_state_dict, buffer, frame_counter, debug_proc_id, device='cpu', ob_rms=None):
    ob_rms_data = None
    if ob_rms:
        count, mean, var = ob_rms.get()
        ob_rms_data = (count, mean.cpu() if isinstance(mean, torch.Tensor) else mean, var.cpu() if isinstance(var, torch.Tensor) else var)
    
    processes = [Process(target=run_episode, args=(pid, agent_state_dict, buffer, frame_counter, pid == debug_proc_id, 'cpu', ob_rms_data)) for pid in range(n_procs)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

def ppo_update(agent, optimizer, obs_mb, act_mb, old_logp_mb, adv_mb, ret_int_mb, ret_ext_mb, clip_coef=CLIP_COEF, ent_coef=0.01, vf_coef=VF_COEF):
    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
    logits, v_int_new, v_ext_new = agent.forward(obs_mb)
    dist = torch.distributions.Categorical(logits=logits)
    logp_new_mb, entropy, ratio = dist.log_prob(act_mb), dist.entropy().mean(), (dist.log_prob(act_mb) - old_logp_mb).exp()

    pi_loss = -torch.min(ratio * adv_mb, torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_mb).mean()
    v_loss_int, v_loss_ext = 0.5 * F.mse_loss(v_int_new, ret_int_mb), 0.5 * F.mse_loss(v_ext_new, ret_ext_mb)
    rnd_loss = agent.rnd.predictor_loss(obs_mb).mean()

    total_loss = pi_loss + vf_coef * (v_loss_int + v_loss_ext) - ent_coef * entropy + rnd_loss
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    return pi_loss.item(), (v_loss_int + v_loss_ext).item(), v_loss_int.item(), v_loss_ext.item(), -entropy.item(), rnd_loss.item()

def compute_gae_1d(rewards, values, dones, gamma, gae_lambda):
    T, device = rewards.shape[0], rewards.device
    advantages, lastgaelam = torch.zeros(T, dtype=torch.float32, device=device), 0.0
    for t in reversed(range(T)):
        nextnonterminal = 0.0 if (dones[t if t == T - 1 else t+1].float() > 0.5) else 1.0
        nextvalues = torch.tensor(0.0, dtype=torch.float32, device=device) if t == T - 1 else values[t+1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + values

def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda'), "CUDA GPU"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            device = torch.device('mps')
            torch.zeros(1).to(device)
            return device, "MPS (Metal)"
        except:
            pass
    return torch.device('cpu'), "CPU"

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    import sys
    n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else multiprocessing.cpu_count() - 4
    device, device_name = setup_device()
    print(f"Device: {device_name} | Workers: {n_threads}")
    
    torch.autograd.set_detect_anomaly(True)
    debug_dir = f'debug/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(debug_dir, exist_ok=True)

    init_wandb(wandb_config)
    agent = RNDAgent(OBS_SIZE, n_actions, hidden=(256,256), rnd_feat_dim=128, conv_out_dim=128).to(device)
    
    if device.type != 'mps':
        try:
            agent = torch.compile(agent, mode='default')
            print("torch.compile: enabled")
        except:
            print("torch.compile: failed")
    
    agent_optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    rff_rms = RunningStats()
    ob_rms = RunningStats(shape=(OBS_HEIGHT, OBS_WIDTH, 1), dtype=torch.float32, device=device)

    # Initialize observation normalization parameters using a random agent
    print("Warming up observation normalization with random policy...")
    warmup_observation_stats(ob_rms)
    print("Warmup complete.")

    for iter_idx in range(ITERS):
        print(f"\n=== Iteration {iter_idx+1}/{ITERS} ===")
        iter_start = time.time()
        
        agent.eval()
        agent_state_dict = {k.replace('_orig_mod.', ''): v.cpu() for k, v in agent.state_dict().items()}
        manager = multiprocessing.Manager()
        buffer, frame_counter, debug_proc_id = manager.Queue(), FrameCounter(manager), random.randint(0, n_threads - 1)

        debug_episode_saved, total_episodes = False, 0
        rollout_data = {k: [] for k in ['obs', 'acts', 'logps', 'vals_int', 'vals_ext', 'rwds_e', 'rwds_i', 'dones', 'rollout_rwds_i_raw', 'rollout_rwds_e_raw']}
        timing_data = {k: [] for k in ['state_get', 'frame_stack', 'obs_norm', 'agent_fwd', 'action_prep', 'env_step', 'data_collect', 'intrinsic']}
        start_time = time.time()

        while frame_counter.get() < MAX_ROLLOUT_FRAMES:
            run_parallel_episodes(
                n_threads, agent_state_dict, buffer, frame_counter, debug_proc_id, device, ob_rms
            )

            while not buffer.empty():
                ep = buffer.get()
                (proc_id, obss, acts, logps, vals_int, vals_ext,
                 rwds_e, rwds_i, dones,
                 obss_rgb, timing_stats) = ep
                total_episodes += 1
                
                # Aggregate timing data
                for key in timing_data.keys():
                    if key == 'intrinsic':
                        timing_data[key].append(timing_stats[key])
                    else:
                        timing_data[key].extend(timing_stats[key])

                if proc_id == debug_proc_id and not debug_episode_saved and obss_rgb is not None:
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

                for k, v in zip(['obs', 'acts', 'logps', 'vals_int', 'vals_ext', 'rwds_e', 'rwds_i', 'dones'], [obss, acts, logps, vals_int, vals_ext, rwds_e, rwds_i, dones]):
                    rollout_data[k].append(v)
                rollout_data["rollout_rwds_i_raw"].append(rwds_i)
                rollout_data["rollout_rwds_e_raw"].append(rwds_e)
        
        # Read values from manager before shutdown
        total_frames = frame_counter.get()
        
        # Clean up multiprocessing manager
        manager.shutdown()

        rollout_time = time.time() - start_time
        print(f"Rollout: {total_episodes}ep {total_frames}f in {rollout_time:.2f}s")
        
        if timing_data['agent_fwd']:
            timings_ms = {k: np.mean(v)*1000 for k, v in timing_data.items() if isinstance(v, list) and v}
            total_ms = sum(timings_ms[k] for k in ['state_get', 'frame_stack', 'obs_norm', 'agent_fwd', 'action_prep', 'env_step', 'data_collect'])
            print(f"Timing: Agent={timings_ms['agent_fwd']:.2f}ms Env={timings_ms['env_step']:.2f}ms Total={total_ms:.2f}ms/step")
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
            mean_hw1_t = mean_hw1 if isinstance(mean_hw1, torch.Tensor) else torch.tensor(mean_hw1, device=device)
            std_hw1_t = std_hw1 if isinstance(std_hw1, torch.Tensor) else torch.tensor(std_hw1, device=device)
            mean_hw1_t = mean_hw1_t.to(dtype=torch.float32, device=device)
            std_hw1_t = torch.clamp(torch.as_tensor(std_hw1_t, dtype=torch.float32, device=device), min=1e-6)
            mean_hw = mean_hw1_t[..., 0]  # (H,W)
            std_hw = std_hw1_t[..., 0]
            mean_chw = mean_hw.unsqueeze(0).repeat(OBS_CHANNELS, 1, 1)
            std_chw = std_hw.unsqueeze(0).repeat(OBS_CHANNELS, 1, 1)
            obs_norm_ep = (torch.tensor(obs_img, dtype=torch.float32, device=device) - mean_chw) / std_chw
            acts_ep = torch.tensor(rollout_data["acts"][ep_idx], dtype=torch.long, device=device)
            logps_ep = torch.tensor(rollout_data["logps"][ep_idx], dtype=torch.float32, device=device)
            vals_i_ep = torch.tensor(rollout_data["vals_int"][ep_idx], dtype=torch.float32, device=device)
            vals_e_ep = torch.tensor(rollout_data["vals_ext"][ep_idx], dtype=torch.float32, device=device)
            dones_ep = torch.tensor(rollout_data["dones"][ep_idx], dtype=torch.bool, device=device)
            rwds_e_ep = torch.tensor(rollout_data["rwds_e"][ep_idx], dtype=torch.float32, device=device)
            rwds_i_raw_ep = torch.tensor(rollout_data["rwds_i"][ep_idx], dtype=torch.float32, device=device)

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

        # Set main agent back to train mode for updates
        agent.train()
        
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
            "time/fps": total_frames / iter_time,
            "data/episodes_collected": total_episodes,
            "data/frames_collected": total_frames,
        }

        with open(f'{debug_dir}/log_{iter_idx:04d}.json', 'w') as f:
            json.dump(log_dict, f)

        for k,v in log_dict.items():
            print(k, ": ", v)
        
        wandb.log(log_dict)
        print(f"Timer {iter_time:.1f}s | FPS: {total_frames/iter_time:.0f}")
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
