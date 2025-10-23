import os
import random
import argparse
import vizdoom as vzd
from datetime import datetime
import json
from PIL import Image
import io
import multiprocessing
from multiprocessing import Process

FRAME_SKIP = 4

# Minimal macro-action set (mirrors icm_rnd.py) and button ordering
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

IGNORE_LABELS = ["BulletPuff", "Column", "TechPillar", "Clip", "Shell"]
# Get project root (parent of agents/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup_game():
    game = vzd.DoomGame()
    # Look in project root (parent of agents/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wad_path = None
    for name in ("DOOM1.WAD", "Doom1.WAD"):
        cand = os.path.join(base_dir, name)
        if os.path.exists(cand):
            wad_path = cand
            break
    if wad_path:
        game.set_doom_game_path(wad_path)
    game.set_doom_map("E1M1")
    for k, v in [("death", -0.5), ("kill", 10), ("armor", 0), ("health", 0), ("map_exit", 100), ("secret", 50)]:
        getattr(game, f"set_{k}_reward")(v)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_objects_info_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args("+freelook 1")
    game.set_available_buttons(BUTTONS_ORDER)
    game.init()
    return game

class FrameCounter:
    def __init__(self, manager):
        self.count, self.lock = manager.Value('i', 0), manager.Lock()
    def get(self):
        with self.lock:
            return self.count.value
    def add(self, value):
        with self.lock:
            self.count.value += value

class EpisodeCounter:
    def __init__(self, manager):
        self.count, self.lock = manager.Value('i', 0), manager.Lock()
    def get(self):
        with self.lock:
            return self.count.value
    def get_and_increment(self):
        with self.lock:
            current = self.count.value
            self.count.value += 1
            return current

unique_labels = set()
def save_frame_info(outdir: str, episode_idx: int, frame_idx: int, state, act_idx: int):
    info = {
        "action": {
            "index": int(act_idx),
            "buttons": [btn.name for btn in MACRO_ACTIONS[act_idx]],
        },
        "objects": {},
    }
    for label in state.labels:
        if label.object_name in IGNORE_LABELS:
            continue
        info["objects"][label.object_name] = dict(
            position_x=label.object_position_x,
            position_y=label.object_position_y,
            position_z=label.object_position_z,
        )
        unique_labels.add(label.object_name)
    
    os.makedirs(os.path.join(outdir, str(episode_idx)), exist_ok=True)
    
    with open(os.path.join(outdir, str(episode_idx), f"frame_{frame_idx:04d}.json"), "w") as f:
        json.dump(info, f)
    
    screen = state.screen_buffer.transpose(1, 2, 0)
    if screen.shape[2] == 1:
        screen = screen.squeeze(2)
    frame_image = Image.fromarray(screen)
    frame_image.save(os.path.join(outdir, str(episode_idx), f"frame_{frame_idx:04d}.png"))

    return info

def run_episode(process_id, episode_id, outdir, buffer, frame_counter, max_frames_per_episode):
    """Run a single episode in a separate process."""
    game = setup_game()
    game.new_episode()
    
    rews = []
    frames_collected = 0
    frame_idx = 0
    frames_png = []
    frames_info = []
    
    while frames_collected < max_frames_per_episode:
        if game.is_episode_finished():
            break
        
        # Stop if episode is too long (>30k rewards means likely stuck)
        if len(rews) > 3e4:
            break
            
        state = game.get_state()
        if state is None:
            continue
            
        act_idx = random.randint(0, len(MACRO_ACTIONS) - 1)
        action_vector = [btn in MACRO_ACTIONS[act_idx] for btn in BUTTONS_ORDER]
        
        # Encode observation image to PNG bytes to return to parent for saving
        screen = state.screen_buffer.transpose(1, 2, 0)
        if screen.shape[2] == 1:
            screen = screen.squeeze(2)
        img = Image.fromarray(screen)
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        frames_png.append(bio.getvalue())

        # Build frame info dict (for parent to save as JSON)
        info = {
            "action": {
                "index": int(act_idx),
                "buttons": [btn.name for btn in MACRO_ACTIONS[act_idx]],
            },
            "objects": {},
        }
        for label in state.labels:
            if label.object_name in IGNORE_LABELS:
                continue
            info["objects"][label.object_name] = dict(
                position_x=label.object_position_x,
                position_y=label.object_position_y,
                position_z=label.object_position_z,
            )
        frames_info.append(info)
        
        r = game.make_action(action_vector, FRAME_SKIP)
        
        if r == 10:
            print(f"Process {process_id}: Kill")
        elif r == 50:
            print(f"Process {process_id}: Secret")
        elif r == 100:
            print(f"Process {process_id}: Map exit")
            
        rews.append(r)
        frames_collected += FRAME_SKIP
        frame_idx += 1
    
    frame_counter.add(frames_collected)
    game.close()
    
    episode_reward = sum(rews)
    buffer.put((process_id, episode_id, episode_reward, frame_idx, frames_png, frames_info))

def run_parallel_episodes(n_procs, outdir, buffer, frame_counter, episode_counter, max_frames_per_episode):
    """Launch multiple episode processes in parallel."""
    processes = []
    episode_ids = []
    for _ in range(n_procs):
        episode_ids.append(episode_counter.get_and_increment())
    for pid in range(n_procs):
        p = Process(target=run_episode, args=(pid, episode_ids[pid], outdir, buffer, frame_counter, max_frames_per_episode))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

def collect_random(steps: int, n_workers: int, outdir: str):
    """Collect random rollouts using parallel workers."""
    manager = multiprocessing.Manager()
    buffer = manager.Queue()
    frame_counter = FrameCounter(manager)
    episode_counter = EpisodeCounter(manager)
    
    episode_rews = []
    total_episodes = 0
    
    # Progress tracking
    progress_interval = 10000
    last_progress_milestone = 0
    
    # Calculate max frames per episode to ensure we don't overshoot
    max_frames_per_episode = steps // n_workers
    
    while frame_counter.get() < steps:
        run_parallel_episodes(n_workers, outdir, buffer, frame_counter, episode_counter, max_frames_per_episode)
        
        # Collect results from buffer
        while not buffer.empty():
            proc_id, episode_id, episode_reward, frame_count, frames_png, frames_info = buffer.get()
            episode_rews.append(episode_reward)
            total_episodes += 1
            print(f"Episode {episode_id} (process {proc_id}): reward={episode_reward:.1f}, frames={frame_count}")
            # Save frames to unique episode directory
            episode_dir = os.path.join(outdir, str(episode_id))
            os.makedirs(episode_dir, exist_ok=True)
            for i, png_bytes in enumerate(frames_png):
                with open(os.path.join(episode_dir, f"frame_{i:04d}.png"), 'wb') as f:
                    f.write(png_bytes)
                # Save paired JSON info for each frame
                if i < len(frames_info):
                    with open(os.path.join(episode_dir, f"frame_{i:04d}.json"), 'w') as jf:
                        json.dump(frames_info[i], jf)
        
        # Print progress every 10,000 steps
        current_frames = frame_counter.get()
        current_milestone = (current_frames // progress_interval) * progress_interval
        if current_milestone > last_progress_milestone:
            progress_pct = (current_frames / steps) * 100
            print(f"\n=== Progress: {current_frames}/{steps} frames ({progress_pct:.1f}%) | Episodes: {total_episodes} ===\n")
            last_progress_milestone = current_milestone
    
    manager.shutdown()
    return episode_rews

def main():
    parser = argparse.ArgumentParser(description="Random agent data collector for ViZDoom")
    parser.add_argument("--steps", type=int, default=10000, help="Total frame budget (after frame-skip)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: cpu_count - 4)")
    args = parser.parse_args()
    
    # Set number of workers
    n_workers = args.workers if args.workers is not None else max(1, multiprocessing.cpu_count() - 4)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(PROJECT_ROOT, f"debug/rnd/{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Starting parallel rollouts with {n_workers} workers")
    print(f"Target steps: {args.steps}")
    print(f"Output directory: {outdir}")
    
    episode_rews = collect_random(args.steps, n_workers, outdir)
    
    print(f"\n=== Summary ===")
    print(f"Total episodes: {len(episode_rews)}")
    print(f"Average episode reward: {sum(episode_rews) / len(episode_rews) if episode_rews else 0:.2f}")
    print(f"Max episode reward: {max(episode_rews) if episode_rews else 0:.2f}")
    print(f"Min episode reward: {min(episode_rews) if episode_rews else 0:.2f}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()


