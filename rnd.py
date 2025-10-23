import os
import random
import argparse
import vizdoom as vzd
from datetime import datetime
import json
from PIL import Image

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
OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"debug/rnd/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
os.makedirs(OUTDIR, exist_ok=True)

def setup_game():
    game = vzd.DoomGame()
    base_dir = os.path.dirname(os.path.abspath(__file__))
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

unique_labels = set()
def save_frame_info(episode_idx: int, frame_idx: int, state, act_idx: int):
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
    
    os.makedirs(os.path.join(OUTDIR, str(episode_idx)), exist_ok=True)
    
    with open(os.path.join(OUTDIR, str(episode_idx), f"frame_{frame_idx:04d}.json"), "w") as f:
        json.dump(info, f)
    
    screen = state.screen_buffer.transpose(1, 2, 0)
    if screen.shape[2] == 1:
        screen = screen.squeeze(2)
    frame_image = Image.fromarray(screen)
    frame_image.save(os.path.join(OUTDIR, str(episode_idx), f"frame_{frame_idx:04d}.png"))

    return info

def collect_random(steps: int):
    game = setup_game()
    episode_idx = 0
    game.new_episode()
    rews = []
    episode_rews = []
    frames = 0
    while frames < steps // FRAME_SKIP:
        if game.is_episode_finished():
            episode_idx += 1
            episode_rews.append(sum(rews))
            rews = []
            game.new_episode()
        state = game.get_state()
        if state is None:
            continue
        act_idx = random.randint(0, len(MACRO_ACTIONS) - 1)
        action_vector = [btn in MACRO_ACTIONS[act_idx] for btn in BUTTONS_ORDER]
        save_frame_info(episode_idx, frames, state, act_idx)
        r = game.make_action(action_vector, FRAME_SKIP)

        if r == 10:
            print("Kill")
        elif r == 50:
            print("Secret")
        elif r == 100:
            print("Map exit")
        rews.append(r)
        frames += 1
    game.close()
    return episode_rews

def main():
    parser = argparse.ArgumentParser(description="Random agent data collector for ViZDoom")
    parser.add_argument("--steps", type=int, default=10000, help="Total frame budget (after frame-skip)")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save output .npz")
    args = parser.parse_args()

    episode_rews = collect_random(args.steps)
    print(f"Total episodes: {len(episode_rews)}")
    print(f"Average episode reward: {sum(episode_rews) / len(episode_rews) if episode_rews else 0}")
    print(f"Unique labels: {unique_labels}")

if __name__ == "__main__":
    main()


