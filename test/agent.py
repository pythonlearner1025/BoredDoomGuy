#!/usr/bin/env python3
"""
Agent playback script for trained Arnold DQN agent.
Loads a trained model and watches it play Doom1 E1M1 in real-time.
Purpose: Automate Doom1 gameplay video collection using the Arnold agent.
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import cv2
from PIL import Image
from datetime import datetime
from collections import namedtuple

# ViZDoom
import vizdoom as vzd

# Arnold imports
sys.path.insert(0, '/home/minjune/doom/Arnold')
from src.model import get_model_class
from src.doom.actions import ActionBuilder
from src.utils import get_device_mapping

# Game state structure (same as Arnold's)
GameState = namedtuple('State', ['screen', 'variables', 'features'])


class DoomAgent:
    """Arnold DQN agent for playing Doom."""

    def __init__(self, model_path, params_path, gpu_id=0):
        """
        Initialize the agent.

        Args:
            model_path: Path to .pth model checkpoint
            params_path: Path to params.pkl from training
            gpu_id: GPU ID (-1 for CPU)
        """
        print(f"Loading parameters from {params_path}")
        with open(params_path, 'rb') as f:
            self.params = pickle.load(f)

        # Force evaluation mode settings
        self.params.gpu_id = gpu_id

        print(f"Model type: {self.params.network_type}")
        print(f"Frame size: {self.params.height}x{self.params.width}")
        print(f"History size: {self.params.hist_size}")

        # Initialize action builder
        self.action_builder = ActionBuilder(self.params)
        print(f"Action space:: {self.params.n_actions}")
        for i, action in enumerate(self.action_builder.available_actions):
            print(f"  Action {i}: {action}")

        # Load network
        print(f"\nLoading model from {model_path}")
        network_class = get_model_class(self.params.network_type)
        self.network = network_class(self.params)

        # Load weights
        map_location = get_device_mapping(gpu_id)
        checkpoint = torch.load(model_path, map_location=map_location)
        self.network.module.load_state_dict(checkpoint)
        self.network.module.eval()
        print("Model loaded successfully!")

        # State history
        self.last_states = []

    def reset(self):
        """Reset agent state for new episode."""
        self.last_states = []
        self.network.reset()

    def preprocess_frame(self, screen_buffer, variables):
        """
        Preprocess frame to match Arnold's format.

        Args:
            screen_buffer: Raw screen from ViZDoom (C, H, W) uint8
            variables: List of game variable values

        Returns:
            Processed screen (C, H, W) uint8
        """
        # Screen buffer is already in (C, H, W) format from ViZDoom
        height, width = self.params.height, self.params.width

        if self.params.gray:
            # Convert to grayscale
            screen = screen_buffer.astype(np.float32).mean(axis=0)
            # Resize if needed
            if screen.shape != (height, width):
                screen = cv2.resize(screen, (width, height), interpolation=cv2.INTER_AREA)
            screen = screen.reshape(1, height, width).astype(np.uint8)
        else:
            # Keep RGB
            if screen_buffer.shape != (3, height, width):
                # Transpose to (H, W, C), resize, transpose back
                screen = cv2.resize(
                    screen_buffer.transpose(1, 2, 0),
                    (width, height),
                    interpolation=cv2.INTER_AREA
                ).transpose(2, 0, 1)
            else:
                screen = screen_buffer

        return screen

    def observe_state(self, screen_buffer, variables):
        """
        Observe current game state and add to history.

        Args:
            screen_buffer: Raw screen from ViZDoom
            variables: List of game variable values [health, sel_ammo]
        """
        # Preprocess frame
        screen = self.preprocess_frame(screen_buffer, variables)

        # Create game state (no features for now)
        game_state = GameState(screen=screen, variables=variables, features=None)
        self.last_states.append(game_state)

        # Initialize history if first frame
        if len(self.last_states) == 1:
            self.last_states.extend([self.last_states[0]] * (self.params.hist_size - 1))
        else:
            assert len(self.last_states) == self.params.hist_size + 1
            del self.last_states[0]

    def get_action(self, debug=False):
        """
        Get next action from the model.

        Returns:
            action_id: Integer action ID
        """
        if len(self.last_states) < self.params.hist_size:
            # Return no-op until we have enough history
            return 0

        # Debug: check input
        if debug:
            print(f"\nDEBUG - State history:")
            print(f"  Number of states: {len(self.last_states)}")
            for i, state in enumerate(self.last_states):
                print(f"  State {i}: screen shape={state.screen.shape}, "
                      f"vars={state.variables}, screen min/max={state.screen.min()}/{state.screen.max()}")

                # Save debug frame to inspect
                if i == 3:  # Save last frame
                    debug_img = np.transpose(state.screen, (1, 2, 0))  # (C,H,W) -> (H,W,C)
                    from PIL import Image as PILImage
                    PILImage.fromarray(debug_img).save(f'/tmp/debug_agent_frame_{self.params.height}x{self.params.width}.png')
                    print(f"  Saved debug frame to /tmp/debug_agent_frame_{self.params.height}x{self.params.width}.png")

        # Get action from network
        with torch.no_grad():
            action_id = self.network.next_action(self.last_states)

        if debug:
            print(f"  Model output: action_id={action_id}, type={type(action_id)}")
            # Also check if model is in eval mode
            print(f"  Model training mode: {self.network.module.training}")

            # Try to get Q-values to see if model is actually running
            if hasattr(self.network, 'pred_scores'):
                print(f"  Q-values available: {self.network.pred_scores is not None}")

        return action_id

    def action_to_buttons(self, action_id):
        """
        Convert action ID to ViZDoom button array.

        Args:
            action_id: Integer action ID

        Returns:
            List of button states (0 or 1) matching ViZDoom button order
        """
        # Get Arnold's internal action representation
        doom_action = self.action_builder.doom_actions[action_id]
        return doom_action


def play_episode(agent, output_dir, max_frames=10000, save_frames=True, tick_delay=0.028):
    """
    Play one episode with the agent.

    Args:
        agent: DoomAgent instance
        output_dir: Directory to save frames
        max_frames: Maximum frames per episode
        save_frames: Whether to save frames to disk
        tick_delay: Delay between frames in seconds (for visualization)
    """
    # Initialize ViZDoom game
    print("\nInitializing Doom...")
    game = vzd.DoomGame()

    # Game setup (E1M1)
    # Need scenario path even though we're not using scenario WAD
    scenario_path = "/home/minjune/doom/Arnold/resources/scenarios/defend_the_center.wad"
    game.set_doom_scenario_path(scenario_path)
    game.set_doom_game_path("/home/minjune/Downloads/Doom1.WAD")
    game.set_doom_map("E1M1")

    # Visual settings - MUST match training resolution
    game.set_screen_resolution(vzd.ScreenResolution.RES_400X225)  # Match Arnold training
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)  # RGB format
    game.set_render_hud(False)  # Match training (render_hud=False)
    game.set_render_crosshair(True)
    game.set_render_weapon(True)

    # Enable window for viewing
    game.set_window_visible(True)

    # Set mode to PLAYER (agent plays, not spectator)
    # NOTE: In ViZDoom, PLAYER = bot plays, SPECTATOR = human plays
    game.set_mode(vzd.Mode.PLAYER)

    # Enable labels for object detection (optional for logging)
    game.set_labels_buffer_enabled(True)

    # Available buttons - MUST match Arnold's exact order
    # Arnold's button order: MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT,
    #                        MOVE_LEFT, MOVE_RIGHT, ATTACK, SPEED, CROUCH
    # Then add_buttons() appends SELECT_WEAPON0-9 (10 more buttons)
    # Total: 19 buttons
    from src.doom.actions import add_buttons

    # Don't call set_available_buttons directly - let Arnold's add_buttons do it
    # This ensures button order matches training exactly
    button_mapping = add_buttons(game, agent.action_builder.available_buttons)

    print(f"\nButton mapping ({len(button_mapping)} buttons):")
    for btn, idx in sorted(button_mapping.items(), key=lambda x: x[1])[:10]:
        print(f"  {idx}: {btn}")

    # Add game args (match Arnold training exactly)
    game.add_game_args("-host 1")  # Single player host
    game.add_game_args("+sv_respawnprotect 1")
    game.add_game_args("+sv_spawnfarthest 1")
    game.add_game_args("+freelook 0")  # Disable freelook (match training)
    game.add_game_args("+name Arnold")
    game.add_game_args("+colorset 0")
    game.add_game_args("+sv_cheats 1")
    #game.add_game_args("-deathmatch")  # Commented out - single player mode
    #game.add_game_args("+sv_forcerespawn 1")
    #game.add_game_args("+sv_noautoaim 1")

    # Doom skill (Arnold uses doom_skill=2, which means skill level 3)
    game.set_doom_skill(3)

    # Initialize game
    game.init()
    print("Doom initialized!")

    # Start episode
    print("\nStarting episode...")
    game.new_episode()
    agent.reset()

    # CRITICAL: Skip 3 initial frames (match Arnold's SKIP_INITIAL_ACTIONS)
    # This avoids bugs from initial weapon changes
    game.advance_action(3)
    print("Skipped 3 initial frames (weapon initialization)")

    frame_idx = 0
    episode_reward = 0
    episode_kills = 0

    while not game.is_episode_finished() and frame_idx < max_frames:
        # Get current state
        state = game.get_state()
        if state is None:
            break

        # Extract screen and variables
        screen_buffer = state.screen_buffer  # (C, H, W) uint8

        # Get game variables (match training: health, sel_ammo)
        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        sel_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        variables = [health, sel_ammo]

        # Agent observes state
        agent.observe_state(screen_buffer, variables)

        # Get action from agent
        debug_mode = (frame_idx == 0 or frame_idx == 100)  # Debug first and 100th frame
        action_id = agent.get_action(debug=debug_mode)
        action_buttons = agent.action_to_buttons(action_id)

        # Log every 100 frames
        if frame_idx % 100 == 0:
            pos_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
            pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
            print(f"Frame {frame_idx}: Action={action_id} {agent.action_builder.available_actions[action_id]}, "
                  f"Health={health:.0f}, Pos=({pos_x:.0f}, {pos_y:.0f})")

        # Execute action with frame skip (match training)
        frame_skip = agent.params.frame_skip
        reward = game.make_action(action_buttons, frame_skip)
        episode_reward += reward

        # Track kills
        if reward >= 9.5:  # Kill detected
            episode_kills += 1
            print(f"  🎯 KILL! Total: {episode_kills}")

        # Save frame if requested
        if save_frames and frame_idx % 10 == 0:  # Save every 10th frame
            rgb = np.moveaxis(screen_buffer, 0, -1)  # (C,H,W) -> (H,W,C)
            img = Image.fromarray(rgb)
            img.save(f"{output_dir}/frame_{frame_idx:06d}.png")

        frame_idx += 1

        # Visualization delay
        time.sleep(tick_delay)

    # Episode finished
    total_reward = game.get_total_reward()
    print(f"\n{'='*60}")
    print(f"Episode finished!")
    print(f"Frames: {frame_idx}")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Episode reward: {episode_reward:.1f}")
    print(f"Kills: {episode_kills}")
    print(f"Level completed: {game.is_episode_finished()}")
    print(f"{'='*60}\n")

    game.close()

    return {
        'frames': frame_idx,
        'total_reward': total_reward,
        'episode_reward': episode_reward,
        'kills': episode_kills,
        'completed': game.is_episode_finished()
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Watch trained Arnold agent play Doom1 E1M1')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--params', type=str, required=True,
                        help='Path to params.pkl file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for frames (default: auto-generated)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--max-frames', type=int, default=10000,
                        help='Maximum frames per episode')
    parser.add_argument('--save-frames', action='store_true',
                        help='Save frames to disk')
    parser.add_argument('--tick-delay', type=float, default=0.028,
                        help='Delay between frames (seconds) for visualization speed')
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to play')

    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        args.output_dir = f'agent_playback/{timestamp}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Initialize agent
    print("\n" + "="*60)
    print("ARNOLD DOOM AGENT - PLAYBACK MODE")
    print("="*60)

    agent = DoomAgent(
        model_path=args.model,
        params_path=args.params,
        gpu_id=args.gpu_id
    )

    # Play episodes
    all_stats = []
    for ep in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {ep + 1} / {args.num_episodes}")
        print(f"{'='*60}")

        episode_dir = os.path.join(args.output_dir, f'episode_{ep:03d}')
        if args.save_frames:
            os.makedirs(episode_dir, exist_ok=True)

        stats = play_episode(
            agent=agent,
            output_dir=episode_dir if args.save_frames else args.output_dir,
            max_frames=args.max_frames,
            save_frames=args.save_frames,
            tick_delay=args.tick_delay
        )
        all_stats.append(stats)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for i, stats in enumerate(all_stats):
        print(f"Episode {i+1}: {stats['frames']} frames, "
              f"{stats['total_reward']:.1f} reward, "
              f"{stats['kills']} kills, "
              f"{'COMPLETED' if stats['completed'] else 'INCOMPLETE'}")

    avg_reward = np.mean([s['total_reward'] for s in all_stats])
    avg_kills = np.mean([s['kills'] for s in all_stats])
    completion_rate = np.mean([s['completed'] for s in all_stats]) * 100

    print(f"\nAverages:")
    print(f"  Reward: {avg_reward:.1f}")
    print(f"  Kills: {avg_kills:.1f}")
    print(f"  Completion rate: {completion_rate:.0f}%")
    print("="*60)


if __name__ == "__main__":
    main()
