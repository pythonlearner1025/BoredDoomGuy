"""
GameNGen Agent Implementation (Appendix 4.1)

Implements PPO-based agent training for ViZDoom following the GameNGen paper.
Retains all data-saving logic from rnd.py for generating training data.
"""

import os
import random
import argparse
import vizdoom as vzd
from datetime import datetime
import json
from PIL import Image
import io
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import gymnasium as gym

FRAME_SKIP = 4

# Minimal macro-action set (mirrors rnd.py)
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GameNGenFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor following GameNGen specification (4.1).
    
    Processes:
    - Frame image (160x120)
    - In-game map (160x120) - using automap buffer
    - Last 32 actions (embedded)
    
    Outputs 512-dim feature vector per image + action history.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        # Total feature dim: 512 (frame) + 512 (map) + 32 (actions) = 1056
        super().__init__(observation_space, features_dim=1056)
        
        # CNN for processing frame (following Mnih et al. 2015)
        # Input: (1, 160, 120) grayscale
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 160, 120)
            cnn_output_size = self.frame_cnn(sample_input).shape[1]
        
        # Linear layer to project CNN output to 512
        self.frame_fc = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU()
        )
        
        # CNN for processing automap (same architecture)
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.map_fc = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU()
        )
        
        # Action history is already a vector of 32 elements (one-hot encoded actions)
        # No additional processing needed
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        frame = observations["frame"]  # (batch, 1, 160, 120)
        automap = observations["automap"]  # (batch, 1, 160, 120)
        actions = observations["action_history"]  # (batch, 32)
        
        # Process frame
        frame_features = self.frame_cnn(frame)
        frame_features = self.frame_fc(frame_features)  # (batch, 512)
        
        # Process automap
        map_features = self.map_cnn(automap)
        map_features = self.map_fc(map_features)  # (batch, 512)
        
        # Concatenate all features
        combined = torch.cat([frame_features, map_features, actions], dim=1)  # (batch, 1056)
        
        return combined


class DoomEnvWrapper(gym.Env):
    """
    Gymnasium wrapper for ViZDoom environment.
    Provides observations matching GameNGen specification.
    """
    
    def __init__(self, save_data: bool = False, episode_id: int = 0, outdir: str = ""):
        super().__init__()
        
        self.save_data = save_data
        self.episode_id = episode_id
        self.outdir = outdir
        self.frame_idx = 0
        
        # Action space: discrete actions
        self.action_space = spaces.Discrete(len(MACRO_ACTIONS))
        
        # Observation space: frame, automap, action history
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(low=0, high=255, shape=(1, 160, 120), dtype=np.uint8),
            "automap": spaces.Box(low=0, high=255, shape=(1, 160, 120), dtype=np.uint8),
            "action_history": spaces.Box(low=0, high=len(MACRO_ACTIONS)-1, shape=(32,), dtype=np.int32),
        })
        
        # Action history buffer
        self.action_history = deque(maxlen=32)
        for _ in range(32):
            self.action_history.append(0)  # Initialize with no-op
        
        self.game = self._setup_game()
        
    def _setup_game(self):
        game = vzd.DoomGame()
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
        
        # Reward functions (can be adjusted per section 4.5 of paper)
        for k, v in [("death", -0.5), ("kill", 10), ("armor", 0), ("health", 0), 
                     ("map_exit", 100), ("secret", 50)]:
            getattr(game, f"set_{k}_reward")(v)
        
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_automap_buffer_enabled(True)  # Enable automap for GameNGen
        game.set_objects_info_enabled(True)
        game.set_labels_buffer_enabled(True)
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.add_game_args("+freelook 1")
        game.set_available_buttons(BUTTONS_ORDER)
        game.init()
        return game
    
    def _get_observation(self, state) -> Dict[str, np.ndarray]:
        """Extract and process observation from game state."""
        if state is None:
            # Return zero observation
            return {
                "frame": np.zeros((1, 160, 120), dtype=np.uint8),
                "automap": np.zeros((1, 160, 120), dtype=np.uint8),
                "action_history": np.array(list(self.action_history), dtype=np.int32),
            }
        
        # Get screen buffer and downscale to 160x120
        screen = state.screen_buffer  # (C, H, W)
        if screen.shape[0] == 3:  # RGB to grayscale
            screen = np.mean(screen, axis=0, keepdims=True)
        screen = screen.astype(np.uint8)
        
        # Downscale using PIL
        img = Image.fromarray(screen[0])
        img_resized = img.resize((160, 120), Image.BILINEAR)
        frame = np.array(img_resized, dtype=np.uint8)[np.newaxis, ...]  # (1, 160, 120)
        
        # Get automap buffer and downscale
        automap = state.automap_buffer if state.automap_buffer is not None else np.zeros_like(screen)
        if automap.shape[0] == 3:
            automap = np.mean(automap, axis=0, keepdims=True)
        automap = automap.astype(np.uint8)
        
        automap_img = Image.fromarray(automap[0])
        automap_resized = automap_img.resize((160, 120), Image.BILINEAR)
        automap_arr = np.array(automap_resized, dtype=np.uint8)[np.newaxis, ...]
        
        return {
            "frame": frame,
            "automap": automap_arr,
            "action_history": np.array(list(self.action_history), dtype=np.int32),
        }
    
    def _save_frame_data(self, state, action_idx: int):
        """Save frame data (same logic as rnd.py)."""
        if not self.save_data or state is None:
            return
        
        info = {
            "action": {
                "index": int(action_idx),
                "buttons": [btn.name for btn in MACRO_ACTIONS[action_idx]],
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
        
        episode_dir = os.path.join(self.outdir, str(self.episode_id))
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save JSON
        with open(os.path.join(episode_dir, f"frame_{self.frame_idx:04d}.json"), "w") as f:
            json.dump(info, f)
        
        # Save frame image (original resolution)
        screen = state.screen_buffer.transpose(1, 2, 0)
        if screen.shape[2] == 1:
            screen = screen.squeeze(2)
        frame_image = Image.fromarray(screen)
        frame_image.save(os.path.join(episode_dir, f"frame_{self.frame_idx:04d}.png"))
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.game.new_episode()
        self.frame_idx = 0
        self.action_history.clear()
        for _ in range(32):
            self.action_history.append(0)
        
        state = self.game.get_state()
        obs = self._get_observation(state)
        
        return obs, {}
    
    def step(self, action: int):
        """Execute action and return observation, reward, done, info."""
        # Update action history
        self.action_history.append(action)
        
        # Convert action index to button vector
        action_vector = [btn in MACRO_ACTIONS[action] for btn in BUTTONS_ORDER]
        
        # Get state before action for saving
        state_before = self.game.get_state()
        
        # Execute action
        reward = self.game.make_action(action_vector, FRAME_SKIP)
        
        # Save frame data if enabled
        if self.save_data and state_before is not None:
            self._save_frame_data(state_before, action)
            self.frame_idx += 1
        
        # Check if episode is done
        done = self.game.is_episode_finished()
        
        # Get new observation
        state = self.game.get_state() if not done else None
        obs = self._get_observation(state)
        
        info = {}
        
        return obs, reward, done, False, info
    
    def close(self):
        self.game.close()


class DataSavingCallback(BaseCallback):
    """
    Callback to track training progress and manage data saving.
    """
    
    def __init__(self, outdir: str, verbose: int = 1):
        super().__init__(verbose)
        self.outdir = outdir
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                self.episode_rewards.append(ep_info['r'])
                if self.verbose > 0:
                    print(f"Episode reward: {ep_info['r']:.2f}")


def make_env(rank: int, save_data: bool = False, outdir: str = ""):
    """Create a single environment."""
    def _init():
        env = DoomEnvWrapper(save_data=save_data, episode_id=rank, outdir=outdir)
        return env
    return _init


def train_gamegen_agent(
    total_timesteps: int = 50_000_000,
    n_envs: int = 8,
    save_data: bool = False,
    outdir: str = "",
    checkpoint_dir: str = "",
):
    """
    Train GameNGen agent following specification in Appendix 4.1.
    
    Args:
        total_timesteps: Total environment steps (default: 50M)
        n_envs: Number of parallel environments (default: 8)
        save_data: Whether to save episode data
        outdir: Directory for saving episode data
        checkpoint_dir: Directory for saving model checkpoints
    """
    
    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, save_data, outdir) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, save_data, outdir)])
    
    # PPO policy kwargs
    policy_kwargs = dict(
        features_extractor_class=GameNGenFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[512, 512], vf=[512, 512])],  # 2-layer MLP for actor and critic
        activation_fn=nn.ReLU,
    )
    
    # Create PPO agent with GameNGen hyperparameters
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=512,  # Replay buffer size per environment
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,  # Standard GAE parameter
        ent_coef=0.1,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(PROJECT_ROOT, "tensorboard_logs"),
    )
    
    # Setup callback
    callback = DataSavingCallback(outdir=outdir, verbose=1)
    
    # Train the agent
    print(f"Starting training for {total_timesteps:,} steps with {n_envs} parallel environments")
    print(f"Model parameters: lr=1e-4, buffer_size=512, batch_size=64, epochs=10, gamma=0.99, ent_coef=0.1")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    
    # Save final model
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, "gamegen_final.zip")
        model.save(final_path)
        print(f"Saved final model to {final_path}")
    
    env.close()
    
    return model, callback.episode_rewards


def main():
    parser = argparse.ArgumentParser(description="GameNGen agent training (PPO)")
    parser.add_argument("--steps", type=int, default=50_000_000, 
                       help="Total timesteps for training (default: 50M)")
    parser.add_argument("--n_envs", type=int, default=8,
                       help="Number of parallel environments (default: 8)")
    parser.add_argument("--save_data", action="store_true",
                       help="Save episode frames and metadata during training")
    parser.add_argument("--checkpoint_dir", type=str, default="",
                       help="Directory to save model checkpoints")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(PROJECT_ROOT, f"debug/gamegen/{timestamp}")
    
    if args.save_data:
        os.makedirs(outdir, exist_ok=True)
        print(f"Data will be saved to: {outdir}")
    
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else os.path.join(
        PROJECT_ROOT, f"checkpoints/gamegen/{timestamp}"
    )
    
    model, episode_rewards = train_gamegen_agent(
        total_timesteps=args.steps,
        n_envs=args.n_envs,
        save_data=args.save_data,
        outdir=outdir,
        checkpoint_dir=checkpoint_dir,
    )
    
    print("\n=== Training Summary ===")
    print(f"Total episodes: {len(episode_rewards)}")
    if episode_rewards:
        print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
        print(f"Max episode reward: {np.max(episode_rewards):.2f}")
        print(f"Min episode reward: {np.min(episode_rewards):.2f}")


if __name__ == "__main__":
    main()

