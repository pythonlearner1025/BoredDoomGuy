#!/usr/bin/env python3
"""
Simplified agent playback using Arnold's Game wrapper directly.
"""

import os
import sys
import time
import pickle
import torch
import numpy as np
from datetime import datetime

# Arnold imports
sys.path.insert(0, '/home/minjune/doom/Arnold')
from src.model import get_model_class
from src.doom.actions import ActionBuilder
from src.doom.game import Game
from src.utils import get_device_mapping


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Watch trained Arnold agent using Arnold Game wrapper')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--params', type=str, required=True, help='Path to params.pkl')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--duration', type=int, default=300, help='Play duration in seconds')

    args = parser.parse_args()

    # Load params
    print("Loading parameters...")
    with open(args.params, 'rb') as f:
        params = pickle.load(f)

    params.gpu_id = args.gpu_id

    # Create action builder
    action_builder = ActionBuilder(params)
    print(f"Actions: {params.n_actions}")

    # Create game with Arnold's wrapper (exactly as training does)
    print("Creating game...")
    game = Game(
        scenario='defend_the_center',
        action_builder=action_builder,
        reward_values={'KILL': 10.0, 'LEVEL_COMPLETE': 500.0},
        score_variable='KILLCOUNT',
        freedoom=False,
        use_screen_buffer=params.use_screen_buffer,
        use_depth_buffer=params.use_depth_buffer,
        labels_mapping=params.labels_mapping,
        game_features=params.game_features,
        mode='PLAYER',
        player_rank=params.player_rank,
        players_per_game=params.players_per_game,
        render_hud=params.render_hud,
        render_crosshair=params.render_crosshair,
        render_weapon=params.render_weapon,
        freelook=params.freelook,
        visible=True,  # Show window
        n_bots=0,
        use_scripted_marines=False,
        exit_coords=(2944, -4832),
        exit_proximity_threshold=200
    )

    # Load network
    print(f"Loading model from {args.model}")
    network_class = get_model_class(params.network_type)
    network = network_class(params)

    map_location = get_device_mapping(args.gpu_id)
    checkpoint = torch.load(args.model, map_location=map_location)
    network.module.load_state_dict(checkpoint)
    network.module.eval()
    print("Model loaded!")

    # Start game
    print("\nStarting episode on E1M1...")
    game.start(map_id=1, log_events=True, manual_control=False)
    network.reset()

    last_states = []
    n_iter = 0
    max_iters = args.duration * 35 // params.frame_skip  # Convert seconds to iterations

    print(f"Playing for {args.duration} seconds (~{max_iters} iterations)...")
    print("=" * 60)

    while n_iter < max_iters:
        n_iter += 1

        # Check if dead and respawn
        if game.is_player_dead():
            print(f"\n💀 Agent died at iteration {n_iter}")
            game.respawn_player()
            network.reset()
            last_states = []

        # Check if episode finished
        if game.is_episode_finished():
            print(f"\n🏆 Level completed at iteration {n_iter}!")
            game.game.new_episode()
            network.reset()
            last_states = []

        # Observe state (using Arnold's method)
        game.observe_state(params, last_states)

        # Get action from network
        action = network.next_action(last_states)

        # Make action
        game.make_action(action, params.frame_skip, sleep=0.028)

        # Log progress
        if n_iter % 100 == 0:
            pos_x = game.properties['position_x']
            pos_y = game.properties['position_y']
            health = game.properties['health']
            print(f"Iter {n_iter}: Action={action} {action_builder.available_actions[action]}, "
                  f"Health={health}, Pos=({pos_x:.0f}, {pos_y:.0f}), Reward={game.reward:.2f}")

    print("\n" + "=" * 60)
    print("Playback complete!")
    game.close()


if __name__ == "__main__":
    main()
