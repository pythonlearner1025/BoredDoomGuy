#!/usr/bin/env python3

#####################################################################
# Test script for human to interact with doom via mouse / keyboard,
# and save frames. The frames will be ran through VLM to choose the best
# captioning model. The best captioning model will then be used to 
# create labeled dataset of DOOM gameplay, which will be used to train 
# text conditioning of SD 1.5 modified to act as a real-time game engine (build on GameNGen work by GDM, but with promptable game events)
#####################################################################
from vizdoom import DoomGame
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution

import os
from argparse import ArgumentParser
from random import choice
from PIL import Image, ImageDraw, ImageFont
import time

import matplotlib.pyplot as plt
import numpy as np
import vizdoom as vzd
import cv2
import json


DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "my_way_home.cfg")


def find_label_position(x, y, text_bbox, placed_boxes, max_r=50):
    """Find non-overlapping position for label text."""
    w, h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    for r in range(0, max_r, 10):
        best_pos, min_overlap = None, float('inf')
        for angle in [i * 40 for i in range(9)]:  # 9 points around circle
            rad = np.radians(angle)
            nx, ny = int(x + r * np.cos(rad)), int(y + r * np.sin(rad))

            # Check overlap
            overlap = sum((max(0, min(nx + w, bx + bw) - max(nx, bx)) *
                          max(0, min(ny + h, by + bh) - max(ny, by)))
                         for bx, by, bw, bh in placed_boxes)

            if overlap == 0:
                return nx, ny
            if overlap < min_overlap:
                best_pos, min_overlap = (nx, ny), overlap

        if r == 0 and min_overlap == 0:
            return best_pos

    return best_pos if best_pos else (x, y)


def save_frame_with_labels(img, state, frame_idx, action_data, output_dir="screens", blacklist=None):
    """
    Save frame with labels and annotations.

    Args:
        img: PIL Image of the frame
        state: ViZDoom game state
        frame_idx: Frame index for naming
        action_data: Dictionary containing action information for this frame
        output_dir: Output directory
        blacklist: Set of object names to exclude from labeling (e.g., {"Clip", "DoomPlayer"})
    """
    if blacklist is None:
        blacklist = set()

    # Save raw frame
    img.save(f"{output_dir}/frame_{frame_idx:06d}.png")

    # Create annotated version
    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)

    # Load font (50% smaller)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 9)
    except:
        font = ImageFont.load_default()

    labels_data = []
    yolo_labels = []
    placed_boxes = []

    img_width, img_height = img.size

    for label in state.labels:
        # Skip blacklisted objects completely (don't save to JSON)
        if label.object_name in blacklist:
            continue

        # Save to labels_data (blacklist already filtered above)
        labels_data.append({
            'object_id': label.object_id,
            'object_name': label.object_name,
            'x': label.x,
            'y': label.y,
            'width': label.width,
            'height': label.height,
            'position_x': label.object_position_x,
            'position_y': label.object_position_y,
            'position_z': label.object_position_z,
        })

        # Draw bounding box
        x1, y1 = label.x, label.y
        x2, y2 = label.x + label.width, label.y + label.height

        # Color code: blue for items, red for others
        color = (255, 0, 0)
        if 'Armor' in label.object_name or 'Medkit' in label.object_name or 'Stimpack' in label.object_name:
            color = (0, 0, 255)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

        # Find non-overlapping position for label text
        text = label.object_name
        bbox = draw.textbbox((0, 0), text, font=font)
        tx, ty = find_label_position(x1, y1 - 18, bbox, placed_boxes)

        # Draw label text
        bbox = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

        placed_boxes.append((bbox[0]-2, bbox[1]-2, bbox[2]-bbox[0]+4, bbox[3]-bbox[1]+4))

        # YOLO format (only non-blacklisted)
        x_center = (label.x + label.width / 2) / img_width
        y_center = (label.y + label.height / 2) / img_height
        norm_width = label.width / img_width
        norm_height = label.height / img_height

        yolo_labels.append(f"{label.object_name} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

    # Save annotated image
    annotated_img.save(f"{output_dir}/frame_{frame_idx:06d}_annotated.png")

    # Save JSON labels with action data
    frame_data = {
        'frame_idx': frame_idx,
        'actions': action_data,
        'labels': labels_data
    }

    with open(f"{output_dir}/frame_{frame_idx:06d}_labels.json", 'w') as f:
        json.dump(frame_data, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser(
        "ViZDoom example showing how to use information about objects and map."
    )
    parser.add_argument(
        dest="config",
        default=DEFAULT_CONFIG,
        nargs="?",
        help="Path to the configuration file of the scenario."
        " Please see "
        "../../scenarios/*cfg for more scenarios.",
    )

    args = parser.parse_args()

    game = vzd.DoomGame()
    game.set_doom_game_path("/home/minjune/Downloads/Doom1.WAD")
    #game.set_doom_scenario_path("")
    game.set_doom_map("E1M1")
    game.set_render_hud(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_SPECTATOR)

    # Enable freelook for mouse turning
    game.add_game_args("+freelook 1")

    # Enable labels buffer for object detection ground truth
    game.set_labels_buffer_enabled(True)

    # Enable objects info to get all objects in level
    game.set_objects_info_enabled(True)

    # CRITICAL: Must set available buttons for keyboard input to work!
    # Even in ASYNC_SPECTATOR mode, ViZDoom disables all buttons by default,
    # then only enables the ones you specify here.
    game.set_available_buttons([
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.TURN_LEFT_RIGHT_DELTA,  # For mouse turning
        vzd.Button.ATTACK,
        vzd.Button.USE,
        vzd.Button.SPEED,
        vzd.Button.STRAFE,
        vzd.Button.JUMP,
        vzd.Button.CROUCH,
    ])

    game.init()
    game.new_episode()
    print("new ep")
    os.makedirs("screens", exist_ok=True)

    # Blacklist objects you don't want to label
    blacklist = {"Clip", "DoomPlayer", "Blood", "BulletPuff"}
    
    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H%M")
    output_dir = f'screens/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Button mapping for action capture
    button_names = {
        vzd.Button.MOVE_FORWARD: "MOVE_FORWARD",
        vzd.Button.MOVE_BACKWARD: "MOVE_BACKWARD",
        vzd.Button.MOVE_LEFT: "MOVE_LEFT",
        vzd.Button.MOVE_RIGHT: "MOVE_RIGHT",
        vzd.Button.TURN_LEFT: "TURN_LEFT",
        vzd.Button.TURN_RIGHT: "TURN_RIGHT",
        vzd.Button.TURN_LEFT_RIGHT_DELTA: "TURN_LEFT_RIGHT_DELTA",
        vzd.Button.ATTACK: "ATTACK",
        vzd.Button.USE: "USE",
        vzd.Button.SPEED: "SPEED",
        vzd.Button.STRAFE: "STRAFE",
        vzd.Button.JUMP: "JUMP",
        vzd.Button.CROUCH: "CROUCH",
    }

    frame_idx = 0
    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            break

        print(f'Frame {frame_idx}')
        screen = state.screen_buffer
        rgb = np.moveaxis(screen, 0, -1)

        # Capture current action state
        # In ASYNC_SPECTATOR mode, we query button states directly
        action_data = {
            'button_states': {},
            'game_variables': {}
        }

        # Get button states by checking if each button is pressed
        for button, name in button_names.items():
            try:
                # Get the button state (pressed or not)
                is_pressed = game.get_button(button)
                action_data['button_states'][name] = 1.0 if is_pressed else 0.0
            except:
                # If button state can't be read, default to 0
                action_data['button_states'][name] = 0.0

        # Capture relevant game variables for context
        try:
            action_data['game_variables']['health'] = game.get_game_variable(vzd.GameVariable.HEALTH)
            action_data['game_variables']['ammo'] = game.get_game_variable(vzd.GameVariable.AMMO2)
            action_data['game_variables']['position_x'] = game.get_game_variable(vzd.GameVariable.POSITION_X)
            action_data['game_variables']['position_y'] = game.get_game_variable(vzd.GameVariable.POSITION_Y)
            action_data['game_variables']['position_z'] = game.get_game_variable(vzd.GameVariable.POSITION_Z)
            action_data['game_variables']['angle'] = game.get_game_variable(vzd.GameVariable.ANGLE)
        except:
            pass  # Some variables might not be available

        # Save frame with labels and actions
        img = Image.fromarray(rgb)
        save_frame_with_labels(img, state, frame_idx, action_data, output_dir=output_dir, blacklist=blacklist)

        # CRITICAL: Must call advance_action() to process game tics
        # even in ASYNC_SPECTATOR mode. This allows the game to progress
        # and process your keyboard/mouse inputs from the game window.
        game.advance_action()

        frame_idx += 1
        time.sleep(0.028)

    game.close()
