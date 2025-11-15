import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
import cv2

from utils import encode_images_to_latents, decode_latents_to_images
from train_doom import (
    DoomSimCfg, build_models,
    cfg_with_context_dropout, VALID_ACTIONS, buttons_to_action_one_hot
)

def load_checkpoint(ckpt_dir, device):
    cfg = DoomSimCfg()

    # Load full weights checkpoint
    print(f"Loading full weights from {ckpt_dir}/checkpoint.pt...")

    # Build models with full fine-tuning mode
    cfg_full = DoomSimCfg()
    cfg_full.use_lora = False
    vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder = build_models(cfg_full, device)

    # Load checkpoint
    checkpoint_path = os.path.join(ckpt_dir, "checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dicts
    unet.load_state_dict(checkpoint['unet'])
    conditioner.load_state_dict(checkpoint['conditioner'])
    noise_bucketer.load_state_dict(checkpoint['noise_bucketer'])

    print(f"Full weights loaded successfully! (Step {checkpoint.get('step', 'unknown')})")

    unet.eval()
    conditioner.eval()
    noise_bucketer.eval()

    return vae, unet, scheduler, conditioner, noise_bucketer, cfg

def predict_next_frame(vae, unet, scheduler, conditioner, noise_bucketer, cfg,
                       context_frames, action_history, device, model_dtype):
    """Predict next frame given context and action history.

    Args:
        context_frames: [1, N_hist, 3, H, W] - image context
        action_history: [1, N_hist, action_vocab_size] - action history buffer
    """
    with torch.no_grad():
        # Use last N_hist-1 frames and actions (matching training rollout lines 803-808)
        context_for_encoding = context_frames[:, -(cfg.n_hist - 1):]  # [1, N_hist-1, 3, H, W]
        action_context = action_history[:, -(cfg.n_hist - 1):]  # [1, N_hist-1, action_vocab_size]

        ctx = encode_images_to_latents(
            vae, context_for_encoding.flatten(0, 1), cfg.vae_scaling_factor
        ).reshape(1, cfg.n_hist - 1, cfg.latent_channels, cfg.height//8, cfg.width//8)

        # Get action conditioning using N_hist-1 actions (matching training)
        action_text_embeds, obs_mask = conditioner(action_context, None)
        timestep_cond = noise_bucketer(torch.zeros(1, dtype=torch.long, device=device))

        # Denoise
        lat = torch.randn(1, cfg.latent_channels, cfg.height // 8, cfg.width // 8,
                         device=device, dtype=model_dtype)
        scheduler.set_timesteps(cfg.num_inference_steps, device=device)

        for t in scheduler.timesteps:
            eps = cfg_with_context_dropout(
                unet, lat, ctx, t.expand(1), action_text_embeds, timestep_cond, cfg.cfg_weight
            )
            lat = scheduler.step(eps, t, lat).prev_sample

        next_frame = decode_latents_to_images(vae, lat, cfg.vae_scaling_factor)
        return next_frame

def main():
    parser = argparse.ArgumentParser(description="Interactive Doom World Model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--episode", type=str, default="debug/rnd/20251023_014026/0", help="Path to episode directory (e.g., debug/rnd/20251023_014026/0)")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame index in episode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vae, unet, scheduler, conditioner, noise_bucketer, cfg = load_checkpoint(args.ckpt, device)
    model_dtype = next(unet.parameters()).dtype
    
    # Load initial context from real episode
    print(f"Loading {cfg.n_hist} frames from {args.episode} starting at frame {args.start_frame}")
    frame_files = sorted([f for f in os.listdir(args.episode) if f.endswith(".png")])
    json_files = sorted([
        f for f in os.listdir(args.episode)
        if f.startswith("frame_") and f.endswith(".json")
    ])
    
    if len(frame_files) < args.start_frame + cfg.n_hist:
        print(f"Error: Not enough frames in episode. Need {args.start_frame + cfg.n_hist}, got {len(frame_files)}")
        return

    if len(json_files) < args.start_frame + cfg.n_hist:
        print(
            "Error: Not enough action annotations in episode. "
            f"Need {args.start_frame + cfg.n_hist}, got {len(json_files)}"
        )
        return
    
    context = []
    actions = []
    for i in range(args.start_frame, args.start_frame + cfg.n_hist):
        img = Image.open(os.path.join(args.episode, frame_files[i])).convert("RGB")
        img = img.resize((cfg.width, cfg.height), Image.BILINEAR)
        context.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)

        with open(os.path.join(args.episode, json_files[i]), "r") as f:
            buttons = json.load(f)["action"].get("buttons", [])
        actions.append(buttons_to_action_one_hot(buttons))
    
    context = torch.stack(context).unsqueeze(0).to(device=device, dtype=model_dtype)
    action_history = torch.stack(actions).unsqueeze(0).to(device=device, dtype=model_dtype)
    print(f"Loaded context from frames {args.start_frame} to {args.start_frame + cfg.n_hist - 1}")

    print(f"Loaded action history buffer: {action_history.shape}")

    print("\nControls:")
    print("  w - MOVE_FORWARD")
    print("  s - MOVE_BACKWARD")
    print("  a - TURN_LEFT")
    print("  d - TURN_RIGHT")
    print("  f - ATTACK")
    print("  e - USE")
    print("  q - QUIT")
    print("\nPress any key to start...")

    cv2.namedWindow("Doom World Model", cv2.WINDOW_NORMAL)

    step = 0
    while True:
        # Display current frame
        current_frame = context[0, -1].cpu().float().permute(1, 2, 0).numpy()
        current_frame = (current_frame * 255).clip(0, 255).astype(np.uint8)
        current_frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        current_frame_display = cv2.resize(current_frame_bgr, (640, 480))
        
        cv2.putText(current_frame_display, f"Step: {step}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Doom World Model", current_frame_display)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        
        # Map key to action
        action_buttons = []
        if key == ord('w'):
            action_buttons = ["MOVE_FORWARD"]
        elif key == ord('s'):
            action_buttons = ["MOVE_BACKWARD"]
        elif key == ord('a'):
            action_buttons = ["TURN_LEFT"]
        elif key == ord('d'):
            action_buttons = ["TURN_RIGHT"]
        elif key == ord('f'):
            action_buttons = ["ATTACK"]
        elif key == ord('e'):
            action_buttons = ["USE"]
        else:
            continue
        
        print(f"Step {step}: {action_buttons}")

        # Convert to one-hot
        action = buttons_to_action_one_hot(action_buttons).to(device=device, dtype=model_dtype)

        # Predict next frame using full context and action history
        next_frame = predict_next_frame(
            vae, unet, scheduler, conditioner, noise_bucketer, cfg,
            context, action_history, device, model_dtype
        )

        # Roll both buffers (just like in train_doom.py rollout function)
        context = torch.cat([context[:, 1:], next_frame.unsqueeze(1)], dim=1)
        action_history = torch.cat([action_history[:, 1:], action.unsqueeze(0).unsqueeze(0)], dim=1)

        step += 1
    
    cv2.destroyAllWindows()
    print(f"Total steps: {step}")

if __name__ == "__main__":
    main()

