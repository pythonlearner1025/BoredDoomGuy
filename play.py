import argparse
import os
import torch
import numpy as np
from PIL import Image
import cv2

from train_doom import (
    DoomSimCfg, build_models, encode_images_to_latents, 
    cfg_only_on_observation, VALID_ACTIONS, buttons_to_action_one_hot
)
from peft import PeftModel

def load_checkpoint(ckpt_dir, device):
    """Load models and LoRA adapters from checkpoint."""
    cfg = DoomSimCfg()
    
    # Build models WITHOUT LoRA first
    cfg_no_lora = DoomSimCfg()
    cfg_no_lora.use_lora = False
    vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder = build_models(cfg_no_lora, device)
    
    # Now load the trained LoRA adapters
    print(f"Loading LoRA adapters from {ckpt_dir}...")
    unet = PeftModel.from_pretrained(unet, os.path.join(ckpt_dir, "unet"))
    conditioner = PeftModel.from_pretrained(conditioner, os.path.join(ckpt_dir, "conditioner"))
    noise_bucketer = PeftModel.from_pretrained(noise_bucketer, os.path.join(ckpt_dir, "noise_bucketer"))
    
    print("LoRA adapters loaded successfully!")
    
    unet.eval()
    conditioner.eval()
    noise_bucketer.eval()
    
    return vae, unet, scheduler, conditioner, noise_bucketer, cfg

def predict_next_frame(vae, unet, scheduler, conditioner, noise_bucketer, cfg, 
                       context_frames, action, device, model_dtype):
    """Predict next frame given context and action."""
    with torch.no_grad():
        # Encode context
        ctx = encode_images_to_latents(
            vae, context_frames.flatten(0, 1), cfg.vae_scaling_factor
        ).reshape(1, cfg.n_hist - 1, cfg.latent_channels, cfg.height//8, cfg.width//8)
        
        # Get action conditioning
        action_expanded = action.unsqueeze(0).unsqueeze(0).expand(-1, cfg.n_hist - 1, -1)
        action_text_embeds, obs_mask = conditioner(action_expanded, None)
        timestep_cond = noise_bucketer(torch.zeros(1, dtype=torch.long, device=device))
        
        # Denoise
        lat = torch.randn(1, cfg.latent_channels, cfg.height // 8, cfg.width // 8, 
                         device=device, dtype=model_dtype)
        scheduler.set_timesteps(cfg.num_inference_steps, device=device)
        
        for t in scheduler.timesteps:
            x_in = torch.cat([lat, ctx.flatten(1, 2)], dim=1)
            eps = cfg_only_on_observation(
                unet, x_in, t.expand(1), action_text_embeds, obs_mask, timestep_cond, cfg.cfg_weight
            )
            lat = scheduler.step(eps, t, lat).prev_sample
        
        next_frame = vae.decode(lat / cfg.vae_scaling_factor).sample
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
    
    if len(frame_files) < args.start_frame + cfg.n_hist:
        print(f"Error: Not enough frames in episode. Need {args.start_frame + cfg.n_hist}, got {len(frame_files)}")
        return
    
    context = []
    for i in range(args.start_frame, args.start_frame + cfg.n_hist):
        img = Image.open(os.path.join(args.episode, frame_files[i])).convert("RGB")
        img = img.resize((cfg.width, cfg.height), Image.BILINEAR)
        context.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)
    
    context = torch.stack(context).unsqueeze(0).to(device=device, dtype=model_dtype)
    print(f"Loaded context from frames {args.start_frame} to {args.start_frame + cfg.n_hist - 1}")
    
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
        
        # Predict next frame
        context_for_pred = context[:, -(cfg.n_hist-1):]
        next_frame = predict_next_frame(
            vae, unet, scheduler, conditioner, noise_bucketer, cfg,
            context_for_pred, action, device, model_dtype
        )
        
        # Update context
        context = torch.cat([context[:, 1:], next_frame.unsqueeze(1)], dim=1)
        step += 1
    
    cv2.destroyAllWindows()
    print(f"Total steps: {step}")

if __name__ == "__main__":
    main()

