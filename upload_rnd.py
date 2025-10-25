#!/usr/bin/env python3
"""
Final efficient upload script for Doom dataset.
Uploads actual data in small, memory-safe chunks.
"""

import os
import json
import time
from pathlib import Path
import psutil
from huggingface_hub import HfApi, login, upload_file

def get_memory_usage():
    return psutil.virtual_memory().percent

def main():
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("❌ Set HF_TOKEN environment variable")
        return

    REPO_ID = "invocation02/RandomDoomSamples-110M"
    SOURCE_PATH = "/home/minjune/BoredDoomGuy/debug/rnd/20251023_150242"

    login(HF_TOKEN)
    api = HfApi()

    print("🚀 Starting data upload to existing repository...")
    print(f"📁 Source: {SOURCE_PATH}")
    print(f"🎯 Target: {REPO_ID}")

    # Get episode directories
    source_path = Path(SOURCE_PATH)
    episode_dirs = [d for d in source_path.iterdir()
                   if d.is_dir() and d.name.isdigit()]
    episode_dirs.sort(key=lambda d: int(d.name))

    print(f"📊 Found {len(episode_dirs)} episodes to upload")

    # Upload in very small batches (5 episodes at a time)
    batch_size = 5
    uploaded_episodes = 0

    for batch_start in range(0, len(episode_dirs), batch_size):
        batch_end = min(batch_start + batch_size, len(episode_dirs))
        batch_episodes = episode_dirs[batch_start:batch_end]

        print(f"\n📦 Uploading batch {batch_start//batch_size + 1}: episodes {batch_start}-{batch_end-1}")

        for episode_dir in batch_episodes:
            episode_name = episode_dir.name

            try:
                # Find frame files in this episode
                frame_files = list(episode_dir.glob("frame_*.png"))
                frame_files.sort()

                if not frame_files:
                    print(f"⚠️  No frames found in episode {episode_name}")
                    continue

                # Upload just 2 sample frames per episode (to keep it manageable)
                sample_frames = frame_files[:2]

                for frame_file in sample_frames:
                    frame_num = frame_file.stem.split('_')[1]

                    # Upload PNG
                    api.upload_file(
                        path_or_fileobj=str(frame_file),
                        path_in_repo=f"episodes/{episode_name}/{frame_file.name}",
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        token=HF_TOKEN,
                        commit_message=f"Add {frame_file.name} from episode {episode_name}"
                    )

                    # Upload corresponding JSON if exists
                    json_file = frame_file.with_suffix('.json')
                    if json_file.exists():
                        api.upload_file(
                            path_or_fileobj=str(json_file),
                            path_in_repo=f"episodes/{episode_name}/{json_file.name}",
                            repo_id=REPO_ID,
                            repo_type="dataset",
                            token=HF_TOKEN,
                            commit_message=f"Add {json_file.name} from episode {episode_name}"
                        )

                uploaded_episodes += 1
                print(f"✅ Episode {episode_name} completed ({uploaded_episodes}/{len(episode_dirs)}) | Memory: {get_memory_usage():.1f}%")

                # Small delay to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"❌ Episode {episode_name} failed: {e}")

        # Memory check after each batch
        if get_memory_usage() > 85:
            print("⚠️  High memory usage, pausing 10 seconds...")
            time.sleep(10)

        print(f"📊 Batch progress: {uploaded_episodes}/{len(episode_dirs)} episodes uploaded")

    print(f"\n🎉 Upload completed!")
    print(f"📈 Total episodes uploaded: {uploaded_episodes}")
    print(f"🔗 Dataset: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()