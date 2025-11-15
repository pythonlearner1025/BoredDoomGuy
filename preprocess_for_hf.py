#!/usr/bin/env python3
"""
Preprocess Doom dataset into HuggingFace Dataset format.

PARALLEL STREAMING APPROACH:
- 8 workers process episodes in parallel (configurable)
- Each worker streams frames one-by-one to Arrow (no accumulation)
- Each chunk = one episode's frames (stored as individual rows)
- Schema: one row per frame (not per episode)
- Uploads are serialized to avoid Hub API conflicts

This handles episodes with 10K+ frames without OOM and processes 8x faster.

Usage:
    # Set your token in environment (recommended):
    export HF_TOKEN=your_token_here
    
    python preprocess_for_hf.py \
        --input_dir debug/rnd/20251023_150242 \
        --output_dir doom_dataset_processed \
        --target_height 120 \
        --target_width 160 \
        --hf_repo_id invocation02/doom-dataset-1 \
        --num_workers 8
"""

import os
import json
import argparse
import uuid
from io import BytesIO
import gc
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image
from datasets import Features, Array3D, Sequence, Value
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError
from datasets.arrow_writer import ArrowWriter
from concurrent.futures import ThreadPoolExecutor
import threading

VALID_ACTIONS = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD", 
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
    "USE",
]

def buttons_to_action_one_hot(buttons):
    """Convert button list to one-hot encoded action vector."""
    one_hot = np.zeros(len(VALID_ACTIONS), dtype=np.float32)
    for button in buttons:
        if button in VALID_ACTIONS:
            one_hot[VALID_ACTIONS.index(button)] = 1.0
    return one_hot


def resolve_hf_token(token_override: Optional[str]) -> Optional[str]:
    """Resolve HuggingFace token from explicit override or environment."""
    if token_override:
        return token_override
    for env_key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val
    return None


def load_json_from_hub(
    repo_id: str,
    filename: str,
    *,
    token: Optional[str],
    revision: str,
) -> Optional[Dict]:
    """Load a JSON file from the HuggingFace Hub if it exists."""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            revision=revision,
            local_dir_use_symlinks=False,
        )
    except HfHubHTTPError as err:
        if err.response is not None and err.response.status_code == 404:
            return None
        raise
    except FileNotFoundError:
        return None

    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_initial_state(split: str) -> Dict:
    """Return a fresh dataset state descriptor compatible with load_from_disk."""
    return {
        "_data_files": [],
        "_fingerprint": uuid.uuid4().hex,
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": split,
    }


def build_initial_dataset_info(features: Features, config_name: str) -> Dict:
    """Return baseline dataset_info metadata for Hub uploads."""
    return {
        "citation": "",
        "description": "",
        "homepage": "",
        "license": "",
        "features": features.to_dict(),
        "builder_name": "preprocess_for_hf",
        "config_name": config_name,
        "splits": {},
        "download_size": 0,
        "dataset_size": 0,
    }


def json_to_bytes(data: Dict) -> BytesIO:
    """Utility to convert dicts to BytesIO for hub commits."""
    return BytesIO(json.dumps(data, indent=2, sort_keys=True).encode("utf-8"))


def process_single_episode(
    episode_dir: Path,
    target_height: int,
    target_width: int,
    arrow_path: str,
    features: Features,
):
    """
    Process one episode: stream frames one at a time to Arrow file.
    Each frame is written immediately, no accumulation in memory.
    Returns episode_id and frame count.
    """
    # Get sorted frame files
    frame_files = sorted([f for f in os.listdir(episode_dir) if f.endswith('.png')])
    json_files = sorted([f for f in os.listdir(episode_dir) if f.startswith('frame_') and f.endswith('.json')])
    
    if len(frame_files) != len(json_files):
        print(f"  Warning: Mismatch in {episode_dir}: {len(frame_files)} PNGs vs {len(json_files)} JSONs")
    
    n_frames = min(len(frame_files), len(json_files))
    if n_frames == 0:
        return None
    
    episode_id = episode_dir.name
    
    # Write frames one at a time to Arrow
    with ArrowWriter(features=features, path=arrow_path, writer_batch_size=1) as writer:
        for i in range(n_frames):
            # Load and resize image
            img = Image.open(episode_dir / frame_files[i]).convert('RGB')
            img = img.resize((target_width, target_height), Image.BILINEAR)
            frame_array = np.array(img, dtype=np.uint8)
            
            # Load action
            with open(episode_dir / json_files[i], 'r') as f:
                buttons = json.load(f)['action']['buttons']
                action_array = buttons_to_action_one_hot(buttons)
            
            # Write single frame immediately
            writer.write_batch(
                {
                    'episode_id': [episode_id],
                    'frame_idx': [i],
                    'frame': [frame_array],
                    'action': [action_array],
                },
                writer_batch_size=1,
            )
            
            # Free everything immediately
            del img, frame_array, action_array
            
            # Periodic garbage collection every 100 frames
            if i % 100 == 0:
                gc.collect()
        
        written_examples, _ = writer.finalize()
    
    gc.collect()
    
    return episode_id, n_frames


def preprocess_dataset(
    input_dir: str,
    output_dir: str, 
    target_height: int = 120,
    target_width: int = 160,
    *,
    hf_repo_id: str,
    hf_token: Optional[str] = None,
    hf_branch: str = "main",
    hf_split: str = "train",
    hf_config: str = "default",
    hf_private: bool = False,
    num_workers: int = 8,
):
    """Process episodes in parallel and upload immediately."""

    if not hf_repo_id:
        raise ValueError("hf_repo_id must be provided")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(
        d for d in input_path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    )

    print(f"Found {len(episode_dirs)} episodes in {input_dir}")
    print(f"Target resolution: {target_height}x{target_width}")
    print(f"Processing with {num_workers} parallel workers, streaming frames to avoid OOM...")

    # Schema: one row per frame (not per episode)
    features = Features({
        'episode_id': Value('string'),
        'frame_idx': Value('int32'),
        'frame': Array3D(dtype='uint8', shape=(target_height, target_width, 3)),
        'action': Sequence(feature=Value('float32'), length=len(VALID_ACTIONS)),
    })
    features_dict = features.to_dict()

    resolved_token = resolve_hf_token(hf_token)
    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=hf_private, exist_ok=True)

    # Load existing metadata from Hub
    state = load_json_from_hub(hf_repo_id, "state.json", token=resolved_token, revision=hf_branch)
    if state is None:
        state = build_initial_state(hf_split)
        existing_chunks = 0
    else:
        state["_split"] = hf_split
        existing_chunks = len(state.get("_data_files", []))

    dataset_info = load_json_from_hub(hf_repo_id, "dataset_info.json", token=resolved_token, revision=hf_branch)
    if dataset_info is None:
        dataset_info = build_initial_dataset_info(features, hf_config)
    else:
        existing_features = dataset_info.get("features")
        if existing_features and existing_features != features_dict:
            print("\n" + "="*70)
            print("ERROR: Schema mismatch detected!")
            print("="*70)
            print("The dataset on the Hub has a different schema than what we're using.")
            print("\nExisting schema (Hub):")
            print(json.dumps(existing_features, indent=2))
            print("\nNew schema (local):")
            print(json.dumps(features_dict, indent=2))
            print("\nOptions:")
            print("1. Use a different --hf_branch (e.g., --hf_branch v2)")
            print("2. Use a different --hf_repo_id")
            print("3. Delete the existing dataset on the Hub and start fresh")
            print("="*70)
            raise ValueError("Existing dataset features on the Hub do not match the requested schema.")
        dataset_info["features"] = features_dict
        dataset_info.setdefault("builder_name", "preprocess_for_hf")
        dataset_info["config_name"] = hf_config

    stats = load_json_from_hub(hf_repo_id, "stats.json", token=resolved_token, revision=hf_branch)
    if stats is None:
        stats = {"total_episodes": 0, "total_frames": 0, "chunks": existing_chunks}
    else:
        stats.setdefault("total_episodes", 0)
        stats.setdefault("total_frames", 0)
        stats["chunks"] = max(stats.get("chunks", existing_chunks), existing_chunks)

    initial_total_episodes = stats.get("total_episodes", 0)
    initial_total_frames = stats.get("total_frames", 0)
    initial_chunk_count = existing_chunks

    # Thread-safe counter and lock for chunk indices
    chunk_lock = threading.Lock()
    upload_lock = threading.Lock()
    
    def process_and_upload_episode(ep_dir: Path):
        """Process one episode and upload it (thread-safe)."""
        nonlocal state, dataset_info, stats
        
        # Get chunk index atomically
        with chunk_lock:
            logical_chunk_idx = len(state.get("_data_files", []))
            # Reserve this slot immediately to prevent race conditions
            state["_data_files"].append({"filename": f"data/chunk_{logical_chunk_idx:05d}.arrow"})
        
        chunk_arrow_path = output_path / f"chunk_{logical_chunk_idx:05d}.arrow"
        remote_arrow_path = f"data/chunk_{logical_chunk_idx:05d}.arrow"
        
        try:
            result = process_single_episode(
                ep_dir,
                target_height,
                target_width,
                str(chunk_arrow_path),
                features,
            )
            
            if result is None:
                print(f"  [{ep_dir.name}] Skipping: no frames")
                with chunk_lock:
                    state["_data_files"].remove({"filename": remote_arrow_path})
                return
                
            episode_id, episode_length = result
            
        except Exception as e:
            print(f"  [{ep_dir.name}] Error: {e}")
            if chunk_arrow_path.exists():
                chunk_arrow_path.unlink()
            with chunk_lock:
                state["_data_files"].remove({"filename": remote_arrow_path})
            return

        chunk_bytes = chunk_arrow_path.stat().st_size

        # Upload (serialized to avoid Hub API race conditions)
        with upload_lock:
            # Update metadata
            state["_fingerprint"] = uuid.uuid4().hex
            state["_split"] = hf_split

            splits = dataset_info.setdefault("splits", {})
            split_info = splits.get(hf_split) or {"name": hf_split, "num_bytes": 0, "num_examples": 0, "shard_lengths": []}
            split_info["num_examples"] += episode_length
            split_info["num_bytes"] += chunk_bytes
            shard_lengths = split_info.get("shard_lengths") or []
            shard_lengths.append(episode_length)
            split_info["shard_lengths"] = shard_lengths
            splits[hf_split] = split_info
            dataset_info["splits"] = splits
            dataset_info["dataset_size"] = dataset_info.get("dataset_size", 0) + chunk_bytes
            dataset_info["download_size"] = dataset_info.get("download_size", 0) + chunk_bytes

            stats["total_episodes"] = stats.get("total_episodes", 0) + 1
            stats["total_frames"] = stats.get("total_frames", 0) + episode_length
            stats["chunks"] = len(state["_data_files"])

            # Upload to Hub
            operations = [
                CommitOperationAdd(path_in_repo=remote_arrow_path, path_or_fileobj=str(chunk_arrow_path)),
                CommitOperationAdd(path_in_repo="state.json", path_or_fileobj=json_to_bytes(state)),
                CommitOperationAdd(path_in_repo="dataset_info.json", path_or_fileobj=json_to_bytes(dataset_info)),
                CommitOperationAdd(path_in_repo="stats.json", path_or_fileobj=json_to_bytes(stats)),
            ]

            commit_message = f"Add chunk {logical_chunk_idx:05d}: {episode_id}, {episode_length} frames"
            
            api.create_commit(
                repo_id=hf_repo_id,
                operations=operations,
                commit_message=commit_message,
                token=resolved_token,
                repo_type="dataset",
                revision=hf_branch,
            )

            print(f"  ✓ [{episode_id}] Uploaded chunk {logical_chunk_idx:05d} ({episode_length} frames)")

        # Delete local file
        chunk_arrow_path.unlink(missing_ok=True)
        gc.collect()

    # Process episodes in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(process_and_upload_episode, episode_dirs),
            total=len(episode_dirs),
            desc="Episodes"
        ))

    # Summary
    total_uploaded_chunks = len(state["_data_files"]) - initial_chunk_count
    if total_uploaded_chunks == 0:
        print("\nNo new chunks uploaded.")
    else:
        print(f"\n✓ Uploaded {total_uploaded_chunks} new chunk(s)")

    total_episodes_hub = stats.get("total_episodes", 0)
    total_frames_hub = stats.get("total_frames", 0)
    avg_length = (total_frames_hub / total_episodes_hub) if total_episodes_hub else 0.0

    print("\n" + "="*60)
    print("Hub Dataset Summary:")
    print("="*60)
    print(f"Total chunks:   {len(state['_data_files'])}")
    print(f"Total episodes: {total_episodes_hub:,}")
    print(f"Total frames:   {total_frames_hub:,}")
    print(f"Avg length:     {avg_length:.1f} frames")
    print("="*60)
    print(f"\n✓ Dataset: https://huggingface.co/datasets/{hf_repo_id}/tree/{hf_branch}")

    return {
        "repo_id": hf_repo_id,
        "branch": hf_branch,
        "total_episodes": total_episodes_hub,
        "total_frames": total_frames_hub,
        "uploaded_chunks": total_uploaded_chunks,
        "new_episodes": total_episodes_hub - initial_total_episodes,
        "new_frames": total_frames_hub - initial_total_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Doom dataset for HuggingFace")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with episode subdirectories')
    parser.add_argument('--output_dir', type=str, default='doom_dataset_processed',
                       help='Temporary output directory (files deleted after upload)')
    parser.add_argument('--target_height', type=int, default=120,
                       help='Target image height (default: 120)')
    parser.add_argument('--target_width', type=int, default=160,
                       help='Target image width (default: 160)')
    parser.add_argument('--hf_repo_id', type=str, required=True,
                       help='HuggingFace dataset repository (e.g. username/doom-dataset)')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token (or use HF_TOKEN env var)')
    parser.add_argument('--hf_branch', type=str, default='main',
                       help='Repository branch (default: main)')
    parser.add_argument('--hf_split', type=str, default='train',
                       help='Dataset split name (default: train)')
    parser.add_argument('--hf_config', type=str, default='default',
                       help='Dataset config name (default: default)')
    parser.add_argument('--hf_private', action='store_true',
                       help='Create private repository')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_height=args.target_height,
        target_width=args.target_width,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
        hf_branch=args.hf_branch,
        hf_split=args.hf_split,
        hf_config=args.hf_config,
        hf_private=args.hf_private,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
