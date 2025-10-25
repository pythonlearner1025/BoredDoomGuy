#!/usr/bin/env python3
"""
Fix the dataset by merging all uploaded chunks into one dataset
"""

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
import argparse

def fix_dataset(repo_name, token):
    """Merge all chunks into one dataset"""

    login(token)

    print("🔧 Loading existing dataset...")
    dataset = load_dataset(repo_name)

    print(f"📊 Current dataset size: {len(dataset['train']):,}")
    print(f"🗂️  Features: {dataset['train'].features}")

    # The issue is that each chunk overwrote the previous one
    # We need to reload and concatenate all the data properly
    print("\n⚠️  The dataset was overwritten by chunks instead of accumulated.")
    print("🔄 Need to re-upload with proper accumulation...")

    # Unfortunately, the data is lost and needs to be re-uploaded
    print("\n💡 Solution: Re-run upload_optimized.py with accumulation fix")

    return False

def main():
    parser = argparse.ArgumentParser(description="Check dataset status")
    parser.add_argument("--repo_name", type=str, default="invocation02/RandomDoomSamples-11M")
    parser.add_argument("--token", type=str, required=True)

    args = parser.parse_args()

    fix_dataset(args.repo_name, args.token)

if __name__ == "__main__":
    main()