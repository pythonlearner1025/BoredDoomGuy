#!/usr/bin/env python3

"""
CLI tool to annotate DOOM gameplay frames using VLMs via OpenRouter.
Processes frames with history context to generate terse, imperative commands.
"""

import os
import json
import base64
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

cwd = os.path.dirname(os.path.abspath(__file__))

def find_latest_screen_dir(base_dir: str = "screens") -> Optional[Path]:
    """Find the latest timestamped directory in screens/."""
    screens_path = Path(os.path.join(cwd, base_dir))
    print(cwd)
    print(screens_path)
    if not screens_path.exists():
        return None

    # Find all subdirectories that match timestamp pattern (YYYY-MM-DD-HHMM)
    timestamp_dirs = [d for d in screens_path.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        return None

    # Sort by name (timestamp format sorts chronologically)
    latest_dir = sorted(timestamp_dirs, key=lambda d: d.name)[-1]
    return latest_dir


def get_frame_files(frame_dir: Path, frame_idx: int) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Get paths for raw frame, annotated frame, and labels JSON for a given frame index."""
    raw_frame = frame_dir / f"frame_{frame_idx:06d}.png"
    annotated_frame = frame_dir / f"frame_{frame_idx:06d}_annotated.png"
    labels_json = frame_dir / f"frame_{frame_idx:06d}_labels.json"

    return (
        raw_frame if raw_frame.exists() else None,
        annotated_frame if annotated_frame.exists() else None,
        labels_json if labels_json.exists() else None
    )


def load_json_labels(json_path: Path) -> Dict:
    """Load JSON labels from file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def build_prompt(
    current_frame_idx: int,
    frame_dir: Path,
    history_length: int,
    prompt_template: str,
    add_annotation: bool = True
) -> Tuple[str, List[Dict]]:
    """
    Build the prompt with history context.
    Returns (text_prompt, list_of_image_content_dicts).
    """
    # Get current frame
    _, current_annotated, current_labels = get_frame_files(frame_dir, current_frame_idx)

    if not current_annotated or not current_labels:
        raise FileNotFoundError(f"Current frame {current_frame_idx} missing files")

    current_labels_data = load_json_labels(current_labels)

    # Build history
    history_parts = []
    image_contents = []

    for i in range(history_length, 0, -1):
        hist_idx = current_frame_idx - i
        if hist_idx < 0:
            continue  # Skip if before start of recording

        _, hist_annotated, hist_labels = get_frame_files(frame_dir, hist_idx)

        if hist_annotated and hist_labels:
            hist_labels_data = load_json_labels(hist_labels)

            # Add to history text
            history_parts.append(f"frame t-{i} entities json: {json.dumps(hist_labels_data)}")

            # Add annotated image
            hist_img_b64 = encode_image_base64(hist_annotated)
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{hist_img_b64}"
                }
            })
            history_parts.append(f"frame t-{i}: [see image above]")

            # Add previous annotation if available
            if add_annotation:
                hist_annotation_path = frame_dir / f"frame_{hist_idx:06d}_annotation.txt"
                if hist_annotation_path.exists():
                    with open(hist_annotation_path, 'r') as f:
                        hist_annotation = f.read().strip()
                    history_parts.append(f"frame t-{i} generated annotation: {hist_annotation}")

    # Add current frame
    history_parts.append(f"frame t=0 entities json: {json.dumps(current_labels_data)}")
    current_img_b64 = encode_image_base64(current_annotated)
    image_contents.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{current_img_b64}"
        }
    })
    history_parts.append("frame t=0: [see image above]")

    # Format the prompt
    history_text = "\n".join(history_parts)

    # Replace placeholders
    prompt_text = prompt_template.replace(
        "{ground_truth_labels_json}",
        json.dumps(current_labels_data, indent=2)
    )
    prompt_text = prompt_text.replace(
        "{history_length}",
        str(history_length)
    )

    # Replace history section
    prompt_text = prompt_text.replace(
        "<history>\nframe {t} entities json: {frame_t_entities}\nframe {t}: {frame_t}\n</history>",
        f"<history>\n{history_text}\n</history>"
    )

    return prompt_text, image_contents


def annotate_frame(
    client: OpenAI,
    frame_idx: int,
    frame_dir: Path,
    history_length: int,
    prompt_template: str,
    model: str = "google/gemini-2.0-flash-exp:free",
    add_annotation: bool = True
) -> str:
    """
    Annotate a single frame using OpenRouter API.
    Returns the generated annotation text.
    """
    # Build prompt with history
    prompt_text, image_contents = build_prompt(
        frame_idx,
        frame_dir,
        history_length,
        prompt_template,
        add_annotation
    )

    # Build messages with interleaved text and images
    user_content = [{"type": "text", "text": prompt_text}]
    user_content.extend(image_contents)

    # Call OpenRouter
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://github.com/Farama-Foundation/ViZDoom",
            "X-Title": "DOOM Frame Annotation Tool",
        },
        model=model,
        messages=[
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.7,
        max_tokens=50  # Keep responses short
    )

    # Handle None response
    if not completion.choices or not completion.choices[0].message.content:
        return "none"  # Default response when API returns empty

    return completion.choices[0].message.content.strip()


def save_annotation(frame_dir: Path, frame_idx: int, annotation: str):
    """Save annotation to text file."""
    annotation_path = frame_dir / f"frame_{frame_idx:06d}_annotation.txt"
    with open(annotation_path, 'w') as f:
        f.write(annotation)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate DOOM gameplay frames using VLMs via OpenRouter"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to frame directory. If not specified, uses latest timestamped dir in ./screens"
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=2,
        help="Number of previous frames to include in context (default: 2)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it:free",
        help="OpenRouter model to use (default: google/gemma-2-27b-it:free)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index (default: 0)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Ending frame index (default: process all frames)"
    )
    parser.add_argument(
        "--add-annotation",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include previous frame annotations in history (default: True)"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Process every Nth frame (e.g., --skip 5 processes frames 0, 5, 10, ...) (default: 1)"
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        return 1

    # Determine frame directory
    if args.path:
        frame_dir = Path(args.path)
    else:
        frame_dir = find_latest_screen_dir()
        if not frame_dir:
            print("ERROR: No timestamped directories found in ./screens")
            return 1

    if not frame_dir.exists():
        print(f"ERROR: Frame directory does not exist: {frame_dir}")
        return 1

    print(f"Processing frames in: {frame_dir}")

    # Load prompt template
    prompt_path = Path("prompt.txt")
    if not prompt_path.exists():
        print("ERROR: prompt.txt not found")
        return 1

    with open(prompt_path, 'r') as f:
        prompt_template = f.read()

    # Initialize OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Find all frames
    frame_files = sorted(frame_dir.glob("frame_*_annotated.png"))
    frame_indices = [
        int(f.stem.split('_')[1])
        for f in frame_files
    ]

    if not frame_indices:
        print(f"ERROR: No annotated frames found in {frame_dir}")
        return 1

    # Apply frame range
    start_idx = args.start_frame
    end_idx = args.end_frame if args.end_frame is not None else max(frame_indices)

    frame_indices = [idx for idx in frame_indices if start_idx <= idx <= end_idx]

    # Apply skip filter
    if args.skip > 1:
        frame_indices = [idx for idx in frame_indices if idx % args.skip == 0]

    print(f"Found {len(frame_indices)} frames to process")
    print(f"Using model: {args.model}")
    print(f"History length: {args.history_length}")
    if args.skip > 1:
        print(f"Skip: Every {args.skip} frames")
    print()

    # Process each frame
    for i, frame_idx in enumerate(frame_indices):
        # Check if already annotated
        annotation_path = frame_dir / f"frame_{frame_idx:06d}_annotation.txt"
        if annotation_path.exists():
            print(f"[{i+1}/{len(frame_indices)}] Frame {frame_idx:06d}: Already annotated, skipping")
            continue

        print(f"[{i+1}/{len(frame_indices)}] Frame {frame_idx:06d}: Generating annotation...", end=" ", flush=True)

        try:
            annotation = annotate_frame(
                client,
                frame_idx,
                frame_dir,
                args.history_length,
                prompt_template,
                args.model,
                args.add_annotation
            )

            save_annotation(frame_dir, frame_idx, annotation)
            print(f" '{annotation}'")

        except Exception as e:
            print(f" ERROR: {e}")
            continue

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
