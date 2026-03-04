"""
debug_masks.py — Standalone mask visualization for debugging SAM3 .pt outputs.

Produces:
  1. Static mask images at sampled frames (default every 0.5s based on source FPS)
  2. A mask-only video (white mask on black background) across all frames

All output goes to debug_masks/<pt_basename>/ so it doesn't interfere with
the existing overlay/ folder.

Usage examples:
  # Using the source video to determine FPS & resolution:
  python debug_masks.py --pt outputs/videos/bedroom_mask.pt --video inputs/videos/bedroom.mp4

  # Without a source video (defaults to 30 fps, resolution from mask tensor):
  python debug_masks.py --pt outputs/videos/bedroom_mask.pt

  # Custom sample interval (every 1 second):
  python debug_masks.py --pt outputs/videos/bedroom_mask.pt --video inputs/videos/bedroom.mp4 --interval 1.0
"""

import torch
import cv2
import argparse
import os
import sys
import numpy as np


# ─── Mask extraction (mirrors logic from visualize_masks.py) ───────────────────

def extract_mask_from_pt(output, frame_idx=0):
    """Extract a single 2-D binary mask array for *frame_idx* from the loaded .pt data."""
    current_mask = None
    try:
        if isinstance(output, list) and frame_idx < len(output):
            frame_data = output[frame_idx]
            if isinstance(frame_data, dict):
                if "masks" in frame_data:
                    current_mask = frame_data["masks"]
                elif "out_binary_masks" in frame_data:
                    current_mask = frame_data["out_binary_masks"]
            elif isinstance(frame_data, torch.Tensor):
                current_mask = frame_data

        elif isinstance(output, dict):
            if frame_idx in output:
                current_mask = output[frame_idx]
            elif str(frame_idx) in output:
                current_mask = output[str(frame_idx)]
            elif "out_binary_masks" in output:
                if frame_idx == 0:
                    current_mask = output["out_binary_masks"]
            elif "masks" in output:
                if frame_idx == 0:
                    current_mask = output["masks"]
    except Exception:
        pass

    if current_mask is not None:
        if hasattr(current_mask, "cpu"):
            current_mask = current_mask.cpu().numpy()

        if current_mask.size == 0:
            return None

        current_mask = np.squeeze(current_mask)

        # If multi-object, take the first object mask
        if current_mask.ndim > 2:
            current_mask = current_mask[0]

    return current_mask


def count_frames_in_pt(output):
    """Try to figure out how many frames the .pt data covers."""
    if isinstance(output, list):
        return len(output)
    elif isinstance(output, dict):
        # Try integer keys
        int_keys = [k for k in output.keys() if isinstance(k, int)]
        if int_keys:
            return max(int_keys) + 1
        str_keys = [k for k in output.keys() if isinstance(k, str) and k.isdigit()]
        if str_keys:
            return max(int(k) for k in str_keys) + 1
    return 1  # fallback


# ─── Rendering helpers ─────────────────────────────────────────────────────────

def mask_to_image(mask, height=None, width=None):
    """Convert a 2-D binary mask to a grayscale uint8 image (0 / 255)."""
    if mask is None:
        if height and width:
            return np.zeros((height, width), dtype=np.uint8)
        return None

    img = (mask > 0).astype(np.uint8) * 255

    if height and width and img.shape != (height, width):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    return img


def mask_to_color_image(mask, height=None, width=None):
    """
    Convert a 2-D binary mask to a coloured BGR image:
      ▸ mask region  → semi-transparent green overlay on dark bg
      ▸ background   → near-black
    """
    gray = mask_to_image(mask, height, width)
    if gray is None:
        if height and width:
            return np.zeros((height, width, 3), dtype=np.uint8)
        return None

    h, w = gray.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark grey background
    canvas[:] = (30, 30, 30)
    # Green mask region
    binary = gray > 0
    canvas[binary] = (0, 220, 80)  # BGR green

    return canvas


# ─── Main routine ──────────────────────────────────────────────────────────────

def debug_masks(pt_path, video_path=None, interval=0.5):
    # ── Load mask data ──
    print(f"Loading .pt file: {pt_path}")
    output = torch.load(pt_path, map_location="cpu", weights_only=False)

    # ── Print structure info for debugging ──
    print(f"\n{'='*60}")
    print("PT FILE STRUCTURE REPORT")
    print(f"{'='*60}")
    print(f"  Top-level type  : {type(output).__name__}")
    if isinstance(output, list):
        print(f"  Number of items : {len(output)}")
        if len(output) > 0:
            first = output[0]
            print(f"  First item type : {type(first).__name__}")
            if isinstance(first, dict):
                print(f"  First item keys : {list(first.keys())}")
                for k, v in first.items():
                    desc = f"Tensor{v.shape}" if isinstance(v, torch.Tensor) else repr(v)[:80]
                    print(f"    '{k}' → {desc}")
            elif isinstance(first, torch.Tensor):
                print(f"  First item shape: {first.shape}")
    elif isinstance(output, dict):
        print(f"  Keys            : {list(output.keys())[:20]}")
        for k, v in list(output.items())[:5]:
            desc = f"Tensor{v.shape}" if isinstance(v, torch.Tensor) else repr(v)[:80]
            print(f"    '{k}' → {desc}")
    elif isinstance(output, torch.Tensor):
        print(f"  Shape           : {output.shape}")
    print(f"{'='*60}\n")

    total_frames = count_frames_in_pt(output)
    print(f"Detected {total_frames} frame(s) in the .pt data.")

    # ── Determine FPS and resolution ──
    fps = 30.0
    width, height = None, None

    if video_path and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if vid_fps > 0 and vid_fps == vid_fps:
            fps = vid_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"Source video: {video_path}  →  {width}x{height} @ {fps} fps")
    else:
        # Try to get resolution from first non-None mask
        for fi in range(min(total_frames, 10)):
            m = extract_mask_from_pt(output, fi)
            if m is not None:
                height, width = m.shape[:2]
                break
        if height is None or width is None:
            print("ERROR: Cannot determine mask resolution. Provide --video.")
            sys.exit(1)
        print(f"No source video provided; using mask resolution {width}x{height}, fps={fps}")

    # ── Output directory ──
    pt_basename = os.path.splitext(os.path.basename(pt_path))[0]
    out_root = os.path.join("debug_masks", pt_basename)
    frames_dir = os.path.join(out_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # ── 1) Static frame images at sampled intervals ──
    frame_step = max(1, int(fps * interval))
    sampled_indices = list(range(0, total_frames, frame_step))
    print(f"\nSaving {len(sampled_indices)} static mask frames (every {interval}s = every {frame_step} frames)…")

    masks_found = 0
    for idx in sampled_indices:
        mask = extract_mask_from_pt(output, idx)
        color_img = mask_to_color_image(mask, height, width)
        if color_img is None:
            continue

        if mask is not None and np.any(mask > 0):
            masks_found += 1

        time_sec = idx / fps
        fname = f"frame_{idx:05d}_t{time_sec:.2f}s.png"
        cv2.imwrite(os.path.join(frames_dir, fname), color_img)

    print(f"  → {masks_found}/{len(sampled_indices)} sampled frames contain a non-empty mask.")
    print(f"  → Saved to: {frames_dir}/")

    # ── 2) Mask-only video ──
    video_out_path = os.path.join(out_root, f"{pt_basename}_mask_only.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    total_nonempty = 0
    print(f"\nWriting mask-only video ({total_frames} frames)…")
    for fi in range(total_frames):
        mask = extract_mask_from_pt(output, fi)
        color_img = mask_to_color_image(mask, height, width)
        if color_img is None:
            color_img = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(color_img)

        if mask is not None and np.any(mask > 0):
            total_nonempty += 1

    writer.release()
    print(f"  → {total_nonempty}/{total_frames} frames contain a non-empty mask.")
    print(f"  → Saved to: {video_out_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    if total_nonempty == 0:
        print("⚠  WARNING: NO frames had non-empty masks!")
        print("   The .pt file may contain all-zero masks, meaning the model")
        print("   did not detect anything for the given prompt. This explains")
        print("   why the overlayed video looks the same as the input.")
    else:
        pct = total_nonempty / total_frames * 100
        print(f"✓  {total_nonempty}/{total_frames} frames ({pct:.1f}%) have non-empty masks.")
    print(f"{'='*60}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug SAM3 .pt mask files — generates static frames + mask-only video"
    )
    parser.add_argument("--pt", required=True, help="Path to the .pt mask file")
    parser.add_argument("--video", default=None, help="(Optional) source video for FPS & resolution")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Interval in seconds between sampled frames (default: 0.5)")
    args = parser.parse_args()

    debug_masks(args.pt, args.video, args.interval)
