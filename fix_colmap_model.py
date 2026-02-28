#!/usr/bin/env python3
"""Fix COLMAP multi-model issue: ensure sparse/0/ contains the largest reconstruction.

When COLMAP creates multiple sparse models, nerfstudio always uses model 0.
Sometimes the largest reconstruction ends up in model 1, 2, etc.
This script finds the model with the most registered images and moves it to 0/.

Usage:
    python fix_colmap_model.py data/scene

    # Then regenerate transforms.json
    python fix_colmap_model.py data/scene --regenerate-transforms

Run this after ns-process-data if you see "COLMAP only found poses for X% of the images".
"""

import argparse
import struct
import shutil
from pathlib import Path


def count_images_in_model(model_dir):
    """Count registered images in a COLMAP binary model."""
    images_bin = model_dir / "images.bin"
    if not images_bin.exists():
        return 0
    with open(images_bin, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def fix_colmap_model(scene_dir, regenerate=False):
    sparse_dir = Path(scene_dir) / "colmap" / "sparse"

    if not sparse_dir.exists():
        print(f"No sparse directory found at {sparse_dir}")
        return False

    # Find all model directories
    model_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                        key=lambda d: int(d.name))

    if not model_dirs:
        print("No COLMAP models found")
        return False

    # Count images in each model
    print(f"Found {len(model_dirs)} COLMAP model(s):")
    models = []
    for d in model_dirs:
        n = count_images_in_model(d)
        print(f"  Model {d.name}: {n} registered images")
        models.append((d, n))

    # Find the largest model
    best_dir, best_count = max(models, key=lambda x: x[1])

    if best_dir.name == "0":
        print(f"\nModel 0 is already the largest ({best_count} images). No fix needed.")
    else:
        print(f"\nLargest model is {best_dir.name} ({best_count} images). Swapping with model 0...")
        model_0 = sparse_dir / "0"
        tmp = sparse_dir / "_tmp_swap"
        model_0.rename(tmp)
        best_dir.rename(model_0)
        tmp.rename(best_dir)
        print(f"  Swapped model 0 ↔ model {best_dir.name}")

    if regenerate:
        print("\nRegenerating transforms.json...")
        from nerfstudio.process_data.colmap_utils import colmap_to_json
        n = colmap_to_json(
            recon_dir=sparse_dir / "0",
            output_dir=Path(scene_dir),
        )
        print(f"  Generated transforms.json with {n} matched images")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix COLMAP multi-model issue")
    parser.add_argument("scene_dir", help="Scene directory (e.g. data/scene)")
    parser.add_argument("--regenerate-transforms", action="store_true",
                        help="Regenerate transforms.json after fixing")
    args = parser.parse_args()
    fix_colmap_model(args.scene_dir, args.regenerate_transforms)
