#!/usr/bin/env python3
"""Remove outlier gaussians from a 3DGS PLY file.

Filters by distance from the scene center (median position) and optionally
by opacity and scale thresholds. Produces a cleaned PLY ready for SPZ conversion.

Usage:
    python cleanup_ply.py export/splat.ply export/splat_clean.ply
    python cleanup_ply.py export/splat.ply export/splat_clean.ply --distance 5.0
    python cleanup_ply.py export/splat.ply export/splat_clean.ply --distance 3.0 --min-opacity -2.0
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement

def cleanup(input_path, output_path, max_distance=5.0, min_opacity=None, max_scale=None):
    print(f"Loading {input_path}...")
    ply = PlyData.read(input_path)
    vertex = ply["vertex"]
    n_orig = len(vertex.data)

    positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])

    # Use median as robust center estimate (not affected by outliers)
    center = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)

    print(f"  Gaussians: {n_orig}")
    print(f"  Center (median): [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Distance: min={distances.min():.3f} max={distances.max():.3f} "
          f"mean={distances.mean():.3f} p95={np.percentile(distances, 95):.3f}")

    # Distance filter
    mask = distances <= max_distance
    print(f"  Distance filter (<= {max_distance}): keeping {mask.sum()}/{n_orig}")

    # Opacity filter (values are in logit space: sigmoid(-2) ≈ 0.12)
    if min_opacity is not None:
        opacity = np.array(vertex["opacity"])
        opacity_mask = opacity >= min_opacity
        mask &= opacity_mask
        print(f"  Opacity filter (>= {min_opacity}): keeping {mask.sum()}/{n_orig}")

    # Scale filter (values are in log space: exp(max_scale))
    if max_scale is not None:
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scales = np.column_stack([np.array(vertex[s]) for s in scale_names])
        scale_mask = scales.max(axis=1) <= max_scale
        mask &= scale_mask
        print(f"  Scale filter (<= {max_scale}): keeping {mask.sum()}/{n_orig}")

    filtered = vertex.data[mask]
    n_kept = len(filtered)
    pct = (1 - n_kept / n_orig) * 100

    new_vertex = PlyElement.describe(filtered, "vertex")
    PlyData([new_vertex]).write(output_path)

    print(f"\nDone: {output_path}")
    print(f"  Removed {n_orig - n_kept} gaussians ({pct:.1f}%): {n_orig} → {n_kept}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean outlier gaussians from a 3DGS PLY file")
    parser.add_argument("input", help="Input PLY file")
    parser.add_argument("output", help="Output PLY file")
    parser.add_argument("--distance", type=float, default=5.0,
                        help="Max distance from median center (default: 5.0)")
    parser.add_argument("--min-opacity", type=float, default=None,
                        help="Min opacity in logit space (e.g. -2.0 ≈ 12%% visible)")
    parser.add_argument("--max-scale", type=float, default=None,
                        help="Max scale in log space (e.g. 3.0 = exp(3) ≈ 20x)")
    args = parser.parse_args()
    cleanup(args.input, args.output, args.distance, args.min_opacity, args.max_scale)
