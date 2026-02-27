#!/usr/bin/env python3
"""Clean and transform a 3DGS PLY file.

Filters outlier gaussians and applies rotation/centering transforms.
Produces a cleaned PLY ready for SPZ conversion and web upload.

Usage:
    # Basic cleanup
    python cleanup_ply.py export/splat.ply export/splat_clean.ply

    # Rotate 90° around X axis (fix "lying down" orientation from phone video)
    python cleanup_ply.py export/splat.ply export/splat_clean.ply --rotate-x 90

    # Rotate around multiple axes
    python cleanup_ply.py export/splat.ply export/splat_clean.ply --rotate-x 90 --rotate-y 45

    # Full cleanup + rotation
    python cleanup_ply.py export/splat.ply export/splat_clean.ply --distance 3.0 --min-opacity 0.0 --rotate-x -90
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def rotation_matrix(axis, degrees):
    """Create a 3x3 rotation matrix for rotation around X, Y, or Z axis."""
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]. Returns q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def cleanup(input_path, output_path, max_distance=5.0, min_opacity=None,
            max_scale=None, rotate_x=0, rotate_y=0, rotate_z=0, recenter=False):
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

    filtered = vertex.data[mask].copy()
    n_kept = len(filtered)
    pct = (1 - n_kept / n_orig) * 100
    print(f"  Kept {n_kept}/{n_orig} gaussians (removed {pct:.1f}%)")

    # Apply rotation if any axis is non-zero
    has_rotation = rotate_x != 0 or rotate_y != 0 or rotate_z != 0
    if has_rotation or recenter:
        positions = np.column_stack([filtered["x"], filtered["y"], filtered["z"]])

        # Recenter to median before rotation (rotate around scene center)
        center = np.median(positions, axis=0)
        positions -= center
        print(f"  Centering to origin (shifted by [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}])")

    if has_rotation:
        # Build combined rotation matrix (applied in order: X, then Y, then Z)
        R = np.eye(3)
        if rotate_x != 0:
            R = rotation_matrix("x", rotate_x) @ R
            print(f"  Rotating {rotate_x}° around X axis")
        if rotate_y != 0:
            R = rotation_matrix("y", rotate_y) @ R
            print(f"  Rotating {rotate_y}° around Y axis")
        if rotate_z != 0:
            R = rotation_matrix("z", rotate_z) @ R
            print(f"  Rotating {rotate_z}° around Z axis")

        # Rotate positions
        positions = (R @ positions.T).T

        # Rotate gaussian quaternions (rot_0=w, rot_1=x, rot_2=y, rot_3=z)
        q_rot = rotation_matrix_to_quaternion(R)
        quats = np.column_stack([
            filtered["rot_0"], filtered["rot_1"],
            filtered["rot_2"], filtered["rot_3"]
        ])
        quats = quaternion_multiply(q_rot, quats)

        # Write back
        filtered["rot_0"] = quats[:, 0]
        filtered["rot_1"] = quats[:, 1]
        filtered["rot_2"] = quats[:, 2]
        filtered["rot_3"] = quats[:, 3]

    if has_rotation or recenter:
        filtered["x"] = positions[:, 0]
        filtered["y"] = positions[:, 1]
        filtered["z"] = positions[:, 2]

    new_vertex = PlyElement.describe(filtered, "vertex")
    PlyData([new_vertex]).write(output_path)

    print(f"\nDone: {output_path}")
    print(f"  {n_orig} → {n_kept} gaussians")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and transform a 3DGS PLY file")
    parser.add_argument("input", help="Input PLY file")
    parser.add_argument("output", help="Output PLY file")
    parser.add_argument("--distance", type=float, default=5.0,
                        help="Max distance from median center (default: 5.0)")
    parser.add_argument("--min-opacity", type=float, default=None,
                        help="Min opacity in logit space (e.g. -2.0 ≈ 12%% visible)")
    parser.add_argument("--max-scale", type=float, default=None,
                        help="Max scale in log space (e.g. 3.0 = exp(3) ≈ 20x)")
    parser.add_argument("--rotate-x", type=float, default=0,
                        help="Rotation around X axis in degrees (e.g. 90, -90)")
    parser.add_argument("--rotate-y", type=float, default=0,
                        help="Rotation around Y axis in degrees")
    parser.add_argument("--rotate-z", type=float, default=0,
                        help="Rotation around Z axis in degrees")
    parser.add_argument("--recenter", action="store_true",
                        help="Move scene center to origin (applied even without rotation)")
    args = parser.parse_args()
    cleanup(args.input, args.output, args.distance, args.min_opacity,
            args.max_scale, args.rotate_x, args.rotate_y, args.rotate_z, args.recenter)
