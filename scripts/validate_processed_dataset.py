#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def validate_array(path, num_points):
    errors = []
    try:
        array = np.load(path)
    except Exception as exc:
        return [f"{path}: failed to load: {exc}"]

    expected_shape = (num_points, 3)
    if array.shape != expected_shape:
        errors.append(f"{path}: shape {array.shape} != {expected_shape}")

    try:
        points = array.astype(np.float32, copy=False)
    except Exception as exc:
        errors.append(f"{path}: dtype {array.dtype} is not float-convertible: {exc}")
        return errors

    if not np.isfinite(points).all():
        errors.append(f"{path}: contains NaN or Inf")
        return errors

    centroid_norm = float(np.linalg.norm(points.mean(axis=0)))
    radii = np.linalg.norm(points, axis=1)
    max_radius = float(radii.max()) if radii.size else 0.0
    if centroid_norm > 0.1:
        errors.append(f"{path}: centroid norm too large ({centroid_norm:.6f})")
    if max_radius > 1.1:
        errors.append(f"{path}: max radius too large ({max_radius:.6f})")

    return errors


def validate_class_dir(class_dir, mode, num_points):
    errors = []
    if mode == "cross" and not class_dir.name.isdigit():
        errors.append(f"{class_dir}: cross class directory is not numeric")

    sample_count = 0
    for split in ("train", "test"):
        split_dir = class_dir / split
        if not split_dir.is_dir():
            errors.append(f"{class_dir}: missing {split} directory")
            continue

        files = sorted(split_dir.glob("*.npy"))
        if not files:
            errors.append(f"{split_dir}: no .npy files")
            continue

        sample_count += len(files)
        for path in files:
            errors.extend(validate_array(path, num_points))

    return sample_count, errors


def run(args):
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found under {root}")

    all_errors = []
    total_samples = 0
    for class_dir in class_dirs:
        sample_count, errors = validate_class_dir(class_dir, args.mode, args.num_points)
        total_samples += sample_count
        all_errors.extend(errors)

    print(f"root: {root}")
    print(f"mode: {args.mode}")
    print(f"classes: {len(class_dirs)}")
    print(f"samples: {total_samples}")
    print(f"errors: {len(all_errors)}")
    for error in all_errors[:100]:
        print(f"ERROR: {error}")
    if len(all_errors) > 100:
        print(f"ERROR: ... {len(all_errors) - 100} more")

    if all_errors:
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=["single", "cross"], required=True)
    parser.add_argument("--num_points", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
