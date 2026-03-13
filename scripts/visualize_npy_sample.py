#!/usr/bin/env python3
"""
visualize_npy_sample.py

単一サンプル点群 (.npy) を読み込み、PNG として保存します。

使い方例:
python scripts/visualize_npy_sample.py --input processed/sample.npy

python scripts/visualize_npy_sample.py \
  --input processed/sample.npy \
  --out-dir outputs/vis_single \
  --name sample_view \
  --elev 18 --azim -42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 必要ならここを直接編集して使えます。
DEFAULT_INPUT = "processed/sample.npy"
DEFAULT_OUT_DIR = "outputs/vis_single"


def load_point_cloud(npy_path: Path) -> np.ndarray:
    """Load one point cloud from .npy and return shape (N, 3)."""
    arr = np.load(npy_path, allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        arr = arr.item()

    if isinstance(arr, dict):
        for key in ("xyz", "points", "pc", "pointcloud", "data"):
            if key in arr:
                arr = np.asarray(arr[key])
                break
        else:
            raise ValueError(f"Unsupported dict keys in {npy_path}: {list(arr.keys())}")
    else:
        arr = np.asarray(arr)

    if arr.ndim == 2 and arr.shape[1] >= 3:
        xyz = arr[:, :3]
    elif arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[2] >= 3:
            xyz = arr[0, :, :3]
        elif arr.shape[-1] >= 3:
            xyz = arr.reshape(-1, arr.shape[-1])[:, :3]
        else:
            raise ValueError(f"Unexpected 3D array shape: {arr.shape}")
    else:
        raise ValueError(f"Expected Nx3 or 1xNx3 style array, got: {arr.shape}")

    xyz = xyz.astype(np.float32)
    if xyz.shape[0] != 1024:
        print(f"[WARN] point count is {xyz.shape[0]} (expected: 1024)")
    return xyz


def equal_axes(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0 + 1e-6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_visualization(
    xyz: np.ndarray,
    output_png: Path,
    elev: float,
    azim: float,
    point_size: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=point_size,
        c=xyz[:, 2],
        cmap="jet",
        linewidths=0,
    )

    equal_axes(ax, xyz)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one point-cloud .npy sample and save PNG.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input .npy path")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--name", type=str, default="pointcloud_1024", help="Output png stem name")
    parser.add_argument("--elev", type=float, default=25.0, help="Camera elevation")
    parser.add_argument("--azim", type=float, default=-55.0, help="Camera azimuth")
    parser.add_argument("--point-size", type=float, default=2.0, help="Scatter point size")
    parser.add_argument("--dpi", type=int, default=220, help="Saved image DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    output_png = out_dir / f"{args.name}.png"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    xyz = load_point_cloud(input_path)
    save_visualization(
        xyz=xyz,
        output_png=output_png,
        elev=args.elev,
        azim=args.azim,
        point_size=args.point_size,
        dpi=args.dpi,
    )

    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()
