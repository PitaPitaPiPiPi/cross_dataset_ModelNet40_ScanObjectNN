#!/usr/bin/env python3
"""
save_scanvis.py

Usage example:
python3 save_scanvis.py \
  --uni-file "/path/to/xyz_label.npy" \
  --my-root "~/scanobjectnn" \
  --out-uni "~/scanobjectnn_uni_vis" \
  --out-my "~/scanobjectnn_my_vis" \
  --max-points 5000 \
  --elev 30 --azim -60 \
  --try-transforms

This script:
- Loads uni single .npy (dictionary-like with key 'xyz' or array-like of samples)
- Traverses a per-sample dataset root (class subdirs) looking for .npy files
- Renders each sample to a 3D scatter and saves PNG images in mirror folder layout.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display required
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from tqdm import tqdm
import itertools

SCANOBJECTNN_CLASS_NAMES = [
    'bag',
    'bed',
    'bin',
    'box',
    'cabinet',
    'chair',
    'desk',
    'display',
    'door',
    'pillow',
    'shelf',
    'sink',
    'sofa',
    'table',
    'toilet',
]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_sample_from_npy(path):
    """Load a .npy file that may be:
       - an array Nx3
       - a dict-like saved with allow_pickle (e.g. {'xyz': [...]})
       - an object array containing array
    """
    data = np.load(path, allow_pickle=True)
    # If it's an array with dtype=object and length==1 containing dict, unwrap
    try:
        if isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
            data = data.item()
    except Exception:
        pass
    # If it's a dict-like with 'xyz'
    if isinstance(data, dict):
        if 'xyz' in data:
            xyz = np.asarray(data['xyz'])
        else:
            # try common keys
            for k in ('points', 'pc', 'pointcloud', 'data'):
                if k in data:
                    xyz = np.asarray(data[k])
                    break
            else:
                raise ValueError(f"Unrecognized dict keys in {path}: {list(data.keys())}")
    else:
        xyz = np.asarray(data)
    # final check
    if xyz.ndim == 2 and xyz.shape[1] >= 3:
        xyz = xyz[:, :3]
    elif xyz.ndim == 3:
        # maybe shape (1,N,3) or (num_samples,N,3)
        xyz = xyz.reshape(-1, 3)
    else:
        raise ValueError(f"Loaded array from {path} has unexpected shape {xyz.shape}")
    return xyz.astype(np.float32)

def load_uni_samples(uni_path):
    """Load uni-style .npy (dictionary or array of samples).
       Return samples list and optional labels array.
    """
    data = np.load(uni_path, allow_pickle=True)
    try:
        if isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
            data = data.item()
    except Exception:
        pass
    labels = None
    if isinstance(data, dict):
        if 'xyz' in data:
            xyz_all = data['xyz']
            # xyz_all might be list-like or ndarray
            if isinstance(xyz_all, np.ndarray):
                if xyz_all.ndim == 2:
                    samples = [xyz_all.astype(np.float32)]
                elif xyz_all.ndim == 3:
                    samples = [xyz_all[i].astype(np.float32) for i in range(xyz_all.shape[0])]
                else:
                    raise ValueError(f"Unexpected xyz shape in {uni_path}: {xyz_all.shape}")
            else:
                samples = [np.asarray(x).astype(np.float32) for x in xyz_all]
            if 'label' in data:
                labels = np.asarray(data['label']).reshape(-1)
        else:
            raise ValueError(f"uni npy dict has no 'xyz' key: {list(data.keys())}")
    elif isinstance(data, np.ndarray):
        # if shape (num_samples, N, 3)
        if data.ndim == 3:
            samples = [data[i].astype(np.float32) for i in range(data.shape[0])]
        else:
            # fallback: treat as single sample
            samples = [np.asarray(data).astype(np.float32)]
    else:
        raise ValueError(f"Unsupported uni data type: {type(data)}")
    return samples, labels

def equalize_axes(ax, xyz):
    """Set equal aspect (approx) for 3D scatter by setting limits to same ranges."""
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    min_vals = np.min(x), np.min(y), np.min(z)
    max_vals = np.max(x), np.max(y), np.max(z)
    mins = np.array(min_vals)
    maxs = np.array(max_vals)
    centers = 0.5 * (mins + maxs)
    max_range = np.max(maxs - mins) * 0.5 + 1e-6
    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

def plot_and_save(xyz, out_path, elev=30, azim=-60, max_points=5000, dpi=150, point_size=1, cmap='jet'):
    # downsample for plotting
    N = xyz.shape[0]
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        xyz_disp = xyz[idx]
    else:
        xyz_disp = xyz
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xyz_disp[:,0], xyz_disp[:,1], xyz_disp[:,2], s=point_size, c=xyz_disp[:,2], cmap=cmap, linewidths=0)
    equalize_axes(ax, xyz)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # remove margins
    plt.tight_layout(pad=0)
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def try_transformed_variants(xyz, out_prefix, elev, azim, max_points, dpi, point_size):
    """Generate images for permutations and sign flips to help find coordinate transform that matches.
       We'll produce up to 12 variants by default (6 permutations). Optionally could do sign flips (8x).
    """
    axes = [0,1,2]
    permuts = list(itertools.permutations(axes))
    # only permutations (6). If you want flips too, uncomment flips loop (heavy).
    for p in permuts:
        xyz_p = xyz[:, list(p)]
        fname = f"{out_prefix}_perm{''.join(map(str,p))}.png"
        plot_and_save(xyz_p, fname, elev=elev, azim=azim, max_points=max_points, dpi=dpi, point_size=point_size)
    # --- If you want sign flips as well, you can expand here (commented-out by default)
    # signs = list(itertools.product([1,-1],[1,-1],[1,-1]))
    # for p in permuts:
    #     for s in signs:
    #         xyz_ps = xyz[:, list(p)] * np.array(s)
    #         fname = f"{out_prefix}_perm{''.join(map(str,p))}_sign{''.join(['+' if si>0 else '-' for si in s])}.png"
    #         plot_and_save(xyz_ps, fname, elev=elev, azim=azim, max_points=max_points, dpi=dpi, point_size=point_size)

def process_my_dataset(root_dir, out_root, elev, azim, max_points, dpi, point_size, try_transforms_flag):
    root_dir = Path(root_dir).expanduser()
    out_root = Path(out_root).expanduser()
    # Expect structure: root/class_name/{train,test}/files.npy  OR root/class_name/*.npy
    class_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    for cls in class_dirs:
        # find .npy files recursively under class dir
        npy_paths = list(cls.rglob('*.npy'))
        if not npy_paths:
            continue
        # determine relative class path (just class name)
        rel_cls = cls.name
        for p in tqdm(npy_paths, desc=f"Processing {rel_cls}", unit="file"):
            try:
                xyz = load_sample_from_npy(str(p))
            except Exception as e:
                print(f"Failed to load {p}: {e}")
                continue
            # determine output path: keep subpath under class (mirror 'train' or 'test' if present)
            # compute subpath relative to class dir
            rel_sub = p.relative_to(cls.parent) if cls.parent in p.parents else p.relative_to(cls)
            # prefer: out_root/classname/<subfolders without .npy>/
            out_dir = out_root / rel_sub.parent
            ensure_dir(out_dir)
            out_fname = p.stem + ".png"
            out_path = out_dir / out_fname
            plot_and_save(xyz, str(out_path), elev=elev, azim=azim, max_points=max_points, dpi=dpi, point_size=point_size)
            if try_transforms_flag:
                out_prefix = str(out_path.with_suffix(''))
                try_transformed_variants(xyz, out_prefix + "_transform", elev, azim, max_points, dpi, point_size)

def process_uni_file(uni_path, out_root, elev, azim, max_points, dpi, point_size, try_transforms_flag):
    out_root = Path(out_root).expanduser()
    samples, labels = load_uni_samples(uni_path)
    if labels is None:
        raise ValueError("uni npy has no 'label' key; cannot route samples to class folders")
    if len(labels) < len(samples):
        raise ValueError(f"Label count {len(labels)} is smaller than samples {len(samples)}")
    for idx, xyz in enumerate(tqdm(samples, desc="Processing uni samples", unit="sample")):
        label = int(labels[idx])
        if label < 0 or label >= len(SCANOBJECTNN_CLASS_NAMES):
            raise ValueError(f"Invalid label {label} at sample {idx}")
        class_name = SCANOBJECTNN_CLASS_NAMES[label]
        out_dir = out_root / "classes" / class_name / "test"
        ensure_dir(out_dir)
        out_path = out_dir / f"{idx:06d}.png"
        plot_and_save(xyz, str(out_path), elev=elev, azim=azim, max_points=max_points, dpi=dpi, point_size=point_size)
        if try_transforms_flag:
            out_prefix = str(out_path.with_suffix(''))
            try_transformed_variants(xyz, out_prefix + "_transform", elev, azim, max_points, dpi, point_size)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uni-file", type=str, default=None, help="Path to Uni3D .npy (dictionary-like with 'xyz').")
    p.add_argument("--my-root", type=str, default=None, help="Root of your per-sample dataset (class subfolders).")
    p.add_argument("--out-uni", type=str, default="~/scanobjectnn_uni_vis", help="Output root for Uni visualizations.")
    p.add_argument("--out-my", type=str, default="~/scanobjectnn_my_vis", help="Output root for your dataset visualizations.")
    p.add_argument("--max-points", type=int, default=5000)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--point-size", type=float, default=1.0)
    p.add_argument("--elev", type=float, default=30)
    p.add_argument("--azim", type=float, default=-60)
    p.add_argument("--try-transforms", action='store_true', help="Also generate permutation variants to help find coordinate mapping.")
    args = p.parse_args()

    if args.uni_file is None and args.my_root is None:
        p.error("At least one of --uni-file or --my-root must be provided.")

    if args.uni_file:
        print("Processing Uni file:", args.uni_file)
        process_uni_file(args.uni_file, args.out_uni, args.elev, args.azim, args.max_points, args.dpi, args.point_size, args.try_transforms)
        print("Uni processing done ->", args.out_uni)

    if args.my_root:
        print("Processing my dataset root:", args.my_root)
        process_my_dataset(args.my_root, args.out_my, args.elev, args.azim, args.max_points, args.dpi, args.point_size, args.try_transforms)
        print("My dataset processing done ->", args.out_my)

if __name__ == "__main__":
    main()
