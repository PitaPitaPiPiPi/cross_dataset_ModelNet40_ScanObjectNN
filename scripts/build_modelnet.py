#!/usr/bin/env python3
# python -m scripts.build_modelnet --modelnet_root raw_datasets/modelnet40 --out_root outputs

import os
import glob
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import trimesh
from scripts.utils.io import save_npy_and_meta
from scripts.utils.normalize import pc_normalize_unified
from scripts.utils.logger import get_logger

def process_one_off(off_path, out_dir, sample_surface_n, fps_k):
    logger = get_logger('build_modelnet')
    try:
        mesh = trimesh.load(off_path, process=True)
        pts = mesh.sample(sample_surface_n)
        pts, centroid, scale = pc_normalize_unified(
            pts, 
            openshape=True, 
            return_meta=True,
            use_fps=True,
            fps_k=fps_k,
            seed=42
            )
        base = os.path.splitext(os.path.basename(off_path))[0]
        out_npy = os.path.join(out_dir, base + '.npy')
        meta = dict(orig=off_path, dataset='ModelNet', centroid=centroid.tolist(), scale=scale)
        save_npy_and_meta(out_npy, pts, meta)
        logger.debug(f"Saved {out_npy}")
        return out_npy
    except Exception as e:
        logger.exception(f"Failed: {off_path}: {e}")
        return None

def load_point_cloud(npy_path):
    pts = np.load(npy_path)
    if pts.ndim != 2:
        raise ValueError(f"Point cloud must be 2D, got shape {pts.shape}")
    if pts.shape[1] != 3 and pts.shape[0] == 3:
        pts = pts.T
    if pts.shape[1] != 3:
        raise ValueError(f"Point cloud must have 3 columns, got shape {pts.shape}")
    return pts.astype(np.float32)

def process_one_npy(npy_path, out_dir, fps_k):
    logger = get_logger('build_modelnet')
    try:
        pts = load_point_cloud(npy_path)
        pts, centroid, scale = pc_normalize_unified(
            pts,
            openshape=True,
            return_meta=True,
            use_fps=True,
            fps_k=fps_k,
            seed=42
            )
        base = os.path.splitext(os.path.basename(npy_path))[0]
        out_npy = os.path.join(out_dir, base + '.npy')
        meta = dict(orig=npy_path, dataset='ModelNet', centroid=centroid.tolist(), scale=scale)
        save_npy_and_meta(out_npy, pts, meta)
        logger.debug(f"Saved {out_npy}")
        return out_npy
    except Exception as e:
        logger.exception(f"Failed: {npy_path}: {e}")
        return None

def walk_and_process(modelnet_root, out_root, sample_surface_n, workers, chunk_size, fps_k, sample_from_mesh):
    logger = get_logger('build_modelnet')
    input_paths = []
    ext = 'off' if sample_from_mesh else 'npy'
    for split in ['train', 'test']:
        pattern = os.path.join(modelnet_root, '*', split, f'*.{ext}')
        input_paths.extend(glob.glob(pattern))
    logger.info(f"Input files: {len(input_paths)} .{ext}")
    os.makedirs(out_root, exist_ok=True)
    if not input_paths:
        logger.warning("No input files found. Check --modelnet_root and input format.")
        return
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as exe:
        for src in input_paths:
            rel = os.path.relpath(src, modelnet_root)
            parts = rel.split(os.sep)
            class_name = parts[0]
            split = parts[1]
            out_dir = os.path.join(out_root, 'ModelNet', class_name, split)
            os.makedirs(out_dir, exist_ok=True)
            if sample_from_mesh:
                futures.append(exe.submit(process_one_off, src, out_dir, sample_surface_n, fps_k))
            else:
                futures.append(exe.submit(process_one_npy, src, out_dir, fps_k))
        for fut in as_completed(futures):
            _ = fut.result()
    train_count = len(glob.glob(os.path.join(out_root, 'ModelNet', '*', 'train', '*.npy')))
    test_count = len(glob.glob(os.path.join(out_root, 'ModelNet', '*', 'test', '*.npy')))
    train_files = sorted(glob.glob(os.path.join(out_root, 'ModelNet', '*', 'train', '*.npy')))
    test_files = sorted(glob.glob(os.path.join(out_root, 'ModelNet', '*', 'test', '*.npy')))
    class_dirs = sorted(glob.glob(os.path.join(out_root, 'ModelNet', '*')))
    logger.info(f"ModelNet train: {train_count}")
    logger.info(f"ModelNet test: {test_count}")
    if class_dirs:
        logger.info("ModelNet per-class counts:")
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            class_train = len(glob.glob(os.path.join(class_dir, 'train', '*.npy')))
            class_test = len(glob.glob(os.path.join(class_dir, 'test', '*.npy')))
            logger.info(f"  {class_name}: train={class_train} test={class_test}")
    else:
        logger.warning("No ModelNet class directories found.")
    if train_files:
        train_shape = np.load(train_files[0]).shape
        logger.info(f"ModelNet first train shape: {train_shape}")
    else:
        logger.warning("No ModelNet train sample found for shape check.")
    if test_files:
        test_shape = np.load(test_files[0]).shape
        logger.info(f"ModelNet first test shape: {test_shape}")
    else:
        logger.warning("No ModelNet test sample found for shape check.")
    logger.info("ModelNet done")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--modelnet_root', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--sample_surface_n', type=int, default=10000)
    p.add_argument('--fps_k', type=int, default=1024, help='Number of points after FPS sampling')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk_size', type=int, default=64)
    p.add_argument('--sample_from_mesh', action='store_true', help='Sample points from .off meshes in modelnet_root')
    p.add_argument('--no_sample_from_mesh', dest='sample_from_mesh', action='store_false', help='Use existing .npy point clouds in modelnet_root')
    p.set_defaults(sample_from_mesh=True)
    args = p.parse_args()
    logger = get_logger('build_modelnet', log_file=os.path.join(args.out_root, 'build_modelnet.log'))
    walk_and_process(
        args.modelnet_root,
        args.out_root,
        args.sample_surface_n,
        args.workers,
        args.chunk_size,
        args.fps_k,
        args.sample_from_mesh,
    )
