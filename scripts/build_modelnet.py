#!/usr/bin/env python3

#python -m scripts.build_modelnet --modelnet_root /home/kita/Desktop/master-projects/make_datasets/cross_dataset_ModelNet40_ScanObjectNN/data/ModelNet40 --out_root /home/kita/Desktop/master-projects/make_datasets/cross_dataset_ModelNet40_ScanObjectNN/outputs --target_n 1024

import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import trimesh
from scripts.utils.io import save_npy_and_meta
from scripts.utils.normalize import compute_centroid_and_scale, center_and_scale
from scripts.utils.logger import get_logger

def process_one_off(off_path, out_dir, sample_surface_n, target_n, percentile, seed=None, fps_backend='auto'):
    logger = get_logger('build_modelnet')
    try:
        mesh = trimesh.load(off_path, process=True)
        pts = mesh.sample(sample_surface_n)
        centroid, scale = compute_centroid_and_scale(pts, percentile)
        pts = center_and_scale(pts, centroid, scale)
        base = os.path.splitext(os.path.basename(off_path))[0]
        out_npy = os.path.join(out_dir, base + '.npy')
        meta = dict(orig=off_path, dataset='ModelNet', centroid=centroid.tolist(), scale=scale)
        save_npy_and_meta(out_npy, pts, meta)
        logger.debug(f"Processed {off_path} -> {out_npy}")
        return out_npy
    except Exception as e:
        logger.exception(f"Failed processing {off_path}: {e}")
        return None

def walk_and_process(modelnet_root, out_root, sample_surface_n, target_n, percentile, workers, chunk_size, seed, fps_backend='auto'):
    logger = get_logger('build_modelnet')
    off_paths = []
    for split in ['train', 'test']:
        pattern = os.path.join(modelnet_root, '*', split, '*.off')
        off_paths.extend(glob.glob(pattern))
    logger.info(f"Found {len(off_paths)} .off files")
    os.makedirs(out_root, exist_ok=True)
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as exe:
        for off in off_paths:
            rel = os.path.relpath(off, modelnet_root)
            parts = rel.split(os.sep)
            class_name = parts[0]
            split = parts[1]
            out_dir = os.path.join(out_root, 'ModelNet', class_name, split)
            os.makedirs(out_dir, exist_ok=True)
            futures.append(exe.submit(process_one_off, off, out_dir, sample_surface_n, target_n, percentile, seed, fps_backend))
        for fut in as_completed(futures):
            _ = fut.result()
    logger.info("ModelNet processing finished")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--modelnet_root', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--sample_surface_n', type=int, default=10000)
    p.add_argument('--target_n', type=int, default=1024)
    p.add_argument('--percentile', type=float, default=99.0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fps_backend', type=str, default='auto')
    args = p.parse_args()
    logger = get_logger('build_modelnet', log_file=os.path.join(args.out_root, 'build_modelnet.log'))
    walk_and_process(args.modelnet_root, args.out_root, args.sample_surface_n, args.target_n, args.percentile, args.workers, args.chunk_size, args.seed, args.fps_backend)
