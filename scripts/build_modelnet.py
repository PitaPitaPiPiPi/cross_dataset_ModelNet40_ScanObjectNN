#!/usr/bin/env python3

#python -m scripts.build_modelnet --modelnet_root /home/kita/Desktop/master-projects/make_datasets/cross_dataset_ModelNet40_ScanObjectNN/data/ModelNet40 --out_root /home/kita/Desktop/master-projects/make_datasets/cross_dataset_ModelNet40_ScanObjectNN/outputs

import os
import glob
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import trimesh
from scripts.utils.io import save_npy_and_meta
from scripts.utils.normalize import pc_normalize_unified
from scripts.utils.logger import get_logger

def process_one_off(off_path, out_dir, sample_surface_n):
    logger = get_logger('build_modelnet')
    try:
        mesh = trimesh.load(off_path, process=True)
        pts = mesh.sample(sample_surface_n)
        centroid = pts.mean(axis=0)
        pts, centroid, scale = pc_normalize_unified(pts, openshape=True, return_meta=True)
        base = os.path.splitext(os.path.basename(off_path))[0]
        out_npy = os.path.join(out_dir, base + '.npy')
        meta = dict(orig=off_path, dataset='ModelNet', centroid=centroid.tolist(), scale=scale)
        save_npy_and_meta(out_npy, pts, meta)
        logger.debug(f"Processed {off_path} -> {out_npy}")
        return out_npy
    except Exception as e:
        logger.exception(f"Failed processing {off_path}: {e}")
        return None

def walk_and_process(modelnet_root, out_root, sample_surface_n, workers, chunk_size):
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
            futures.append(exe.submit(process_one_off, off, out_dir, sample_surface_n))
        for fut in as_completed(futures):
            _ = fut.result()
    train_count = len(glob.glob(os.path.join(out_root, 'ModelNet', '*', 'train', '*.npy')))
    test_count = len(glob.glob(os.path.join(out_root, 'ModelNet', '*', 'test', '*.npy')))
    train_files = sorted(glob.glob(os.path.join(out_root, 'ModelNet', '*', 'train', '*.npy')))
    test_files = sorted(glob.glob(os.path.join(out_root, 'ModelNet', '*', 'test', '*.npy')))
    class_dirs = sorted(glob.glob(os.path.join(out_root, 'ModelNet', '*')))
    logger.info(f"ModelNet train samples: {train_count}")
    logger.info(f"ModelNet test samples: {test_count}")
    if class_dirs:
        logger.info("ModelNet per-class sample counts:")
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            class_train = len(glob.glob(os.path.join(class_dir, 'train', '*.npy')))
            class_test = len(glob.glob(os.path.join(class_dir, 'test', '*.npy')))
            logger.info(f"  {class_name} train={class_train} test={class_test}")
    else:
        logger.warning("ModelNet class directories not found for per-class counts")
    if train_files:
        train_shape = np.load(train_files[0]).shape
        logger.info(f"ModelNet first train shape: {train_shape}")
    else:
        logger.warning("ModelNet train samples not found for shape check")
    if test_files:
        test_shape = np.load(test_files[0]).shape
        logger.info(f"ModelNet first test shape: {test_shape}")
    else:
        logger.warning("ModelNet test samples not found for shape check")
    logger.info("ModelNet processing finished")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--modelnet_root', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--sample_surface_n', type=int, default=10000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk_size', type=int, default=64)
    args = p.parse_args()
    logger = get_logger('build_modelnet', log_file=os.path.join(args.out_root, 'build_modelnet.log'))
    walk_and_process(args.modelnet_root, args.out_root, args.sample_surface_n, args.workers, args.chunk_size)
