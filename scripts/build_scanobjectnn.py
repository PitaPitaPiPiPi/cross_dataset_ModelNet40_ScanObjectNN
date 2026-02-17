#!/usr/bin/env python3
import os
import argparse
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
from scripts.utils.io import save_npy_and_meta
from scripts.utils.normalize import pc_normalize_unified
from scripts.utils.logger import get_logger

SCANOBJECTNN_CLASS_NAMES = [
    'bag',   # 0
    'bed',   # 1
    'bin',   # 2
    'box',   # 3
    'cabinet',  # 4
    'chair',    # 5
    'desk',     # 6
    'display',  # 7
    'door',     # 8
    'pillow',   # 9
    'shelf',    # 10
    'sink',     # 11
    'sofa',     # 12
    'table',    # 13
    'toilet',   # 14
]


def process_single_sample(args_tuple):
    (h5_path, idx, out_root, split) = args_tuple
    logger = get_logger('build_scanobjectnn')
    try:
        with h5py.File(h5_path, 'r') as f:
            pts = f['data'][idx].astype(np.float32)
            if 'label' in f:
                label_raw = np.asarray(f['label'][idx]).squeeze()
                label = int(label_raw)
            else:
                label = None
        if label is None or label < 0 or label >= len(SCANOBJECTNN_CLASS_NAMES):
            raise ValueError(f"Invalid label {label} at idx {idx}")
        class_name = SCANOBJECTNN_CLASS_NAMES[label]
        out_dir = os.path.join(out_root, 'ScanObjectNN', class_name, split)
        os.makedirs(out_dir, exist_ok=True)
        centroid = pts.mean(axis=0)
        pts, centroid, scale = pc_normalize_unified(pts, openshape=True, return_meta=True)
        prefix = 'training' if split == 'train' else 'test'
        base = f"{prefix}_{class_name}_{idx:06d}"
        out_npy = os.path.join(out_dir, base + '.npy')
        meta = dict(orig=f"{h5_path}:idx={idx}", dataset='ScanObjectNN', label=label, class_name=class_name, centroid=centroid.tolist(), scale=scale)
        save_npy_and_meta(out_npy, pts, meta)
        return out_npy
    except Exception as e:
        logger.exception(f"Failed sample {h5_path} idx {idx}: {e}")
        return None

def process_h5_file(h5_path, out_root, split, workers, chunk_size):
    logger = get_logger('build_scanobjectnn')
    try:
        with h5py.File(h5_path, 'r') as f:
            N = f['data'].shape[0]
        args_list = [(h5_path, i, out_root, split) for i in range(N)]
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(process_single_sample, args) for args in args_list]
            for fut in as_completed(futures):
                _ = fut.result()
        logger.info(f"Finished processing h5: {h5_path} ({split})")
    except Exception as e:
        logger.exception(f"Failed processing h5 {h5_path}: {e}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5_root', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--split_dir', default='main_split_nobg')
    p.add_argument('--train_h5', default='training_objectdataset.h5')
    p.add_argument('--test_h5', default='test_objectdataset.h5')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk_size', type=int, default=64)
    args = p.parse_args()
    logger = get_logger('build_scanobjectnn', log_file=os.path.join(args.out_root, 'build_scanobjectnn.log'))
    split_root = os.path.join(args.h5_root, args.split_dir)
    train_path = os.path.join(split_root, args.train_h5)
    test_path = os.path.join(split_root, args.test_h5)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train h5 not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test h5 not found: {test_path}")
    process_h5_file(train_path, args.out_root, 'train', args.workers, args.chunk_size)
    process_h5_file(test_path, args.out_root, 'test', args.workers, args.chunk_size)
    train_count = len(glob.glob(os.path.join(args.out_root, 'ScanObjectNN', '*', 'train', '*.npy')))
    test_count = len(glob.glob(os.path.join(args.out_root, 'ScanObjectNN', '*', 'test', '*.npy')))
    train_files = sorted(glob.glob(os.path.join(args.out_root, 'ScanObjectNN', '*', 'train', '*.npy')))
    test_files = sorted(glob.glob(os.path.join(args.out_root, 'ScanObjectNN', '*', 'test', '*.npy')))
    logger.info(f"ScanObjectNN train samples: {train_count}")
    logger.info(f"ScanObjectNN test samples: {test_count}")
    logger.info("ScanObjectNN per-class sample counts:")
    for class_name in SCANOBJECTNN_CLASS_NAMES:
        class_train = len(glob.glob(os.path.join(args.out_root, 'ScanObjectNN', class_name, 'train', '*.npy')))
        class_test = len(glob.glob(os.path.join(args.out_root, 'ScanObjectNN', class_name, 'test', '*.npy')))
        logger.info(f"  {class_name} train={class_train} test={class_test}")
    if train_files:
        train_shape = np.load(train_files[0]).shape
        logger.info(f"ScanObjectNN first train shape: {train_shape}")
    else:
        logger.warning("ScanObjectNN train samples not found for shape check")
    if test_files:
        test_shape = np.load(test_files[0]).shape
        logger.info(f"ScanObjectNN first test shape: {test_shape}")
    else:
        logger.warning("ScanObjectNN test samples not found for shape check")
