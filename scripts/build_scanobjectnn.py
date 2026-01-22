#!/usr/bin/env python3
import os
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
from scripts.utils.io import save_npy_and_meta, load_h5_data
from scripts.utils.normalize import compute_centroid_and_scale, center_and_scale
from scripts.utils.fps import fps
from scripts.utils.logger import get_logger

def process_single_sample(args_tuple):
    (h5_path, idx, out_dir, target_n, percentile, seed, fps_backend) = args_tuple
    logger = get_logger('build_scanobjectnn')
    try:
        with h5py.File(h5_path, 'r') as f:
            pts = f['data'][idx].astype(np.float32)
            label = int(f['label'][idx]) if 'label' in f else None
            mask = f['mask'][idx].astype(bool) if 'mask' in f else None
        if mask is not None:
            obj_pts = pts[mask]
            used_mask = True
        else:
            obj_pts = pts
            used_mask = False
        centroid, scale = compute_centroid_and_scale(obj_pts, percentile)
        pts = center_and_scale(pts, centroid, scale)
        if pts.shape[0] >= target_n:
            idxs = fps(pts, target_n, seed=seed, backend=fps_backend)
            pts_out = pts[idxs]
        else:
            extra = target_n - pts.shape[0]
            choice = np.random.choice(pts.shape[0], extra, replace=True)
            pts_out = np.concatenate([pts, pts[choice]], axis=0)
        base = f"{os.path.splitext(os.path.basename(h5_path))[0]}_{idx:06d}"
        out_npy = os.path.join(out_dir, base + '.npy')
        meta = dict(orig=f"{h5_path}:idx={idx}", dataset='ScanObjectNN', label=label, centroid=centroid.tolist(), scale=scale, used_mask=used_mask)
        save_npy_and_meta(out_npy, pts_out, meta)
        return out_npy
    except Exception as e:
        logger.exception(f"Failed sample {h5_path} idx {idx}: {e}")
        return None

def process_h5_file(h5_path, out_root, variant, target_n, percentile, workers, chunk_size, seed, fps_backend='auto'):
    logger = get_logger('build_scanobjectnn')
    try:
        split_dir = os.path.dirname(h5_path).split(os.sep)[-1]
        out_dir = os.path.join(out_root, 'ScanObjectNN', split_dir)
        os.makedirs(out_dir, exist_ok=True)
        with h5py.File(h5_path, 'r') as f:
            N = f['data'].shape[0]
        args_list = [(h5_path, i, out_dir, target_n, percentile, seed, fps_backend) for i in range(N)]
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(process_single_sample, args) for args in args_list]
            for fut in as_completed(futures):
                _ = fut.result()
        logger.info(f"Finished processing h5: {h5_path}")
    except Exception as e:
        logger.exception(f"Failed processing h5 {h5_path}: {e}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5_root', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--variant', default='main_split')
    p.add_argument('--target_n', type=int, default=1024)
    p.add_argument('--percentile', type=float, default=99.0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--chunk_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fps_backend', type=str, default='auto')
    args = p.parse_args()
    logger = get_logger('build_scanobjectnn', log_file=os.path.join(args.out_root, 'build_scanobjectnn.log'))
    import glob
    pattern = os.path.join(args.h5_root, args.variant, '*.h5')
    h5_list = glob.glob(pattern)
    logger.info(f"Found {len(h5_list)} h5 files in {pattern}")
    for h5_path in h5_list:
        process_h5_file(h5_path, args.out_root, args.variant, args.target_n, args.percentile, args.workers, args.chunk_size, args.seed, args.fps_backend)
