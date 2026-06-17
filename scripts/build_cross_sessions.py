#!/usr/bin/env python3
# python -m scripts.build_cross_sessions --modelnet_root outputs/ModelNet --scanobjectnn_root outputs/ScanObjectNN --out_root outputs
import os
import argparse
import glob
import shutil
from scripts.utils.logger import get_logger

logger = get_logger('build_cross_sessions')

CLASS_ID_MAP = [
    {'class_id': 0, 'class_name': 'airplane', 'dataset': 'ModelNet'},
    {'class_id': 1, 'class_name': 'bathtub', 'dataset': 'ModelNet'},
    {'class_id': 2, 'class_name': 'bottle', 'dataset': 'ModelNet'},
    {'class_id': 3, 'class_name': 'bowl', 'dataset': 'ModelNet'},
    {'class_id': 4, 'class_name': 'car', 'dataset': 'ModelNet'},
    {'class_id': 5, 'class_name': 'cone', 'dataset': 'ModelNet'},
    {'class_id': 6, 'class_name': 'cup', 'dataset': 'ModelNet'},
    {'class_id': 7, 'class_name': 'curtain', 'dataset': 'ModelNet'},
    {'class_id': 8, 'class_name': 'flower pot', 'dataset': 'ModelNet'},
    {'class_id': 9, 'class_name': 'glass box', 'dataset': 'ModelNet'},
    {'class_id': 10, 'class_name': 'guitar', 'dataset': 'ModelNet'},
    {'class_id': 11, 'class_name': 'keyboard', 'dataset': 'ModelNet'},
    {'class_id': 12, 'class_name': 'lamp', 'dataset': 'ModelNet'},
    {'class_id': 13, 'class_name': 'laptop', 'dataset': 'ModelNet'},
    {'class_id': 14, 'class_name': 'mantel', 'dataset': 'ModelNet'},
    {'class_id': 15, 'class_name': 'night stand', 'dataset': 'ModelNet'},
    {'class_id': 16, 'class_name': 'person', 'dataset': 'ModelNet'},
    {'class_id': 17, 'class_name': 'piano', 'dataset': 'ModelNet'},
    {'class_id': 18, 'class_name': 'plant', 'dataset': 'ModelNet'},
    {'class_id': 19, 'class_name': 'radio', 'dataset': 'ModelNet'},
    {'class_id': 20, 'class_name': 'range hood', 'dataset': 'ModelNet'},
    {'class_id': 21, 'class_name': 'stairs', 'dataset': 'ModelNet'},
    {'class_id': 22, 'class_name': 'tent', 'dataset': 'ModelNet'},
    {'class_id': 23, 'class_name': 'tv stand', 'dataset': 'ModelNet'},
    {'class_id': 24, 'class_name': 'vase', 'dataset': 'ModelNet'},
    {'class_id': 25, 'class_name': 'xbox', 'dataset': 'ModelNet'},
    {'class_id': 26, 'class_name': 'cabinet', 'dataset': 'ScanObjectNN'},
    {'class_id': 27, 'class_name': 'chair', 'dataset': 'ScanObjectNN'},
    {'class_id': 28, 'class_name': 'desk', 'dataset': 'ScanObjectNN'},
    {'class_id': 29, 'class_name': 'display', 'dataset': 'ScanObjectNN'},
    {'class_id': 30, 'class_name': 'door', 'dataset': 'ScanObjectNN'},
    {'class_id': 31, 'class_name': 'shelf', 'dataset': 'ScanObjectNN'},
    {'class_id': 32, 'class_name': 'table', 'dataset': 'ScanObjectNN'},
    {'class_id': 33, 'class_name': 'bed', 'dataset': 'ScanObjectNN'},
    {'class_id': 34, 'class_name': 'sink', 'dataset': 'ScanObjectNN'},
    {'class_id': 35, 'class_name': 'sofa', 'dataset': 'ScanObjectNN'},
    {'class_id': 36, 'class_name': 'toilet', 'dataset': 'ScanObjectNN'},
]

def gather_files_for_class(root_dir, class_name, split):
    pattern = os.path.join(root_dir, class_name, split, '*.npy')
    return glob.glob(pattern)

def copy_class_split(class_id, class_name, split, src_root, out_root):
    files = gather_files_for_class(src_root, class_name, split)
    if len(files) == 0:
        logger.warning(f"No samples: class_id={class_id} class={class_name} split={split}")
        return 0
    out_dir = os.path.join(out_root, 'modelnet_scanobjectnn', str(class_id), split)
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        dst = os.path.join(out_dir, os.path.basename(f))
        shutil.copy2(f, dst)
    logger.info(f"Copied {len(files)} files: class_id={class_id} split={split}")
    return len(files)

def build_cross_dataset(modelnet_root, scanobjectnn_root, out_root):
    modelnet_train = 0
    modelnet_test = 0
    scanobject_train = 0
    scanobject_test = 0
    for entry in CLASS_ID_MAP:
        class_id = entry['class_id']
        class_name = entry['class_name']
        ds = entry['dataset']
        src_root = modelnet_root if ds == 'ModelNet' else scanobjectnn_root
        class_dir = class_name.replace(' ', '_') if ds == 'ModelNet' else class_name
        train_count = copy_class_split(class_id, class_dir, 'train', src_root, out_root)
        test_count = copy_class_split(class_id, class_dir, 'test', src_root, out_root)
        if class_id <= 25:
            modelnet_train += train_count
            modelnet_test += test_count
        else:
            scanobject_train += train_count
            scanobject_test += test_count
    logger.info(f"ModelNet train: {modelnet_train}")
    logger.info(f"ModelNet test: {modelnet_test}")
    logger.info(f"ScanObjectNN train: {scanobject_train}")
    logger.info(f"ScanObjectNN test: {scanobject_test}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--modelnet_root', required=True)
    p.add_argument('--scanobjectnn_root', required=True)
    p.add_argument('--out_root', required=True)
    args = p.parse_args()
    logger = get_logger('build_cross_sessions', log_file=os.path.join(args.out_root, 'build_cross_sessions.log'))
    build_cross_dataset(args.modelnet_root, args.scanobjectnn_root, args.out_root)
