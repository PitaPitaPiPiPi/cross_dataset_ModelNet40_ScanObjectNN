#!/usr/bin/env python3
"""
Visualize and verify class_id-based cross-dataset structure.

This script reads modelnet_scanobjectnn outputs and saves simple bar charts
describing sample counts per class_id for train and test splits.

Outputs (saved under out_root/modelnet_scanobjectnn/visualization/):
 - train_class_counts.png
 - test_class_counts.png

Usage example:
python scripts/visualize_cross_sessions.py \
    --out_root outputs
"""
import os
import argparse
import glob
import matplotlib.pyplot as plt
from scripts.utils.logger import get_logger

logger = get_logger('visualize_cross_sessions')

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
    {'class_id': 26, 'class_name': 'Cabinet', 'dataset': 'ScanObjectNN'},
    {'class_id': 27, 'class_name': 'Chair', 'dataset': 'ScanObjectNN'},
    {'class_id': 28, 'class_name': 'Desk', 'dataset': 'ScanObjectNN'},
    {'class_id': 29, 'class_name': 'Display', 'dataset': 'ScanObjectNN'},
    {'class_id': 30, 'class_name': 'Door', 'dataset': 'ScanObjectNN'},
    {'class_id': 31, 'class_name': 'Shelf', 'dataset': 'ScanObjectNN'},
    {'class_id': 32, 'class_name': 'Table', 'dataset': 'ScanObjectNN'},
    {'class_id': 33, 'class_name': 'Bed', 'dataset': 'ScanObjectNN'},
    {'class_id': 34, 'class_name': 'Sink', 'dataset': 'ScanObjectNN'},
    {'class_id': 35, 'class_name': 'Sofa', 'dataset': 'ScanObjectNN'},
    {'class_id': 36, 'class_name': 'Toilet', 'dataset': 'ScanObjectNN'},
]

def class_label(entry):
    return f"{entry['class_id']}:{entry['class_name']}"

def count_files_for_class(out_root, class_id, split):
    class_dir = os.path.join(out_root, 'modelnet_scanobjectnn', str(class_id), split)
    return len(glob.glob(os.path.join(class_dir, '*.npy')))

def plot_counts(class_ids, counts, labels, out_png, title):
    plt.figure(figsize=(14, 4))
    plt.bar(range(len(class_ids)), counts)
    plt.xticks(range(len(class_ids)), labels, rotation=45, ha='right', fontsize=8)
    plt.xlabel('class_id:class_name')
    plt.ylabel('sample count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main(args):
    base_dir = os.path.join(args.out_root, 'modelnet_scanobjectnn')
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"modelnet_scanobjectnn not found under: {args.out_root}")

    class_ids = [e['class_id'] for e in CLASS_ID_MAP]
    labels = [class_label(e) for e in CLASS_ID_MAP]
    train_counts = [count_files_for_class(args.out_root, cid, 'train') for cid in class_ids]
    test_counts = [count_files_for_class(args.out_root, cid, 'test') for cid in class_ids]

    total_train = sum(train_counts)
    total_test = sum(test_counts)
    print('=== Dataset Summary ===')
    print(f'Train samples : {total_train}')
    print(f'Test samples  : {total_test}')

    viz_dir = args.out_dir if args.out_dir else os.path.join(args.out_root, 'modelnet_scanobjectnn', 'visualization')
    os.makedirs(viz_dir, exist_ok=True)

    train_png = os.path.join(viz_dir, 'train_class_counts.png')
    test_png = os.path.join(viz_dir, 'test_class_counts.png')
    plot_counts(class_ids, train_counts, labels, train_png, 'Train Class Counts')
    plot_counts(class_ids, test_counts, labels, test_png, 'Test Class Counts')
    print(f'Plots saved to: {viz_dir}')
    print('Done.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_root', required=True, help='output root where modelnet_scanobjectnn/ is located')
    p.add_argument('--out_dir', default=None, help='optional explicit visualization output dir')
    args = p.parse_args()
    main(args)
