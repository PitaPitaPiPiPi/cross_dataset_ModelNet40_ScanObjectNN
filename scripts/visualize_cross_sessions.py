#!/usr/bin/env python3
"""
Visualize and verify cross-session dataset structure.

This script reads cross_sessions outputs for a given session and prints a short
summary table (base classes, novel classes, tasks, counts) and saves simple
bar charts describing sample counts per class for train and test splits.

Outputs (saved under out_root/cross_sessions/session{ID}/visualization/):
 - train_class_counts.png
 - test_class_counts.png

Usage example:
python scripts/visualize_cross_sessions.py \
  --out_root outputs/modelnet_scanobject_v1 \
  --session_id 1 \
  --sessions configs/sessions.json
"""
import os
import argparse
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
from scripts.utils.logger import get_logger

logger = get_logger('visualize_cross_sessions')

def load_session_files(out_root, session_id):
    base = os.path.join(out_root, 'cross_sessions', f'session{session_id}')
    train_data_p = os.path.join(base, 'train_data.npy')
    train_labels_p = os.path.join(base, 'train_labels.npy')
    train_meta_p = os.path.join(base, 'train_meta.jsonl')
    test_data_p = os.path.join(base, 'test_data.npy')
    test_labels_p = os.path.join(base, 'test_labels.npy')
    test_meta_p = os.path.join(base, 'test_meta.jsonl')
    for p in [train_data_p, train_labels_p, test_data_p, test_labels_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    train_labels = np.load(train_labels_p)
    test_labels = np.load(test_labels_p)
    return {
        'train_labels': train_labels,
        'test_labels': test_labels,
        'train_data_path': train_data_p,
        'test_data_path': test_data_p,
        'train_meta': train_meta_p if os.path.exists(train_meta_p) else None,
        'test_meta': test_meta_p if os.path.exists(test_meta_p) else None
    }

def read_sessions_json(sessions_json):
    with open(sessions_json, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    sessions = obj.get('sessions', [])
    return sessions

def compute_summary(sessions, session_idx, train_labels, test_labels):
    # sessions: list of {session_id, train_classes}
    # session_idx is the session number (1-based)
    # base classes = classes in session 1
    base_classes = set(sessions[0]['train_classes'])
    novel_classes = set()
    for s in sessions[1:]:
        novel_classes.update(s['train_classes'])
    # tasks = number of sessions
    tasks = len(sessions)
    # compute counts
    train_in_base = int(np.sum(np.isin(train_labels, list(base_classes))))
    test_in_base = int(np.sum(np.isin(test_labels, list(base_classes))))
    test_in_novel = int(np.sum(np.isin(test_labels, list(novel_classes))))
    summary = {
        'Base Classes': len(base_classes),
        'Novel Classes': len(novel_classes),
        'Tasks': tasks,
        'Train in Base': train_in_base,
        'Test in Base': test_in_base,
        'Test in Novel': test_in_novel
    }
    return summary, base_classes, novel_classes

def plot_counts(labels, out_png, title):
    ctr = Counter(labels.tolist())
    classes = sorted(ctr.keys())
    counts = [ctr[c] for c in classes]
    plt.figure(figsize=(10,4))
    plt.bar(range(len(classes)), counts)
    plt.xlabel('class (sorted id)')
    plt.ylabel('sample count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main(args):
    sess = read_sessions_json(args.sessions)
    sess_map = {s['session_id']: s for s in sess}
    if args.session_id not in sess_map:
        raise ValueError('session_id not found in sessions.json')
    data = load_session_files(args.out_root, args.session_id)
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    summary, base_classes, novel_classes = compute_summary(sess, args.session_id, train_labels, test_labels)
    print('=== Session Summary ===')
    for k,v in summary.items():
        print(f'{k:15s}: {v}')
    # create visualization folder
    viz_dir = args.out_dir if args.out_dir else os.path.join(args.out_root, 'cross_sessions', f'session{args.session_id}', 'visualization')
    os.makedirs(viz_dir, exist_ok=True)
    # plots
    train_png = os.path.join(viz_dir, 'train_class_counts.png')
    test_png = os.path.join(viz_dir, 'test_class_counts.png')
    plot_counts(train_labels, train_png, f'Session {args.session_id} Train Class Counts')
    plot_counts(test_labels, test_png, f'Session {args.session_id} Test Class Counts')
    print(f'Plots saved to: {viz_dir}')
    print('Done.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_root', required=True, help='output root where cross_sessions/ is located')
    p.add_argument('--session_id', type=int, required=True)
    p.add_argument('--sessions', required=True, help='path to configs/sessions.json')
    p.add_argument('--out_dir', default=None, help='optional explicit visualization output dir')
    args = p.parse_args()
    main(args)
