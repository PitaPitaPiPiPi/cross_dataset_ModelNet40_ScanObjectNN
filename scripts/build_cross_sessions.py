#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import json
from scripts.utils.logger import get_logger

logger = get_logger('build_cross_sessions')

def load_meta(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def gather_files_for_class(processed_root, ds_name, class_identifier, split):
    pattern = os.path.join(processed_root, ds_name, split, f"*{class_identifier}*.npy")
    files = glob.glob(pattern)
    return files

def build_session_npy(session_idx, mappings, processed_root, out_dir, split='train'):
    X_list = []
    Y_list = []
    meta_lines = []
    for gid in mappings:
        # mapping entry is a dict {dataset, class_name, global_id} or just global_id int
        if isinstance(gid, dict):
            ds = gid['dataset']
            cname = gid['class_name']
            glob_id = int(gid['global_id'])
        else:
            raise ValueError('mapping entries must be dicts')
        files = gather_files_for_class(processed_root, ds, cname, split)
        for f in files:
            pts = np.load(f)
            meta = load_meta(f.replace('.npy', '.json'))
            X_list.append(pts.astype(np.float32))
            Y_list.append(glob_id)
            meta_aug = dict(**meta)
            meta_aug.update(dict(global_id=glob_id, session=session_idx, orig_split=split))
            meta_lines.append(json.dumps(meta_aug, ensure_ascii=False))
    if len(X_list) == 0:
        logger.warning(f"No samples for session {session_idx} {split}")
        return
    X = np.stack(X_list)
    Y = np.array(Y_list, dtype=np.int64)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split}_data.npy"), X)
    np.save(os.path.join(out_dir, f"{split}_labels.npy"), Y)
    with open(os.path.join(out_dir, f"{split}_meta.jsonl"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(meta_lines))
    logger.info(f"Wrote {out_dir} with {len(X)} samples for {split}")

def build_all_sessions(processed_root, sessions_json, out_root, split='train'):
    sessions_obj = json.load(open(sessions_json, 'r', encoding='utf-8'))
    sessions = sessions_obj.get('sessions', [])
    cumulative = []
    for s in sessions:
        session_id = s['session_id']
        new_classes = s['train_classes']
        # new_classes in sessions_json are integer global_ids; we need dataset/class_name mapping
        # Expect processed_root to have a class_map.json which maps global_id -> {dataset, class_name}
        class_map_path = os.path.join(processed_root, 'class_map.json')
        if not os.path.exists(class_map_path):
            raise FileNotFoundError('class_map.json not found in processed_root. It must map global_id->(dataset,class_name).')
        class_map = json.load(open(class_map_path, 'r', encoding='utf-8'))
        mappings = [class_map[str(gid)] for gid in new_classes]
        cumulative.extend(mappings)
        out_dir = os.path.join(out_root, 'cross_sessions', f"session{session_id}")
        # build train: only new_classes
        build_session_npy(session_id, mappings, processed_root, out_dir, split='train')
        # build test: cumulative classes up to this session
        build_session_npy(session_id, cumulative, processed_root, out_dir, split='test')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--processed_root', required=True)
    p.add_argument('--sessions_json', required=True)
    p.add_argument('--out_root', required=True)
    args = p.parse_args()
    logger = get_logger('build_cross_sessions', log_file=os.path.join(args.out_root, 'build_cross_sessions.log'))
    build_all_sessions(args.processed_root, args.sessions_json, args.out_root)
