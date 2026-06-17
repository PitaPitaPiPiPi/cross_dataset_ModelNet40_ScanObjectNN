#!/usr/bin/env python3
# python -m scripts.smoke_test_dataset --dataset co3d --data_root outputs/CO3D --split train
import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPENET_CONFIG = REPO_ROOT / "configs" / "shapenet_class_map.json"
CO3D_CONFIG = REPO_ROOT / "configs" / "co3d_class_map.json"
S2C_CLASS_MAP = REPO_ROOT / "configs" / "shapenet_co3d_class_map.json"
S2C_SESSIONS = REPO_ROOT / "configs" / "shapenet_co3d_sessions.json"
M2S_SESSIONS = REPO_ROOT / "configs" / "sessions.json"


DATASET_ALIASES = {
    "modelnet": "modelnet",
    "scanobjectnn": "scanobjectnn",
    "m2s": "m2s",
    "modelnet_scanobjectnn": "m2s",
    "shapenet": "shapenet",
    "shapeNet": "shapenet",
    "co3d": "co3d",
    "s2c": "s2c",
    "shapenet_co3d": "s2c",
}


def normalize_dataset(value):
    key = value.lower()
    if key == "s2c":
        return "s2c"
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset: {value}")
    return DATASET_ALIASES[key]


def normalize_key(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config_classes(dataset):
    if dataset == "shapenet":
        return load_json(SHAPENET_CONFIG)["classes"]
    if dataset == "co3d":
        return load_json(CO3D_CONFIG)["classes"]
    return None


def build_dir_index(root):
    root = Path(root)
    return {normalize_key(path.name): path for path in root.iterdir() if path.is_dir()}


def resolve_single_classes(dataset, data_root, strict):
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    config_classes = load_config_classes(dataset)
    if config_classes is None:
        classes = []
        for label, path in enumerate(sorted([p for p in root.iterdir() if p.is_dir()])):
            classes.append((label, path.name, path))
        return classes

    dir_index = build_dir_index(root)
    classes = []
    missing = []
    for label, entry in enumerate(config_classes):
        path = root / entry["class_dir"]
        if not path.is_dir():
            path = dir_index.get(normalize_key(entry["class_dir"]))
        if not path or not path.is_dir():
            missing.append(entry["class_dir"])
            continue
        classes.append((label, entry["class_dir"], path))

    if strict and missing:
        raise FileNotFoundError(f"Missing classes: {', '.join(missing)}")
    return classes


def session_class_ids(session_path, session):
    sessions = load_json(session_path)["sessions"]
    if session is None:
        class_ids = []
        for item in sessions:
            class_ids.extend(item["train_classes"])
        return sorted(set(class_ids))
    for item in sessions:
        if int(item["session_id"]) == int(session):
            return item["train_classes"]
    raise ValueError(f"Unknown session: {session}")


def resolve_cross_classes(dataset, data_root, session, strict):
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    if dataset == "s2c":
        expected_ids = [
            int(entry["class_id"]) for entry in load_json(S2C_CLASS_MAP)["classes"]
        ]
        if session is not None:
            expected_ids = session_class_ids(S2C_SESSIONS, session)
    else:
        expected_ids = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        if session is not None:
            expected_ids = session_class_ids(M2S_SESSIONS, session)

    classes = []
    missing = []
    for class_id in sorted(expected_ids):
        path = root / str(class_id)
        if path.is_dir():
            classes.append((class_id, str(class_id), path))
        else:
            missing.append(str(class_id))

    if strict and missing:
        raise FileNotFoundError(f"Missing class_id dirs: {', '.join(missing)}")
    return classes


def first_metadata(npy_path):
    json_path = Path(npy_path).with_suffix(".json")
    if not json_path.exists():
        return None
    return load_json(json_path)


def collect_samples(classes, split):
    samples = []
    class_count = 0
    for label, class_name, class_dir in classes:
        files = sorted((class_dir / split).glob("*.npy"))
        if files:
            class_count += 1
        for path in files:
            samples.append((path, label, class_name))
    return class_count, samples


def run(args):
    dataset = normalize_dataset(args.dataset)
    if dataset in {"m2s", "s2c"}:
        classes = resolve_cross_classes(dataset, args.data_root, args.session, args.strict)
    else:
        classes = resolve_single_classes(dataset, args.data_root, args.strict)

    class_count, samples = collect_samples(classes, args.split)
    if not samples:
        raise FileNotFoundError(
            f"No samples found: dataset={dataset} root={args.data_root} split={args.split}"
        )

    first_path, first_label, _ = samples[0]
    first = np.load(first_path)
    expected_shape = (args.num_points, 3)
    if first.shape != expected_shape:
        raise ValueError(f"First sample shape {first.shape} != {expected_shape}: {first_path}")

    metadata = first_metadata(first_path)
    print(f"dataset: {dataset}")
    print(f"split: {args.split}")
    if args.session is not None:
        print(f"session: {args.session}")
    print(f"classes: {class_count}")
    print(f"samples: {len(samples)}")
    print(f"first sample shape: {first.shape}")
    print(f"first label: {first_label}")
    if metadata is not None:
        print(f"first metadata: {json.dumps(metadata, ensure_ascii=False)}")
    else:
        print("first metadata: None")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--num_points", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
