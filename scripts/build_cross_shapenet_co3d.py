#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

from scripts.utils.logger import get_logger


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_MAP_PATH = REPO_ROOT / "configs" / "shapenet_co3d_class_map.json"


def normalize_key(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("._") or "sample"


def load_class_map(path=CLASS_MAP_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["classes"]


def build_dir_index(root):
    root = Path(root)
    if not root.exists():
        return {}
    return {normalize_key(path.name): path for path in root.iterdir() if path.is_dir()}


def find_source_class_dir(root, entry, dir_index):
    root = Path(root)
    for key in (entry["class_dir"], entry["class_name"]):
        direct = root / key
        if direct.is_dir():
            return direct

    lookup_keys = {
        normalize_key(entry["class_dir"]),
        normalize_key(entry["class_name"]),
        normalize_key(entry["class_name"].replace("_", " ")),
        normalize_key(entry["class_dir"].replace("_", " ")),
    }
    for key in lookup_keys:
        if key in dir_index:
            return dir_index[key]
    return None


def copy_sample(src_npy, dst_npy, entry, source_dataset, overwrite):
    if dst_npy.exists() and not overwrite:
        return False

    dst_npy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_npy, dst_npy)

    src_json = src_npy.with_suffix(".json")
    dst_json = dst_npy.with_suffix(".json")
    if src_json.exists():
        with open(src_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    meta.update(
        {
            "cross_dataset": "shapenet_co3d",
            "cross_class_id": int(entry["class_id"]),
            "cross_session": int(entry["session"]),
            "source_dataset": source_dataset,
            "source_class_name": entry["class_name"],
        }
    )
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return True


def copy_class(entry, source_root, out_root, overwrite, strict):
    logger = get_logger("build_cross_shapenet_co3d")
    source_dataset = entry["dataset"]
    prefix = "shapenet" if source_dataset == "ShapeNet" else "co3d"
    dir_index = build_dir_index(source_root)
    class_dir = find_source_class_dir(source_root, entry, dir_index)
    class_id = int(entry["class_id"])

    if class_dir is None:
        message = (
            f"Missing {source_dataset} class for S2C class_id={class_id}: "
            f"{entry['class_dir']}"
        )
        if strict:
            raise FileNotFoundError(message)
        logger.warning(message)
        return {"train": 0, "test": 0}

    counts = {"train": 0, "test": 0}
    for split in ("train", "test"):
        files = sorted((class_dir / split).glob("*.npy"))
        if not files:
            message = (
                f"No {split} samples for S2C class_id={class_id} "
                f"({source_dataset}/{class_dir.name})"
            )
            if strict:
                raise FileNotFoundError(message)
            logger.warning(message)
            continue

        out_dir = Path(out_root) / "shapenet_co3d" / str(class_id) / split
        for src_npy in files:
            dst_name = f"{prefix}_{class_id}_{safe_name(src_npy.name)}"
            dst_npy = out_dir / dst_name
            copied = copy_sample(src_npy, dst_npy, entry, source_dataset, overwrite)
            if copied:
                counts[split] += 1

    return counts


def summarize(out_root, class_map):
    logger = get_logger("build_cross_shapenet_co3d")
    cross_root = Path(out_root) / "shapenet_co3d"
    dataset_counts = defaultdict(lambda: {"train": 0, "test": 0})
    session_counts = defaultdict(lambda: {"train": 0, "test": 0})

    for entry in class_map:
        class_id = int(entry["class_id"])
        train_count = len(list((cross_root / str(class_id) / "train").glob("*.npy")))
        test_count = len(list((cross_root / str(class_id) / "test").glob("*.npy")))
        logger.info(
            f"S2C class_id={class_id} session={entry['session']} "
            f"{entry['dataset']}/{entry['class_name']}: "
            f"train={train_count} test={test_count}"
        )
        dataset_counts[entry["dataset"]]["train"] += train_count
        dataset_counts[entry["dataset"]]["test"] += test_count
        session_counts[int(entry["session"])]["train"] += train_count
        session_counts[int(entry["session"])]["test"] += test_count

    for dataset, counts in sorted(dataset_counts.items()):
        logger.info(
            f"S2C dataset summary {dataset}: "
            f"train={counts['train']} test={counts['test']}"
        )
    for session_id, counts in sorted(session_counts.items()):
        logger.info(
            f"S2C session summary {session_id}: "
            f"train={counts['train']} test={counts['test']}"
        )


def build_cross(args):
    logger = get_logger(
        "build_cross_shapenet_co3d",
        log_file=os.path.join(args.out_root, "build_cross_shapenet_co3d.log"),
    )
    cross_root = Path(args.out_root) / "shapenet_co3d"
    if args.overwrite and cross_root.exists():
        shutil.rmtree(cross_root)

    class_map = load_class_map()
    for entry in class_map:
        source_root = (
            args.shapenet_root if entry["dataset"] == "ShapeNet" else args.co3d_root
        )
        counts = copy_class(
            entry,
            source_root,
            args.out_root,
            overwrite=args.overwrite,
            strict=args.strict,
        )
        logger.debug(
            f"Copied S2C class_id={entry['class_id']}: "
            f"train={counts['train']} test={counts['test']}"
        )

    summarize(args.out_root, class_map)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_root", default="outputs/ShapeNet")
    parser.add_argument("--co3d_root", default="outputs/CO3D")
    parser.add_argument("--out_root", default="outputs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_cross(parse_args())
