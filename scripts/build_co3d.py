#!/usr/bin/env python3
# python -m scripts.build_co3d --co3d_root raw_datasets/co3d --out_root outputs --categories apple,backpack --num_points 1024
import argparse
import hashlib
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from scripts.utils.io import save_npy_and_meta
from scripts.utils.logger import get_logger
from scripts.utils.normalize import pc_normalize_unified


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "co3d_class_map.json"
CROSS_CONFIG_PATH = REPO_ROOT / "configs" / "shapenet_co3d_class_map.json"

PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def normalize_key(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("._") or "sample"


def load_category_map(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data["classes"]
    alias_to_category = {}
    for entry in entries:
        keys = {entry["class_name"], entry["class_dir"]}
        keys.update(entry.get("aliases", []))
        for key in keys:
            alias_to_category[normalize_key(key)] = entry["class_dir"]
    return entries, alias_to_category


def parse_categories(categories_arg):
    entries, alias_to_category = load_category_map()
    if not categories_arg:
        return [entry["class_dir"] for entry in entries]

    categories = []
    for raw in categories_arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        category = alias_to_category.get(normalize_key(raw))
        if category is None:
            raise ValueError(f"Unknown CO3D category: {raw}")
        categories.append(category)
    return categories


def parse_cross_categories(cross_config_path=CROSS_CONFIG_PATH):
    entries, _ = load_category_map()
    known_categories = {entry["class_dir"] for entry in entries}
    with open(cross_config_path, "r", encoding="utf-8") as f:
        cross_entries = json.load(f)["classes"]
    categories = [
        entry["class_dir"]
        for entry in cross_entries
        if entry["dataset"] == "CO3D"
    ]
    unknown = sorted(set(categories) - known_categories)
    if unknown:
        raise ValueError(f"Unknown CO3D categories in cross class map: {unknown}")
    return categories


def discover_sequences(category_dir):
    sequences = {}
    for pointcloud in sorted(Path(category_dir).glob("*/pointcloud.ply")):
        sequence_name = pointcloud.parent.name
        sequences[sequence_name] = pointcloud
    return sequences


def split_from_text(text):
    text = str(text).lower()
    if "train" in text:
        return "train"
    if "test" in text or "val" in text or "valid" in text:
        return "test"
    return None


def match_sequence_name(text, known_sequences):
    text = str(text).replace("\\", "/")
    if text in known_sequences:
        return text
    for part in text.split("/"):
        if part in known_sequences:
            return part
    return None


def collect_set_list_sequences(obj, known_sequences, split_hint, seq_to_splits):
    if isinstance(obj, dict):
        current_split = split_hint
        for key, value in obj.items():
            key_split = split_from_text(key) or current_split
            if key in {"sequence_name", "sequence", "seq_name", "seq"}:
                seq_name = match_sequence_name(value, known_sequences)
                if seq_name and key_split:
                    seq_to_splits.setdefault(seq_name, set()).add(key_split)
            collect_set_list_sequences(value, known_sequences, key_split, seq_to_splits)
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], str):
            seq_name = match_sequence_name(obj[0], known_sequences)
            if seq_name and split_hint:
                seq_to_splits.setdefault(seq_name, set()).add(split_hint)
        for item in obj:
            collect_set_list_sequences(item, known_sequences, split_hint, seq_to_splits)
    elif isinstance(obj, str):
        seq_name = match_sequence_name(obj, known_sequences)
        if seq_name and split_hint:
            seq_to_splits.setdefault(seq_name, set()).add(split_hint)


def split_from_set_lists(category_dir, sequences):
    logger = get_logger("build_co3d")
    set_lists_dir = Path(category_dir) / "set_lists"
    if not set_lists_dir.is_dir():
        return None

    known_sequences = set(sequences)
    seq_to_splits = {}
    json_files = sorted(set_lists_dir.glob("*.json"))
    for json_path in json_files:
        file_split = split_from_text(json_path.name)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            collect_set_list_sequences(data, known_sequences, file_split, seq_to_splits)
        except Exception as exc:
            logger.warning(f"Skip unreadable set_list: {json_path}: {exc}")

    if not seq_to_splits:
        logger.warning(f"No usable split in {set_lists_dir}")
        return None

    conflicts = {
        seq: splits for seq, splits in seq_to_splits.items() if len(splits) > 1
    }
    if conflicts:
        logger.warning(
            f"set_lists conflicts detected; fallback to deterministic split: {Path(category_dir).name}"
        )
        return None

    if set(seq_to_splits) != known_sequences:
        logger.warning(
            f"set_lists incomplete; fallback to deterministic split: {Path(category_dir).name}"
        )
        return None

    split_map = {"train": [], "test": []}
    for seq_name, splits in seq_to_splits.items():
        split = next(iter(splits))
        split_map[split].append(seq_name)
    split_map["train"].sort()
    split_map["test"].sort()
    if not split_map["train"] or not split_map["test"]:
        logger.warning(
            f"set_lists missing train or test; fallback to deterministic split: {Path(category_dir).name}"
        )
        return None
    return split_map


def deterministic_sequence_split(sequence_names, category, train_ratio, seed):
    ordered = sorted(sequence_names)
    category_seed = seed + int(hashlib.sha1(category.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(category_seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        train_count = len(shuffled)
    else:
        train_count = int(len(shuffled) * train_ratio)
        train_count = min(max(train_count, 1), len(shuffled) - 1)
    train = sorted(shuffled[:train_count])
    test = sorted(shuffled[train_count:])
    return {"train": train, "test": test}


def load_existing_manifest(manifest_path):
    if not Path(manifest_path).exists():
        return {
            "dataset": "CO3D",
            "split_unit": "sequence",
            "categories": {},
        }
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_split_manifest(out_root, category, split_map, split_source, seed, train_ratio):
    manifest_path = Path(out_root) / "CO3D" / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_existing_manifest(manifest_path)
    manifest["seed"] = seed
    manifest["train_ratio"] = train_ratio
    manifest["categories"][category] = {
        "split_source": split_source,
        "train": split_map.get("train", []),
        "test": split_map.get("test", []),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def split_map_from_output_files(out_root, category):
    category_dir = Path(out_root) / "CO3D" / category
    return {
        split: sorted(path.stem for path in (category_dir / split).glob("*.npy"))
        for split in ("train", "test")
    }


def manifest_expected_names(split_map):
    expected = {}
    for split in ("train", "test"):
        expected[split] = {safe_name(name) for name in split_map.get(split, [])}
    return expected


def manifest_skip_report(category, expected_split_map, output_split_map):
    expected = manifest_expected_names(expected_split_map or {})
    output = {split: set(output_split_map.get(split, [])) for split in ("train", "test")}
    skipped_by_split = {
        split: sorted(expected[split] - output[split]) for split in ("train", "test")
    }
    expected_count = sum(len(values) for values in expected.values())
    output_count = sum(len(values) for values in output.values())
    skipped_count = sum(len(values) for values in skipped_by_split.values())
    return {
        "category": category,
        "expected_count": expected_count,
        "output_count": output_count,
        "skipped_missing_output_count": skipped_count,
        "skipped_missing_output_by_split": skipped_by_split,
    }


def save_split_manifest_from_outputs(
    out_root,
    category,
    split_source,
    seed,
    train_ratio,
    expected_split_map=None,
    logger=None,
):
    manifest_path = Path(out_root) / "CO3D" / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_existing_manifest(manifest_path)
    manifest["seed"] = seed
    manifest["train_ratio"] = train_ratio
    manifest["manifest_source"] = "rebuilt_from_existing_npy_outputs"

    output_split_map = split_map_from_output_files(out_root, category)
    report = manifest_skip_report(category, expected_split_map, output_split_map)
    manifest["categories"][category] = {
        "split_source": split_source,
        "manifest_source": "existing_npy_outputs",
        "train": output_split_map.get("train", []),
        "test": output_split_map.get("test", []),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if logger and report["skipped_missing_output_count"]:
        logger.warning(
            "CO3D %s: skipped %d manifest entries without output files "
            "(expected=%d output=%d)",
            category,
            report["skipped_missing_output_count"],
            report["expected_count"],
            report["output_count"],
        )
    return report


def output_categories(out_root):
    co3d_root = Path(out_root) / "CO3D"
    if not co3d_root.is_dir():
        return []
    return sorted(
        path.name
        for path in co3d_root.iterdir()
        if path.is_dir()
        and (any((path / "train").glob("*.npy")) or any((path / "test").glob("*.npy")))
    )


def rebuild_split_manifest_from_outputs(
    out_root,
    seed=None,
    train_ratio=None,
    categories=None,
    report_path=None,
    validate_num_points=None,
    logger=None,
):
    manifest_path = Path(out_root) / "CO3D" / "split_manifest.json"
    old_manifest = load_existing_manifest(manifest_path)
    if categories is None:
        category_names = sorted(
            set(output_categories(out_root)) | set(old_manifest.get("categories", {}))
        )
    else:
        category_names = sorted(set(categories))

    manifest = {
        "dataset": "CO3D",
        "split_unit": "sequence",
        "manifest_source": "rebuilt_from_existing_npy_outputs",
        "categories": {},
    }
    if seed is not None:
        manifest["seed"] = seed
    elif "seed" in old_manifest:
        manifest["seed"] = old_manifest["seed"]
    if train_ratio is not None:
        manifest["train_ratio"] = train_ratio
    elif "train_ratio" in old_manifest:
        manifest["train_ratio"] = old_manifest["train_ratio"]

    report = {
        "manifest": str(manifest_path),
        "categories": {},
        "total_output_count": 0,
        "total_skipped_missing_output_count": 0,
        "shape_bad_count": 0,
        "finite_bad_count": 0,
        "shape_bad": [],
        "finite_bad": [],
    }

    expected_shape = None
    if validate_num_points is not None:
        expected_shape = (int(validate_num_points), 3)

    for category in category_names:
        output_split_map = split_map_from_output_files(out_root, category)
        output_count = sum(len(output_split_map.get(split, [])) for split in ("train", "test"))
        old_entry = old_manifest.get("categories", {}).get(category, {})
        expected_split_map = {
            "train": old_entry.get("train", []),
            "test": old_entry.get("test", []),
        }
        skip_report = manifest_skip_report(category, expected_split_map, output_split_map)
        if output_count == 0:
            if skip_report["expected_count"]:
                report["categories"][category] = skip_report
                report["total_skipped_missing_output_count"] += skip_report[
                    "skipped_missing_output_count"
                ]
            continue

        split_source = old_entry.get("split_source", "existing_outputs")
        manifest["categories"][category] = {
            "split_source": split_source,
            "manifest_source": "existing_npy_outputs",
            "train": output_split_map.get("train", []),
            "test": output_split_map.get("test", []),
        }

        if expected_shape is not None:
            category_dir = Path(out_root) / "CO3D" / category
            for split in ("train", "test"):
                for output_path in sorted((category_dir / split).glob("*.npy")):
                    try:
                        points = np.load(output_path, mmap_mode="r")
                        if points.shape != expected_shape:
                            report["shape_bad"].append(
                                {
                                    "category": category,
                                    "path": str(output_path),
                                    "shape": list(points.shape),
                                }
                            )
                        elif not np.isfinite(np.asarray(points)).all():
                            report["finite_bad"].append(
                                {"category": category, "path": str(output_path)}
                            )
                    except Exception as exc:
                        report["shape_bad"].append(
                            {"category": category, "path": str(output_path), "error": str(exc)}
                        )

        report["categories"][category] = skip_report
        report["total_output_count"] += output_count
        report["total_skipped_missing_output_count"] += skip_report[
            "skipped_missing_output_count"
        ]

    report["shape_bad_count"] = len(report["shape_bad"])
    report["finite_bad_count"] = len(report["finite_bad"])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if logger:
        logger.info(
            "Rebuilt CO3D manifest from outputs: categories=%d outputs=%d skipped_missing_output=%d",
            len(manifest["categories"]),
            report["total_output_count"],
            report["total_skipped_missing_output_count"],
        )
        for category, category_report in report["categories"].items():
            skipped = category_report["skipped_missing_output_count"]
            if skipped:
                logger.warning(
                    "CO3D %s: skipped %d metadata/manifest entries without output files",
                    category,
                    skipped,
                )
    return report


def load_ply_points(path):
    with open(path, "rb") as f:
        if f.readline().rstrip(b"\r\n") != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        ply_format = None
        vertex_count = None
        in_vertex_element = False
        vertex_properties = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            try:
                parts = line.decode("ascii").strip().split()
            except UnicodeDecodeError as exc:
                raise ValueError(f"PLY header is not ASCII: {path}") from exc
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "end_header":
                break
            if parts[0] == "format" and len(parts) == 3:
                ply_format = parts[1]
            elif parts[0] == "element" and len(parts) == 3:
                in_vertex_element = parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex_element:
                if len(parts) != 3 or parts[1] not in PLY_SCALAR_DTYPES:
                    raise ValueError(f"Unsupported PLY vertex property: {' '.join(parts)}")
                vertex_properties.append((parts[2], parts[1]))

        if vertex_count is None:
            raise ValueError(f"PLY vertex count not found: {path}")
        if ply_format == "ascii":
            return load_ascii_ply_points(path)
        byte_order = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(
            ply_format
        )
        if byte_order is None:
            raise ValueError(f"Unsupported PLY format {ply_format!r}: {path}")
        names = [name for name, _ in vertex_properties]
        missing = [axis for axis in ("x", "y", "z") if axis not in names]
        if missing:
            raise ValueError(f"PLY vertex properties missing {missing}: {path}")
        dtype = np.dtype(
            [(name, byte_order + PLY_SCALAR_DTYPES[kind]) for name, kind in vertex_properties]
        )
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
        if len(vertices) != vertex_count:
            raise ValueError(f"Unexpected EOF in binary PLY vertices: {path}")
        return np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(
            np.float32, copy=False
        )


def load_ascii_ply_points(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = []
        for line in f:
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        vertex_count = None
        for line in header:
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                break
        if vertex_count is None:
            raise ValueError(f"PLY vertex count not found: {path}")

        points = []
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not points:
        raise ValueError(f"No ASCII PLY points found: {path}")
    return np.asarray(points, dtype=np.float32)


def sanitize_points(points, num_points, context):
    logger = get_logger("build_co3d")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"{context}: expected 2D points, got {points.shape}")
    if points.shape[1] != 3 and points.shape[0] == 3:
        points = points.T
    if points.shape[1] != 3:
        raise ValueError(f"{context}: expected shape (N, 3), got {points.shape}")

    finite_mask = np.isfinite(points).all(axis=1)
    if not finite_mask.all():
        removed = int(points.shape[0] - finite_mask.sum())
        logger.warning(f"{context}: removed {removed} non-finite points")
        points = points[finite_mask]
    if points.shape[0] == 0:
        raise ValueError(f"{context}: point cloud is empty after filtering")

    if points.shape[0] < num_points:
        logger.warning(f"{context}: pad {points.shape[0]} -> {num_points}")
        repeats = int(np.ceil(num_points / points.shape[0]))
        points = np.tile(points, (repeats, 1))[:num_points]
    return points.astype(np.float32)


def process_sequence(task):
    logger = get_logger("build_co3d")
    out_npy = task["out_npy"]
    if os.path.exists(out_npy) and task["skip_existing"] and not task["overwrite"]:
        return {"ok": True, "skipped": True, "out": out_npy}

    try:
        points = load_ply_points(task["src"])
        points = sanitize_points(points, task["num_points"], task["src"])
        points, centroid, scale = pc_normalize_unified(
            points,
            openshape=False,
            return_meta=True,
            use_fps=True,
            fps_k=task["num_points"],
            seed=task["seed"],
        )
        if points.shape != (task["num_points"], 3):
            raise ValueError(f"Unexpected output shape {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("Output contains NaN or Inf")

        meta = {
            "orig": task["src"],
            "dataset": "CO3D",
            "category": task["category"],
            "sequence_name": task["sequence_name"],
            "split": task["split"],
            "centroid": np.asarray(centroid).astype(float).tolist(),
            "scale": float(scale),
            "num_points": int(task["num_points"]),
            "source_format": "ply",
        }
        save_npy_and_meta(out_npy, points, meta)
        return {
            "ok": True,
            "skipped": False,
            "out": out_npy,
            "category": task["category"],
            "split": task["split"],
            "shape": tuple(points.shape),
        }
    except Exception as exc:
        logger.exception(f"Failed: {task['src']}: {exc}")
        return {"ok": False, "src": task["src"], "error": str(exc)}


def run_tasks(tasks, workers):
    if workers <= 1:
        return [process_sequence(task) for task in tasks]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_sequence, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def summarize_category(out_root, category):
    logger = get_logger("build_co3d")
    category_dir = Path(out_root) / "CO3D" / category
    train_files = sorted((category_dir / "train").glob("*.npy"))
    test_files = sorted((category_dir / "test").glob("*.npy"))
    logger.info(f"CO3D {category}: train={len(train_files)} test={len(test_files)}")
    files = train_files or test_files
    if files:
        logger.info(f"CO3D {category} sample shape: {np.load(files[0]).shape}")
    else:
        logger.warning(f"CO3D {category}: no sample for shape check")


def build_category(category, args):
    logger = get_logger("build_co3d")
    category_dir = Path(args.co3d_root) / category
    if not category_dir.exists():
        message = f"CO3D category not found: {category_dir}"
        if args.strict:
            raise FileNotFoundError(message)
        logger.warning(message)
        return {"category": category, "processed": 0, "skipped": True}

    sequences = discover_sequences(category_dir)
    if not sequences:
        message = f"No pointcloud.ply sequences found for CO3D category: {category}"
        if args.strict:
            raise FileNotFoundError(message)
        logger.warning(message)
        return {"category": category, "processed": 0, "skipped": True}

    split_source = "deterministic"
    split_map = None
    if args.use_set_lists:
        split_map = split_from_set_lists(category_dir, sequences)
        if split_map is not None:
            split_source = "set_lists"
    if split_map is None:
        split_map = deterministic_sequence_split(
            sequences.keys(), category, args.train_ratio, args.seed
        )

    tasks = []
    for split in ("train", "test"):
        for sequence_name in split_map.get(split, []):
            pointcloud = sequences[sequence_name]
            out_dir = Path(args.out_root) / "CO3D" / category / split
            out_npy = out_dir / f"{safe_name(sequence_name)}.npy"
            tasks.append(
                {
                    "src": str(pointcloud),
                    "out_npy": str(out_npy),
                    "category": category,
                    "sequence_name": sequence_name,
                    "split": split,
                    "num_points": args.num_points,
                    "seed": args.seed,
                    "skip_existing": args.skip_existing,
                    "overwrite": args.overwrite,
                }
            )

    results = run_tasks(tasks, args.workers)
    failures = [res for res in results if not res.get("ok")]
    if failures:
        raise RuntimeError(f"Failed to process {len(failures)} CO3D samples in {category}")

    save_split_manifest_from_outputs(
        args.out_root,
        category,
        split_source,
        args.seed,
        args.train_ratio,
        expected_split_map=split_map,
        logger=logger,
    )

    skipped = sum(1 for res in results if res.get("skipped"))
    if skipped:
        logger.info(f"Skipped existing CO3D samples for {category}: {skipped}")
    summarize_category(args.out_root, category)
    return {"category": category, "processed": len(results) - skipped, "skipped": False}


def build_co3d(args):
    logger = get_logger(
        "build_co3d",
        log_file=os.path.join(args.out_root, "build_co3d.log"),
    )
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip_existing and --overwrite cannot be used together")

    if getattr(args, "cross_classes_only", False) and args.categories:
        raise ValueError("--cross_classes_only and --categories cannot be used together")
    categories = (
        parse_cross_categories()
        if getattr(args, "cross_classes_only", False)
        else parse_categories(args.categories)
    )
    logger.info(f"CO3D categories: {', '.join(categories)}")
    results = []
    for category in categories:
        results.append(build_category(category, args))
    processed = sum(result["processed"] for result in results)
    rebuild_split_manifest_from_outputs(
        args.out_root,
        seed=args.seed,
        train_ratio=args.train_ratio,
        logger=logger,
    )
    logger.info(f"CO3D done: processed={processed}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--co3d_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--categories", default=None)
    parser.add_argument("--cross_classes_only", action="store_true")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_set_lists", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_co3d(parse_args())
