#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from scripts.utils.io import save_npy_and_meta
from scripts.utils.logger import get_logger
from scripts.utils.normalize import pc_normalize_unified


SUPPORTED_EXTS = {".obj", ".off", ".ply", ".npy"}
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "shapenet_class_map.json"


def normalize_key(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("._") or "sample"


def load_class_map(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data["classes"]
    alias_to_entry = {}
    for entry in entries:
        keys = {entry["class_name"], entry["class_dir"]}
        keys.update(entry.get("aliases", []))
        if entry.get("synset_id"):
            keys.add(entry["synset_id"])
        for key in keys:
            alias_to_entry[normalize_key(key)] = entry
    return entries, alias_to_entry


def find_supported_files(root):
    files = []
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def output_stem(src, root):
    rel = Path(src).relative_to(root).with_suffix("")
    return safe_name("__".join(rel.parts))


def discover_samples(shapenet_root, class_aliases):
    root = Path(shapenet_root)
    logger = get_logger("build_shapenet")
    samples = []
    splitless = []

    if not root.exists():
        raise FileNotFoundError(f"ShapeNet root not found: {root}")

    for class_root in sorted([p for p in root.iterdir() if p.is_dir()]):
        entry = class_aliases.get(normalize_key(class_root.name))
        if entry is None:
            logger.warning(f"Skipping unknown ShapeNet class folder: {class_root}")
            continue

        class_name = entry["class_name"]
        class_dir = entry["class_dir"]
        split_dirs = {
            split: class_root / split
            for split in ("train", "test")
            if (class_root / split).is_dir()
        }

        if split_dirs:
            for split, split_root in split_dirs.items():
                for src in find_supported_files(split_root):
                    samples.append(
                        {
                            "src": str(src),
                            "class_name": class_name,
                            "class_dir": class_dir,
                            "split": split,
                            "from_manifest_split": False,
                        }
                    )
        else:
            for src in find_supported_files(class_root):
                splitless.append(
                    {
                        "src": str(src),
                        "class_name": class_name,
                        "class_dir": class_dir,
                        "split": None,
                        "from_manifest_split": True,
                    }
                )

    return samples, splitless


def deterministic_split(splitless_samples, root, train_ratio, seed):
    by_class = defaultdict(list)
    for sample in splitless_samples:
        by_class[sample["class_dir"]].append(sample)

    manifest = {
        "dataset": "ShapeNet",
        "seed": seed,
        "train_ratio": train_ratio,
        "classes": {},
    }
    assigned = []

    for class_dir, class_samples in sorted(by_class.items()):
        ordered = sorted(class_samples, key=lambda s: os.path.relpath(s["src"], root))
        class_seed = seed + int(hashlib.sha1(class_dir.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(class_seed)
        shuffled = ordered[:]
        rng.shuffle(shuffled)

        if len(shuffled) <= 1:
            train_count = len(shuffled)
        else:
            train_count = int(len(shuffled) * train_ratio)
            train_count = min(max(train_count, 1), len(shuffled) - 1)

        train_paths = set(sample["src"] for sample in shuffled[:train_count])
        manifest["classes"][class_dir] = {"train": [], "test": []}
        for sample in ordered:
            split = "train" if sample["src"] in train_paths else "test"
            assigned_sample = dict(sample)
            assigned_sample["split"] = split
            assigned.append(assigned_sample)
            rel = os.path.relpath(sample["src"], root).replace(os.sep, "/")
            manifest["classes"][class_dir][split].append(rel)

    return assigned, manifest


def save_split_manifest(out_root, manifest):
    manifest_path = Path(out_root) / "ShapeNet" / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_mesh_points(path, sample_surface_n):
    import trimesh

    loaded = trimesh.load(str(path), process=True)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geom for geom in loaded.geometry.values() if hasattr(geom, "vertices")
        ]
        if not geometries:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        loaded = trimesh.util.concatenate(geometries)

    if hasattr(loaded, "faces") and len(loaded.faces) > 0:
        return np.asarray(loaded.sample(sample_surface_n), dtype=np.float32)
    if hasattr(loaded, "vertices") and len(loaded.vertices) > 0:
        return np.asarray(loaded.vertices, dtype=np.float32)
    raise ValueError(f"No vertices found in mesh: {path}")


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


def load_npy_points(path):
    points = np.load(path)
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError(f"Point cloud must be 2D, got shape {points.shape}")
    if points.shape[1] != 3 and points.shape[0] == 3:
        points = points.T
    if points.shape[1] != 3:
        raise ValueError(f"Point cloud must have 3 columns, got shape {points.shape}")
    return points.astype(np.float32)


def load_ply_points(path, sample_surface_n):
    try:
        import open3d as o3d

        point_cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(point_cloud.points, dtype=np.float32)
        if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] == 3:
            return points
    except Exception:
        pass

    try:
        import trimesh

        loaded = trimesh.load(str(path), process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = [
                geom for geom in loaded.geometry.values() if hasattr(geom, "vertices")
            ]
            if not geometries:
                raise ValueError(f"No PLY geometry found in scene: {path}")
            loaded = trimesh.util.concatenate(geometries)

        if hasattr(loaded, "faces") and len(getattr(loaded, "faces", [])) > 0:
            return np.asarray(loaded.sample(sample_surface_n), dtype=np.float32)
        if hasattr(loaded, "vertices") and len(loaded.vertices) > 0:
            return np.asarray(loaded.vertices, dtype=np.float32)
    except Exception:
        pass

    return load_ascii_ply_points(path)


def load_points(path, sample_surface_n):
    ext = Path(path).suffix.lower()
    if ext in {".obj", ".off"}:
        return load_mesh_points(path, sample_surface_n), ext[1:]
    if ext == ".ply":
        return load_ply_points(path, sample_surface_n), "ply"
    if ext == ".npy":
        return load_npy_points(path), "npy"
    raise ValueError(f"Unsupported input format: {path}")


def sanitize_points(points, num_points, seed, context):
    logger = get_logger("build_shapenet")
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
        logger.warning(
            f"{context}: only {points.shape[0]} points; deterministic repeat to {num_points}"
        )
        repeats = int(np.ceil(num_points / points.shape[0]))
        points = np.tile(points, (repeats, 1))[:num_points]

    return points.astype(np.float32)


def process_one_sample(task):
    logger = get_logger("build_shapenet")
    out_npy = task["out_npy"]
    if os.path.exists(out_npy) and task["skip_existing"] and not task["overwrite"]:
        return {"ok": True, "skipped": True, "out": out_npy}

    try:
        points, source_format = load_points(task["src"], task["sample_surface_n"])
        points = sanitize_points(
            points,
            task["num_points"],
            task["seed"],
            task["src"],
        )
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
            "dataset": "ShapeNet",
            "class_name": task["class_name"],
            "class_dir": task["class_dir"],
            "split": task["split"],
            "centroid": np.asarray(centroid).astype(float).tolist(),
            "scale": float(scale),
            "num_points": int(task["num_points"]),
            "source_format": source_format,
        }
        save_npy_and_meta(out_npy, points, meta)
        return {
            "ok": True,
            "skipped": False,
            "out": out_npy,
            "class_dir": task["class_dir"],
            "split": task["split"],
            "shape": tuple(points.shape),
        }
    except Exception as exc:
        logger.exception(f"Failed processing {task['src']}: {exc}")
        return {"ok": False, "src": task["src"], "error": str(exc)}


def build_tasks(samples, shapenet_root, out_root, args):
    tasks = []
    dataset_root = Path(out_root) / "ShapeNet"
    for sample in samples:
        out_dir = dataset_root / sample["class_dir"] / sample["split"]
        out_npy = out_dir / f"{output_stem(sample['src'], Path(shapenet_root))}.npy"
        tasks.append(
            {
                **sample,
                "out_npy": str(out_npy),
                "num_points": args.num_points,
                "sample_surface_n": args.sample_surface_n,
                "seed": args.seed,
                "skip_existing": args.skip_existing,
                "overwrite": args.overwrite,
            }
        )
    return tasks


def run_tasks(tasks, workers):
    if workers <= 1:
        return [process_one_sample(task) for task in tasks]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one_sample, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def summarize(out_root):
    logger = get_logger("build_shapenet")
    dataset_root = Path(out_root) / "ShapeNet"
    class_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    first_sample = None
    for class_dir in class_dirs:
        train_files = sorted((class_dir / "train").glob("*.npy"))
        test_files = sorted((class_dir / "test").glob("*.npy"))
        logger.info(
            f"ShapeNet {class_dir.name}: train={len(train_files)} test={len(test_files)}"
        )
        if first_sample is None:
            files = train_files or test_files
            if files:
                first_sample = files[0]

    if first_sample is not None:
        logger.info(f"ShapeNet sample shape: {np.load(first_sample).shape}")
    else:
        logger.warning("ShapeNet sample shape unavailable: no .npy files found")


def build_shapenet(args):
    logger = get_logger(
        "build_shapenet",
        log_file=os.path.join(args.out_root, "build_shapenet.log"),
    )
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip_existing and --overwrite cannot be used together")

    _, aliases = load_class_map()
    split_samples, splitless_samples = discover_samples(args.shapenet_root, aliases)
    assigned_samples, manifest = deterministic_split(
        splitless_samples,
        args.shapenet_root,
        args.train_ratio,
        args.seed,
    )
    samples = split_samples + assigned_samples
    logger.info(
        f"Found {len(samples)} ShapeNet samples "
        f"({len(split_samples)} pre-split, {len(assigned_samples)} manifest-split)"
    )
    if manifest["classes"]:
        save_split_manifest(args.out_root, manifest)

    if not samples:
        logger.warning("No ShapeNet input files found")
        return

    tasks = build_tasks(samples, args.shapenet_root, args.out_root, args)
    results = run_tasks(tasks, args.workers)
    failures = [res for res in results if not res.get("ok")]
    if failures:
        raise RuntimeError(f"Failed to process {len(failures)} ShapeNet samples")

    skipped = sum(1 for res in results if res.get("skipped"))
    if skipped:
        logger.info(f"Skipped existing ShapeNet samples: {skipped}")
    summarize(args.out_root)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--sample_surface_n", type=int, default=10000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_shapenet(parse_args())
