#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def npy_output_path(sample, out_root):
    source_name = Path(sample["output_pt"]).with_suffix(".npy").name
    return Path(out_root) / "shapenet_co3d" / str(sample["class_id"]) / sample["split"] / source_name


def load_sidecar(path):
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        return {}
    with open(sidecar, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_shape(path):
    arr = np.load(path, mmap_mode="r")
    return list(arr.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/shapenet_co3d_manifest.json")
    parser.add_argument("--out_root", default="outputs")
    parser.add_argument("--backup_existing", action="store_true", default=True)
    parser.add_argument("--no_backup_existing", dest="backup_existing", action="store_false")
    parser.add_argument("--strict_shape", action="store_true", default=True)
    parser.add_argument("--no_strict_shape", dest="strict_shape", action="store_false")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    out_root = Path(args.out_root)
    target_root = out_root / "shapenet_co3d"
    if target_root.exists():
        if args.backup_existing:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_root = out_root / f"shapenet_co3d_before_manifest_rebuild_{stamp}"
            shutil.move(str(target_root), str(backup_root))
        else:
            shutil.rmtree(target_root)

    counts = defaultdict(Counter)
    dataset_counts = defaultdict(Counter)
    invalid_shapes = []
    missing_sources = []

    for sample in manifest["samples"]:
        src = Path(sample["source_npy"])
        if not src.exists():
            missing_sources.append(str(src))
            continue

        shape = validate_shape(src)
        if shape != [1024, 3]:
            invalid_shapes.append({"source_npy": str(src), "shape": shape})
            if args.strict_shape:
                raise ValueError(f"{src}: shape {shape} != [1024, 3]")

        dst = npy_output_path(sample, out_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        meta = load_sidecar(src)
        meta.update(
            {
                "cross_dataset": "shapenet_co3d",
                "reconstructed_from_manifest": str(manifest_path),
                "source_npy": str(src),
                "source_dataset": sample["dataset"],
                "cross_class_id": int(sample["class_id"]),
                "cross_class_name": sample["class_name"],
                "cross_task": int(sample["task"]),
                "split": sample["split"],
                "source_id": sample["source_id"],
            }
        )
        if "official_subsets" in sample:
            meta["co3d_official_subsets"] = sample["official_subsets"]
        with open(dst.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        key = str(sample["class_id"])
        counts[key][sample["split"]] += 1
        dataset_counts[sample["dataset"]][sample["split"]] += 1

    if missing_sources:
        raise FileNotFoundError(f"Missing source .npy files: {len(missing_sources)}")

    report = {
        "manifest": str(manifest_path),
        "target_root": str(target_root),
        "dataset_counts": {k: dict(v) for k, v in sorted(dataset_counts.items())},
        "class_counts": {k: dict(v) for k, v in sorted(counts.items(), key=lambda kv: int(kv[0]))},
        "invalid_shape_count": len(invalid_shapes),
        "invalid_shapes": invalid_shapes,
    }
    report_path = Path("manifests/shapenet_co3d_npy_rebuild_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["dataset_counts"], indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
