#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def load_points(path):
    points = np.load(path).astype(np.float32, copy=False)
    if points.shape == (1024, 3):
        return points, None
    note = {"original_shape": list(points.shape), "action": None}
    flat = points.reshape(-1, points.shape[-1])[:, :3]
    if len(flat) >= 1024:
        points = flat[:1024]
        note["action"] = "deterministic_trim_first_1024"
    else:
        pad_count = 1024 - len(flat)
        pad = np.repeat(flat[-1:], pad_count, axis=0) if len(flat) else np.zeros((pad_count, 3), dtype=np.float32)
        points = np.concatenate([flat, pad], axis=0)
        note["action"] = "deterministic_pad_last_point"
    return points.astype(np.float32, copy=False), note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/shapenet_co3d_manifest.json")
    parser.add_argument("--out_root", default="v11-shapenet-co3d")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    shape_fixes = []
    written = 0
    for sample in manifest["samples"]:
        dst = Path(sample["output_pt"])
        if args.out_root != "v11-shapenet-co3d":
            dst = out_root / Path(*dst.parts[1:])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.overwrite:
            continue
        points, note = load_points(sample["source_npy"])
        if note is not None:
            note.update({"source_npy": sample["source_npy"], "output_pt": str(dst)})
            shape_fixes.append(note)
        torch.save(torch.from_numpy(points), dst)
        written += 1

    log_path = Path("manifests/v11_pt_write_report.json")
    log_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "out_root": str(out_root),
                "written": written,
                "shape_fix_count": len(shape_fixes),
                "shape_fixes": shape_fixes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written={written} shape_fix_count={len(shape_fixes)} report={log_path}")


if __name__ == "__main__":
    main()
