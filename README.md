# pointcloud-cross-dataset-builder

This repository builds processed point cloud datasets (ModelNet40 and ScanObjectNN),
applies normalization, FPS sampling, and aggregates cross-session datasets for
class-incremental continual learning experiments.

After cloning and installing requirements, edit `configs/preprocessing.yaml` and
`configs/sessions.json` to match your local paths and desired session splits.

Use `python scripts/build_all.py --config configs/preprocessing.yaml --sessions configs/sessions.json --out outputs/NAME`
to run the full pipeline.

Notes:
- The scripts produce processed .npy files (per-sample) and session-level train/test npy sets.
- `modelnet2_ops` support is optional; install if you want its FPS implementation (CUDA needed).
