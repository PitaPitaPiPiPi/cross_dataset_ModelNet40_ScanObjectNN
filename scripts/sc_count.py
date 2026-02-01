import h5py
import numpy as np
from collections import Counter

h5_path = "/home/kita/Desktop/master-projects/make_datasets/cross_dataset_ModelNet40_ScanObjectNN/data/h5_files/main_split_nobg/test_objectdataset.h5"

with h5py.File(h5_path, "r") as f:
    print("=== Keys in h5 file ===")
    for k in f.keys():
        print(f"{k}: shape={f[k].shape}, dtype={f[k].dtype}")

    # ScanObjectNN standard keys
    points = f["data"][:]    # (N, num_points, 3)
    labels = f["label"][:]   # (N, 1) or (N,)

# ---- 全体情報 ----
print("\n=== Dataset overview ===")
print(f"points shape : {points.shape}")
print(f"labels shape : {labels.shape}")
print(f"num samples  : {points.shape[0]}")
print(f"num points   : {points.shape[1]}")

# ---- クラスごとのサンプル数 ----
labels = labels.squeeze()
counter = Counter(labels.tolist())

print("\n=== Samples per class ===")
for cls in sorted(counter):
    print(f"class {cls:2d}: {counter[cls]}")
