import os
import json
import numpy as np
from typing import Dict, Any

def save_npy_and_meta(out_npy: str, points: np.ndarray, meta: Dict[str, Any]):
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, points.astype(np.float32))
    meta_path = out_npy.replace('.npy', '.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_h5_data(h5_path: str):
    import h5py

    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:] if 'data' in f else None
        label = f['label'][:] if 'label' in f else None
        mask = f['mask'][:] if 'mask' in f else None
    return data, label, mask

def write_aggregated_h5(out_h5: str, data_array: np.ndarray, labels: np.ndarray, metas: list):
    import h5py

    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('data', data=data_array, compression='gzip')
        f.create_dataset('label', data=labels)
        dt = h5py.string_dtype(encoding='utf-8')
        metas_str = [json.dumps(m, ensure_ascii=False) for m in metas]
        f.create_dataset('meta', data=np.array(metas_str, dtype=object), dtype=dt)
