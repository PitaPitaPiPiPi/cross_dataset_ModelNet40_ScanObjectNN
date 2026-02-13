import numpy as np

def compute_centroid_and_scale(object_pts: np.ndarray, percentile: float = 99.0):
    if object_pts.size == 0:
        raise ValueError("object_pts is empty")
    centroid = object_pts.mean(axis=0)
    dists = np.linalg.norm(object_pts - centroid[None, :], axis=1)
    scale = float(np.percentile(dists, percentile))
    if scale <= 0:
        scale = float(np.max(dists)) if np.max(dists) > 0 else 1.0
    return centroid.astype(np.float32), float(scale)

def center_and_scale(points: np.ndarray, centroid: np.ndarray, scale: float):
    return (points - centroid[None, :]) / (scale + 1e-12)

def pc_normalize_unified(pc: np.ndarray, openshape: bool = False, return_meta=False):
    pc = pc.copy().astype(np.float32)

    if openshape:
        pc[:, [1, 2]] = pc[:, [2, 1]]

    centroid = pc.mean(axis=0)
    centered = pc - centroid[None, :]
    dists = np.linalg.norm(centered, axis=1)
    maxd = float(np.max(dists))

    if maxd <= 0:
        normalized = np.zeros_like(centered)
    else:
        normalized = centered / maxd

    if return_meta:
        return normalized.astype(np.float32), centroid, maxd

    return normalized.astype(np.float32)
