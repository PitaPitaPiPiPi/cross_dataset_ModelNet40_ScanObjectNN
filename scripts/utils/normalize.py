import numpy as np
from utils.fps import fps

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

def pc_normalize_unified(
    pc: np.ndarray,
    openshape: bool = False,
    return_meta: bool = False,
    use_fps: bool = False,
    fps_k: int = None,
    seed: int = None,
):
    pc = pc.copy().astype(np.float32)

    # ---- 1. 軸入れ替え（回転相当） ----
    if openshape:
        pc[:, [1, 2]] = pc[:, [2, 1]]

    # ---- 2. FPS（ライブラリ無ければnumpyに自動fallback） ----
    if use_fps:
        if fps_k is None:
            raise ValueError("fps_k must be specified when use_fps is True")
        if fps_k > pc.shape[0]:
            raise ValueError(
                f"fps_k ({fps_k}) cannot be greater than number of points ({pc.shape[0]})"
            )

        indices = fps(pc, fps_k, seed=seed, backend="auto")
        pc = pc[indices]

    # ---- 3. 中心化 ----
    centroid = pc.mean(axis=0)
    centered = pc - centroid[None, :]

    # ---- 4. 正規化 ----
    dists = np.linalg.norm(centered, axis=1)
    maxd = float(np.max(dists))

    if maxd <= 0:
        normalized = np.zeros_like(centered)
    else:
        normalized = centered / maxd

    if return_meta:
        return normalized.astype(np.float32), centroid, maxd

    return normalized.astype(np.float32)
