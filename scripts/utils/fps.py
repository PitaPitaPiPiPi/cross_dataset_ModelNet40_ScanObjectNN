import numpy as np

_HAS_PT3D = False
_HAS_MN2 = False
try:
    import torch
    from pytorch3d.ops import sample_farthest_points
    _HAS_PT3D = True
except Exception:
    _HAS_PT3D = False

try:
    import modelnet2_ops
    _HAS_MN2 = True
except Exception:
    _HAS_MN2 = False

def fps_numpy(points: np.ndarray, k: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    N = points.shape[0]
    if k >= N:
        if k == N:
            return np.arange(N)
        idx = list(range(N))
        extra = k - N
        idx.extend(np.random.choice(N, extra, replace=True).tolist())
        return np.array(idx, dtype=np.int64)
    selected = np.zeros(k, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float64)
    selected[0] = np.random.randint(0, N)
    for i in range(1, k):
        last = points[selected[i-1]]
        d = np.sum((points - last[None, :])**2, axis=1)
        distances = np.minimum(distances, d)
        selected[i] = int(np.argmax(distances))
    return selected

def fps(points: np.ndarray, k: int, seed: int = None, backend: str = 'auto'):
    if backend == 'auto':
        if _HAS_PT3D:
            backend = 'pytorch3d'
        elif _HAS_MN2:
            backend = 'modelnet2_ops'
        else:
            backend = 'numpy'
    if backend == 'pytorch3d' and _HAS_PT3D:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pts = torch.from_numpy(points[None].astype(np.float32)).to(device)
        sampled = sample_farthest_points(pts, K=k)[1].squeeze(0).cpu().numpy().astype(np.int64)
        return sampled
    if backend == 'modelnet2_ops' and _HAS_MN2:
        return modelnet2_ops.fps(points.astype(np.float32), k).astype(np.int64)
    return fps_numpy(points, k, seed)
