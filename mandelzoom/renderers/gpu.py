from __future__ import annotations

from typing import Tuple, Dict, Any
from mandelzoom.util.logging_setup import get_logger

def probe_cuda() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        from numba import cuda  # type: ignore
        if not cuda.is_available():
            return info
        dev = cuda.get_current_device()
        info.update({
            "available": True,
            "name": getattr(dev, "name", None),
            "compute_capability": getattr(dev, "compute_capability", None),
            "max_threads_per_block": getattr(dev, "MAX_THREADS_PER_BLOCK", None),
            "warp_size": getattr(dev, "WARP_SIZE", None),
        })
        return info
    except Exception as e:
        info["error"] = str(e)
        return info

def render_frame_gpu_double(*, center: Tuple[float, float], zoom: float, width: int, height: int, max_iter: int, frame_id: str):
    logger = get_logger()
    try:
        from gpu_renderer import render_gpu_frame
    except Exception as e:
        raise RuntimeError(f"GPU renderer not available: {e}") from e
    logger.info("[Frame %s] GPU(double) render start zoom=%s iter=%s", frame_id, zoom, max_iter)
    img = render_gpu_frame(center, zoom, width, height, max_iter)
    logger.info("[Frame %s] GPU(double) render done", frame_id)
    return img

def render_frame_gpu_extreme(*, center: Tuple[float, float], zoom: float, width: int, height: int, max_iter: int, frame_id: str):
    logger = get_logger()
    try:
        from extreme_gpu_renderer import render_gpu_frame
    except Exception as e:
        raise RuntimeError(f"Extreme GPU renderer not available: {e}") from e
    logger.info("[Frame %s] GPU(extreme) render start zoom=%s iter=%s", frame_id, zoom, max_iter)
    img = render_gpu_frame(center, zoom, width, height, max_iter)
    logger.info("[Frame %s] GPU(extreme) render done", frame_id)
    return img
