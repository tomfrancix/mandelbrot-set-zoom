from __future__ import annotations

import glob
import os
from mandelzoom.util.logging_setup import get_logger

def encode_with_opencv(*, input_dir: str, output_file: str, fps: int) -> None:
    logger = get_logger()
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(f"OpenCV not installed: {e}") from e

    frames = sorted([p for p in glob.glob(os.path.join(input_dir, "*")) if p.lower().endswith((".png",".jpg",".jpeg"))])
    if not frames:
        raise ValueError(f"No frames found in {input_dir}")

    first = cv2.imread(frames[0])
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    h, w, _ = first.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_file}")

    logger.info("Encoding video %s from %s frames (%sx%s @ %sfps)", output_file, len(frames), w, h, fps)
    for i, path in enumerate(frames):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        out.write(img)
        if i % 200 == 0:
            logger.info("Encoded %s/%s frames", i, len(frames))
    out.release()
    logger.info("Video written: %s", output_file)
