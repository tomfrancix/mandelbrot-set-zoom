import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import importlib.metadata as importlib_metadata
except Exception:
    import importlib_metadata  # type: ignore

@dataclass(frozen=True)
class RunManifest:
    started_utc: str
    config: Dict[str, Any]
    python: Dict[str, Any]
    packages: Dict[str, str]
    git: Dict[str, Any]
    system: Dict[str, Any]
    renderer: Dict[str, Any]

def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _safe_pkg_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None

def build_manifest(*, config: Dict[str, Any], renderer_info: Dict[str, Any], git_commit: Optional[str]) -> RunManifest:
    pkgs = {}
    for name in ["numpy", "Pillow", "mpmath", "numba", "opencv-python", "moviepy", "natsort"]:
        v = _safe_pkg_version(name)
        if v:
            pkgs[name] = v

    return RunManifest(
        started_utc=_utc_iso(),
        config=config,
        python={"version": sys.version, "executable": sys.executable},
        packages=pkgs,
        git={"commit": git_commit},
        system={"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        renderer=renderer_info,
    )

def write_manifest(path: str, manifest: RunManifest) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.__dict__, f, indent=2, sort_keys=True)
