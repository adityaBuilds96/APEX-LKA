"""
LKA Project — Shared Configuration Loader
==========================================
Loads project_config.yaml and resolves all paths relative to the project root.
Import this in any script:

    from src.config import cfg, PATHS
"""

import os
import yaml
from pathlib import Path

# ── Project root is two levels above this file (src/config.py) ─────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "project_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


cfg = _load_config()


def resolve(relative_path: str) -> Path:
    """Resolve a config path relative to project root."""
    return PROJECT_ROOT / relative_path


class _Paths:
    """Convenience object: PATHS.raw_frames, PATHS.train, etc."""
    def __init__(self, path_cfg: dict):
        for key, value in path_cfg.items():
            setattr(self, key, resolve(value))

    def all(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


PATHS = _Paths(cfg["paths"])


def ensure_dirs():
    """Create all project directories if they don't exist."""
    for path in PATHS.all().values():
        path.mkdir(parents=True, exist_ok=True)
    # sub-dirs that aren't in config
    for sub in ["images", "masks"]:
        (PATHS.annotated / sub).mkdir(parents=True, exist_ok=True)
        (PATHS.train / sub).mkdir(parents=True, exist_ok=True)
        (PATHS.val / sub).mkdir(parents=True, exist_ok=True)
        (PATHS.test / sub).mkdir(parents=True, exist_ok=True)
    (PATHS.logs / "training").mkdir(parents=True, exist_ok=True)
    (PATHS.logs / "inference").mkdir(parents=True, exist_ok=True)
    (PATHS.results / "plots").mkdir(parents=True, exist_ok=True)
    (PATHS.results / "metrics").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config loaded: {list(cfg.keys())}")
    ensure_dirs()
    print("All project directories verified.")
