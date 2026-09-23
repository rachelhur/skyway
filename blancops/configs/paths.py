
import os
from pathlib import Path

def project_root() -> Path:
    from blancops.configs.constants import get_workspace_dir
    return get_workspace_dir()

def data_dir() -> Path:
    override = os.environ.get("PERIFERA_DATA")
    return Path(override) if override else project_root() / "data"

def results_dir() -> Path:
    override = os.environ.get("PERIFERA_RESULTS")
    return Path(override) if override else project_root() / "results"

