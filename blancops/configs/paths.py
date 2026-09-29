"""Single source of truth for re-used filesystem paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ------------------------------------------------------------------ #
# Workspace                                                          #
# ------------------------------------------------------------------ #

PROFILE_POINTER_FILE = Path.home() / ".blancops_profile"


def get_workspace_dir() -> Path:
    """Active workspace: $BLANCOPS_WORKSPACE, else the path in ~/.blancops_profile, else ~/.blancops.

    Returns
    -------
    Path
        Resolved workspace root.
    """
    env_workspace = os.getenv("BLANCOPS_WORKSPACE")
    if env_workspace:
        return Path(env_workspace).resolve()
    if PROFILE_POINTER_FILE.exists():
        saved_path = PROFILE_POINTER_FILE.read_text().strip()
        if saved_path:
            return Path(saved_path).resolve()
    return Path.home() / ".blancops"


@dataclass(frozen=True)
class WorkspacePaths:
    """Directory layout of a workspace.

    Parameters
    ----------
    root : Path
        Workspace root.
    """

    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def deployable_models(self) -> Path:
        return self.root / "deployable_models"

    @property
    def model_comparison(self) -> Path:
        return self.experiments / "model_comparison"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def train_data(self) -> Path:
        return self.data / "train"

    @property
    def test_suite(self) -> Path:
        return self.data / "test_suite"

    @property
    def des_data(self) -> Path:
        return self.train_data / "des"

    @property
    def des_fits(self) -> Path:
        return self.des_data / "fits" / "decam-exposures-20251211.fits"

    @property
    def init_dirs(self) -> list[Path]:
        """Directories created by workspace-init."""
        return [self.root, self.configs, self.experiments, self.deployable_models,
                self.train_data, self.test_suite]


def workspace() -> WorkspacePaths:
    """Layout of the active workspace, resolved on each call.

    Returns
    -------
    WorkspacePaths
        Layout rooted at get_workspace_dir().
    """
    return WorkspacePaths(get_workspace_dir())


# ------------------------------------------------------------------ #
# Data directory layout                                              #
# ------------------------------------------------------------------ #

def lookups_dir(data_dir: Path) -> Path:
    """Lookup-table directory under a data directory.

    Parameters
    ----------
    data_dir : Path
        Data directory (e.g. workspace().des_data).

    Returns
    -------
    Path
        <data_dir>/lookups.
    """
    return Path(data_dir) / "lookups"


def feature_cache_dir(data_dir: Path, nside: int, is_azel: bool) -> Path:
    """Raw feature cache directory for one HEALPix resolution and coordinate frame.

    Parameters
    ----------
    data_dir : Path
        Data directory (e.g. workspace().des_data).
    nside : int
        HEALPix nside of the action bins.
    is_azel : bool
        Whether the bins are in az/el (else ra/dec).

    Returns
    -------
    Path
        <data_dir>/feature_cache_nside<nside>_<azel|radec>.
    """
    coord = "azel" if is_azel else "radec"
    return Path(data_dir) / f"feature_cache_nside{nside}_{coord}"


def field_feature_cache_dir(data_dir: Path) -> Path:
    """Field-level feature cache directory (field_filter runs); independent of any HEALPix grid.

    Parameters
    ----------
    data_dir : Path
        Data directory (e.g. workspace().des_data).

    Returns
    -------
    Path
        <data_dir>/feature_cache_field.
    """
    return Path(data_dir) / "feature_cache_field"


# ------------------------------------------------------------------ #
# Package resources                                                  #
# ------------------------------------------------------------------ #

PACKAGE_DIR = Path(__file__).resolve().parents[1]
LIVE_SCHEDULER_DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "live_scheduler_default.yaml"
DECAM_SKY_CONFIG = PACKAGE_DIR / "blanco" / "decam_sky.conf"
CONFIG_TEMPLATES_DIR = PACKAGE_DIR / "configs" / "templates"


def config_template(algorithm: str) -> Path:
    """Path to the packaged config template for one algorithm.

    Parameters
    ----------
    algorithm : str
        Algorithm key: 'bc', 'ddqn', 'cql', or 'iql'.

    Returns
    -------
    Path
        Path to <algorithm>.yaml in blancops/configs/templates.
    """
    path = CONFIG_TEMPLATES_DIR / f"{algorithm}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in CONFIG_TEMPLATES_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No config template for '{algorithm}'; available: {available}")
    return path


# ------------------------------------------------------------------ #
# Training-run directory layout                                      #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class RunPaths:
    """Directory layout of one training run (<parent_dir>/<experiment_name>/run_<timestamp>).

    Parameters
    ----------
    root : Path
        Run directory.
    """

    root: Path

    RESOLVED_CONFIG = "resolved_config.yaml"
    MODEL_PT = "model.pt"
    LATEST_CHECKPOINT = "latest_checkpoint.pt"
    CHECKPOINT_HISTORY = "checkpoint_history.json"
    PERIODIC_CHECKPOINT = "epoch_{epoch:03d}.pt"

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    @classmethod
    def from_config(cls, cfg) -> RunPaths:
        """Run directory of a loaded ExperimentConfig.

        Uses the directory the config was loaded from (its parent when that is configs/), else cfg.outdir.

        Parameters
        ----------
        cfg : ExperimentConfig
            Loaded experiment config.

        Returns
        -------
        RunPaths
            Layout of the run the config belongs to.
        """
        if cfg.orig_cfg_path:
            cfg_dir = Path(cfg.orig_cfg_path).parent
            return cls(cfg_dir.parent if cfg_dir.name == "configs" else cfg_dir)
        return cls(Path(cfg.outdir))

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    def periodic_checkpoint(self, epoch: int) -> Path:
        """Weights saved on the every-N-epochs schedule, independent of the checkpoint metric.

        Parameters
        ----------
        epoch : int
            Training epoch (1-based).

        Returns
        -------
        Path
            checkpoints/epoch_<epoch>.pt
        """
        return self.checkpoints / self.PERIODIC_CHECKPOINT.format(epoch=epoch)

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def resolved_config(self) -> Path:
        return self.configs / self.RESOLVED_CONFIG

    @property
    def split_json(self) -> Path:
        return self.configs / "split.json"

    @property
    def norm_stats_json(self) -> Path:
        return self.checkpoints / "normalization_stats.json"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoints / self.LATEST_CHECKPOINT

    @property
    def model_pt(self) -> Path:
        return self.checkpoints / self.MODEL_PT

    @property
    def checkpoint_history(self) -> Path:
        return self.checkpoints / self.CHECKPOINT_HISTORY

    @property
    def train_metrics(self) -> Path:
        return self.metrics / "train_metrics.pkl"

    @property
    def val_metrics(self) -> Path:
        return self.metrics / "val_metrics.pkl"

    def dataset_cache(self, split: str = "val") -> Path:
        """Cached normalized dataset for one split: checkpoints/<split>_dataset_cache.pt."""
        return self.checkpoints / f"{split}_dataset_cache.pt"

    def eval_dir(self, split: str = "val") -> Path:
        """Evaluation output directory: holdout_eval for val, else <split>_eval."""
        return self.root / ("holdout_eval" if split == "val" else f"{split}_eval")

    def make_dirs(self) -> None:
        """Create the run's standard sub-directories."""
        for d in (self.figures, self.checkpoints, self.metrics, self.configs, self.logs):
            d.mkdir(parents=True, exist_ok=True)
