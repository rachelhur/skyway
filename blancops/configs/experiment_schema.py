import datetime
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator, ValidationInfo
import yaml
from pathlib import Path
from typing import Any, List, Union, Literal, Dict
import numpy as np
from typing import Optional
from blancops.configs.enums import *
from blancops.configs.constants import _DEFAULT_NORM_MAPPING, _FILTER_DEP_FEATURE_NAMES, DES_FITS_PATH, _BIN_FEATURES
from blancops.configs.constants import FILTER2IDX
from blancops.configs.constants import _ALLOWED_NORMS_PER_FEATURE, _NORM_TYPES
from blancops.survey.profiles import DES
from blancops.data.splits import NightSplit

class ActionConstraints(BaseModel):
    sun_el_limit: float = DES.sun_el_limit
    airmass_limit: float = 3.0
    airmass_failsafe: float = 1.8

    @field_validator('sun_el_limit')
    @classmethod
    def validate_sun_el_limit(cls, v):
        if v == None:
            return DES.sun_el_limit
        if v >= 0:
            raise ValueError('sun_el_limit should be negative (degrees below horizon)')
        if v < -90:
            raise ValueError('sun_el_limit should be >= -90 degrees')
        return v

    @field_validator('airmass_limit', 'airmass_failsafe')
    @classmethod
    def validate_airmass_limit(cls, v):

        if v <= 1.0:
            raise ValueError('airmass limit should be > 1.0 (minimum airmass at zenith)')
        if v > 10.0:
            raise ValueError('airmass limit should be <= 10.0 (extremely high airmass)')
        return v

class NormalizationConfig(BaseModel):
    # feature_names: List[str] = Field(
    #     default_factory=lambda: list(DEFAULT_NORM_MAPPING.keys())
    #     )
    feature_norm_mappings: Dict[str, List[_NORM_TYPES]] = Field(
        default_factory=lambda: {k: v.copy() for k, v in _DEFAULT_NORM_MAPPING.items()})
    fix_nans: bool = True

    @field_validator('feature_norm_mappings', mode='before')
    @classmethod
    def merge_with_defaults(cls, v: Any) -> Any:
        # If the input is a dictionary, merge it with the defaults
        if isinstance(v, dict):
            # 1. Start with a fresh copy of the default mappings
            merged = {k: val.copy() for k, val in _DEFAULT_NORM_MAPPING.items()}

            # 2. Overwrite only the keys the user explicitly provided in the config
            merged.update(v)

            # 3. Return the merged dictionary for Pydantic to validate
            return merged
        return v

    @model_validator(mode='after')
    def validate_legal_normalizations(self) -> 'NormalizationConfig':
        for feature, requested_norms in self.feature_norm_mappings.items():
            allowed_norms = _ALLOWED_NORMS_PER_FEATURE.get(feature)
            if allowed_norms is None:
                raise ValueError(f"Feature '{feature}' is not recognized in allowed normalization rules.")

            if not set(requested_norms).issubset(allowed_norms): # allowed_norms is already a set!
                raise ValueError(
                    f"Normalization {requested_norms} is not allowed for feature '{feature}'. "
                    f"Allowed: {list(allowed_norms)}"
                )
        return self

class SeeingConfig(BaseModel):
    """Parameters for the Seeing rolling-history predictor.

    Shared by the offline causal-fwhm feature builder, the historic
    validation env, and the live env so train and serve agree. Instrument
    components are in arcsec and converted to native angle units when a
    Seeing instance is built.
    """
    window: str = "15m"
    retention_window: Optional[str] = None
    from_instrument: float = 0.5
    to_instrument: float = 0.5


class BaseDataConfig(BaseModel):
    name: str = 'des-data-v0'
    path: str = str(DES_FITS_PATH)
    # cache_in_memory: bool = False

    # Data configuration
    nside: int = 16
    action_space: str

    # Normalization configuration
    norm: NormalizationConfig = Field(default_factory=NormalizationConfig)

    # Seeing predictor configuration
    seeing: SeeingConfig = Field(default_factory=SeeingConfig)

    # Configurations calculated after data processing (required for model instantiation)
    state_dim: Optional[int] = None
    bin_state_dim: Optional[int] = None
    num_bins: Optional[int] = None
    num_filters: Optional[int] = None
    num_actions: Optional[int] = None

    # Features
    global_features: List[str]
    bin_features: List[str]

    @model_validator(mode='after')
    def validate_features(self) -> 'TrainDataConfig':
        for bin_feat in self.bin_features:
            if bin_feat not in _BIN_FEATURES:
                raise ValueError(f"{bin_feat} is not implemented.")
        return self

    @model_validator(mode='after')
    def validate_action_space_consistency(self) -> 'TrainDataConfig':
        # Validate that action_space is consistent with features
        has_filter = 'filter' in self.action_space
        has_radec = 'radec' in self.action_space
        has_azel = 'azel' in self.action_space

        # Check for invalid combinations
        if has_radec and has_azel:
            raise ValueError("action_space cannot contain both 'radec' and 'azel'")

        # Check that filter features are only included when filter action space is used
        if has_filter:
            filter_features = [f for f in self.global_features + self.bin_features
                             if any(filter_str in f for filter_str in ['filter', 'urgency'])]
            if not filter_features:
                # This is just a warning, not an error - we'll allow it but note it
                pass
        else:
            # If no filter in action space, we shouldn't have filter-specific features
            filter_features = [f for f in self.global_features + self.bin_features
                             if any(filter_str in f for filter_str in _FILTER_DEP_FEATURE_NAMES)]
            if filter_features:
                raise ValueError(f"Filter-specific features {filter_features} found but action_space '{self.action_space}' does not include 'filter'")

        return self

class TrainDataConfig(BaseDataConfig):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    years: List[int] = [2013, 2014, 2015, 2016, 2017, 2018, 2019] # full set of data
    months: List[int] = [i+1 for i in range(12)]
    days: List[int] = [i+1 for i in range(31)]
    filters: List[str] = [filt for filt in FILTER2IDX.keys()]

    # Split specification: each of val and test is either an explicit night
    # list or a fraction of the total night count. Explicit lists win.
    val_nights: Optional[List[str]] = None
    val_frac: Optional[float] = None
    test_nights: Optional[List[str]] = None
    test_frac: Optional[float] = None
    train_val_split: float = 0.9  # deprecated; feeds val_frac when val_frac is None

    @field_validator('train_val_split')
    @classmethod
    def validate_train_val_split(cls, v):
        if not 0 < v < 1:
            raise ValueError('train_val_split must be between 0 and 1 exclusive')
        return v

    @field_validator('val_frac', 'test_frac')
    @classmethod
    def validate_split_frac(cls, v):
        if v is not None and not 0 < v < 1:
            raise ValueError('split fractions must be between 0 and 1 exclusive')
        return v

    @model_validator(mode='after')
    def validate_split_fracs_sum(self) -> 'TrainDataConfig':
        total = (self.val_frac or 0.0) + (self.test_frac or 0.0)
        if total >= 1:
            raise ValueError(f'val_frac + test_frac must be < 1, got {total}')
        return self

    @property
    def effective_val_frac(self) -> Optional[float]:
        """Validation fraction actually used, falling back to the deprecated
        ``train_val_split`` when neither ``val_nights`` nor ``val_frac`` is set.

        Returns:
            The fraction, or None when explicit val nights are supplied.
        """
        if self.val_nights:
            return None
        if self.val_frac is not None:
            return self.val_frac
        return 1.0 - self.train_val_split

    @field_validator('years', 'months', 'days', 'filters')
    @classmethod
    def validate_not_empty(cls, v):
        if not v:
            raise ValueError('List cannot be empty')
        return v

class BaseAlgConfig(BaseModel):
    network: Network = Network.CONTEXTUAL_SCORE_MLP
    loss_strategy: ActionArchitecture = ActionArchitecture.PURE_JOINT
    hidden_dim: int = 128
    nlayers: int = 4
    loss_function: str
    contextual_gating: bool = False
    activation: str = "relu"
    global_enc_dim: Optional[int] = 128
    layernorm: bool = True

    @field_validator('global_enc_dim')
    @classmethod
    def validate_global_enc_dim(cls, v):
        if v is not None and v <= 0:
            raise ValueError('global_enc_dim must be positive or None')
        return v

    @field_validator('hidden_dim')
    @classmethod
    def validate_hidden_dim(cls, v):
        if v <= 0:
            raise ValueError('hidden_dim must be positive')
        if v > 4096:
            raise ValueError('hidden_dim is unreasonably large (>4096)')
        return v

    @field_validator('nlayers')
    @classmethod
    def validate_nlayers(cls, v):
        if v <= 0:
            raise ValueError('nlayers must be positive')
        if v > 100:
            raise ValueError('nlayers is unreasonably large (>100)')
        return v

    @model_validator(mode='after')
    def validate_autoregressive_net(self) -> "BaseAlgConfig":
        if is_autoregressive(self.network):
            if self.loss_strategy != ActionArchitecture.AUTOREGRESSIVE:
                raise ValueError(
                    f"Network {self.network} requires "
                    f"LossStrategy.AUTOREGRESSIVE, got {self.loss_strategy}"
                )
        return self

    @model_validator(mode='after')
    def validate_activation(self) -> "BaseAlgConfig":
        # Import here to avoid circular import
        from blancops.rl.registry import _ACTIVATION_REGISTRY
        valid_activations = _ACTIVATION_REGISTRY.keys()
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got {self.activation}")
        return self

class RewardWeights(BaseModel):
    w_slew: float = 1.0
    w_airmass: float = 1.0
    w_t_last_visit: float = 1.0
    w_min_tiling: float = 1.0
    airmass_limit: float = 3.0
    t_ref_seconds: float = 60*60*12

class BCAlgConfig(BaseAlgConfig):
    algorithm: Literal[Algorithm.BC]

    # Loss function knobs (used by some strategies, ignored by others)
    reduction: str = "mean"
    gamma_focal: float = 2.0
    alpha: Optional[float] = None

    # Pseudo-autoregressive knobs
    filter_penalty: float | None = None

    # Hybrid-marginal weights
    alpha_bin: float | None = None
    beta_filter: float | None = None
    zeta_joint: float | None = None
    reward: RewardStructure | None = None
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)

    @model_validator(mode="after")
    def validate_strategy_requirements(self) -> "BCAlgConfig":
        # Optional: enforce that focal-loss configs explicitly set gamma_focal,
        # etc. Pydantic will already have applied defaults, so this is for
        # cross-field constraints only.
        if self.loss_function == "focal_loss" and self.alpha is not None:
            if not 0.0 <= self.alpha <= 1.0:
                raise ValueError("alpha must be in [0, 1] for focal_loss")
        return self

    @field_validator('gamma_focal')
    @classmethod
    def validate_gamma_focal(cls, v):
        if v <= 0:
            raise ValueError('gamma_focal must be positive')
        return v

    @field_validator('filter_penalty', 'alpha_bin', 'beta_filter', 'zeta_joint')
    @classmethod
    def validate_optional_float(cls, v):
        if v is not None and v < 0:
            raise ValueError('Value must be non-negative')
        return v

class DDQNAlgConfig(BaseAlgConfig):
    algorithm: Literal[Algorithm.DDQN]
    reward: RewardStructure = RewardStructure.TEFF
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)
    reward_norm: str = 'minmax'
    tau: float = 0.005 # DDQN specific parameter
    gamma: float = 0.99 # DDQN specific parameter

    @field_validator('tau')
    @classmethod
    def validate_tau(cls, v):
        if not 0 < v <= 1:
            raise ValueError('tau must be between 0 and 1 exclusive')
        return v

    @field_validator('gamma')
    @classmethod
    def validate_gamma(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('gamma must be between 0 and 1 inclusive')
        return v

    @model_validator(mode="after")
    def validate_reward(self) -> "DDQNAlgConfig":
        assert self.reward in RewardStructure, f"Reward structure {self.reward} is not supported."
        return self

class CQLAlgConfig(DDQNAlgConfig):
    algorithm: Literal[Algorithm.CQL]
    cql_alpha: float = 1.0
    cql_margin: float = 0.0

    @field_validator('cql_alpha')
    @classmethod
    def validate_cql_alpha(cls, v):
        if v <= 0:
            raise ValueError('cql_alpha must be positive')
        return v

class IQLAlgConfig(DDQNAlgConfig):
    algorithm: Literal[Algorithm.IQL]
    expectile: float = 0.7
    awr_beta: float = 3.0
    awr_clip: float = 100.0

    @field_validator('expectile')
    @classmethod
    def validate_expectile(cls, v):
        if not 0 < v < 1:
            raise ValueError('expectile must be between 0 and 1 exclusive')
        return v

    @field_validator('awr_beta')
    @classmethod
    def validate_awr_beta(cls, v):
        if v <= 0:
            raise ValueError('awr_beta must be positive')
        return v

    @field_validator('awr_clip')
    @classmethod
    def validate_awr_clip(cls, v):
        if v <= 0:
            raise ValueError('awr_clip must be positive')
        return v

AnyModelConfig = Union[BCAlgConfig, DDQNAlgConfig, CQLAlgConfig, IQLAlgConfig]

class TrainConfig(BaseModel):
    checkpoint_metric: CheckpointMetric = CheckpointMetric.ANGULAR_SEPARATION
    max_epochs: int = 50
    batch_size: int   = 1024
    lr_scheduler: str = "cosine_annealing"
    num_workers: int = 0
    lr_init: float = .001
    lr_final: float = 1e-5
    lr_sched_epoch_start: int = 10
    lr_sched_epoch_duration: int = 30
    patience: int = 20
    device:         str   = "cuda"
    seed:           int   = 42

    lr_scheduler_kwargs: Optional[dict] = None

    @field_validator('max_epochs')
    @classmethod
    def validate_max_epochs(cls, v):
        if v <= 0:
            raise ValueError('max_epochs must be positive')
        if v > 1000:
            raise ValueError('max_epochs is unreasonably large (>1000)')
        return v

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError('batch_size must be positive')
        if v > int(10240):
            raise ValueError('batch_size is unreasonably large (>65536)')
        return v

    @field_validator('lr_init', 'lr_final')
    @classmethod
    def validate_learning_rate(cls, v):
        if v <= 0:
            raise ValueError('learning rate must be positive')
        if v > 1:
            raise ValueError('learning rate should be <= 1.0')
        return v

    @field_validator('lr_sched_epoch_start', 'lr_sched_epoch_duration')
    @classmethod
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v

    @field_validator('patience')
    @classmethod
    def validate_patience(cls, v):
        if v < 0:
            raise ValueError('patience must be non-negative')
        return v

    @field_validator('seed')
    @classmethod
    def validate_seed(cls, v):
        if v < 0:
            raise ValueError('seed must be non-negative')
        return v

    @model_validator(mode='after')
    def validate_lr_scheduler(self):
        # This will now run automatically when model_copy adds the kwargs
        if self.lr_scheduler_kwargs:
            assert self.max_epochs - self.lr_sched_epoch_start - self.lr_sched_epoch_duration >= 0, "The number of epochs must be greater than lr_scheduler_epoch_start + lr_scheduler_dur_epochs"
        return self

    # @model_validator(mode='after')
    # def validate_lr_scheduler_type(self):
    #     valid_schedulers = ['cosine_annealing', 'step', 'exponential', 'reduce_on_plateau', 'constant']
    #     if self.lr_scheduler not in valid_schedulers:
    #         raise ValueError(f"lr_scheduler must be one of {valid_schedulers}, got {self.lr_scheduler}")
    #     return self

class ExperimentConfig(BaseModel):
    experiment_name: str
    parent_dir: str = "experiments/"
    outdir: Optional[str] = None
    orig_cfg_path: Optional[str] = None
    data: TrainDataConfig
    model: AnyModelConfig = Field(discriminator="algorithm")
    train: TrainConfig
    device: str = 'cuda'

    @field_validator('experiment_name')
    @classmethod
    def validate_experiment_name(cls, v):
        if not v or not v.strip():
            raise ValueError('experiment_name cannot be empty')
        return v

    @field_validator('parent_dir')
    @classmethod
    def validate_parent_dir(cls, v):
        if not v:
            raise ValueError('parent_dir cannot be empty')
        return v

    @model_validator(mode='before')
    @classmethod
    def set_outdir(cls, data: Any) -> Any:
        """Intercepts the raw dictionary to compute outdir before validation."""
        if isinstance(data, dict) and data.get('outdir') is None:
            exp_name = data.get('experiment_name')
            parent = data.get('parent_dir', 'experiments/')

            if exp_name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                data['outdir'] = str(Path(parent) / exp_name / f"run_{timestamp}")
                # data['outdir'] = str(Path(parent) / exp_name)

        return data # Return the modified dictionary

    @model_validator(mode="after")
    def validate_algorithm_compatibility(self) -> "ExperimentConfig":
        if (
            self.model.algorithm == Algorithm.BC
            and self.data.action_space == ActionSpace.FILTER
            and self.model.loss_strategy != ActionArchitecture.PURE_JOINT
        ):
            raise ValueError(
                f"action_space=FILTER only supports loss_strategy=PURE_JOINT, "
                f"got {self.model.loss_strategy}"
            )
        return self

    @model_validator(mode='after')
    def validate_paths(self) -> "ExperimentConfig":
        # Validate that parent_dir is a reasonable path
        if not self.parent_dir or self.parent_dir.strip() == '':
            raise ValueError('parent_dir cannot be empty')
        return self

# ----------------------------------
# Experiment Config helpers
# ----------------------------------
def load_and_validate(yaml_path: str | Path) -> ExperimentConfig:
    """Loads the YAML config and validates it against the ExperimentConfig schema."""
    with open(yaml_path, "r") as f:
        raw_data = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw_data) # Added dictionary unpacking
    cfg.orig_cfg_path = str(Path(yaml_path).resolve())
    return cfg

def resolve_and_save(cfg: ExperimentConfig, dataset_dims: dict, dataset_feature_names: dict, lr_scheduler_kwargs: dict, night_split: NightSplit, outdir: str | Path) -> ExperimentConfig:
    """Resolves config by filling in fields calculated after data processing. Saves the resolved config to the output directory."""
    # UPDATE CONFIG.DATA
    data_updates = {
        "state_dim": int(dataset_dims['state_dim']),
        "bin_state_dim": int(dataset_dims['bin_state_dim']),
        "num_bins": int(dataset_dims['num_bins']),
        "num_filters": int(dataset_dims['num_filters']),
        "num_actions": int(dataset_dims['num_actions']),
        # "global_features": dataset_feature_names['global_features'],
        # "bin_features": dataset_feature_names['bin_features'],
        "val_nights": list(night_split.val),
        "test_nights": list(night_split.test),
    }
    updated_data = cfg.data.model_copy(update=data_updates)
    # UPDATE CONFIG.TRAIN
    train_updates = {"lr_scheduler_kwargs": lr_scheduler_kwargs}
    updated_train = cfg.train.model_copy(update=train_updates)

    # COPY CONFIG WITH UPDATES
    resolved_cfg = cfg.model_copy(update={
        "data": updated_data,
        "train": updated_train
    })

    # CONSTRUCT EXPERIMENT_OUTDIR CONFIG FIELD AND SAVE RESOLVED CONFIG
    if resolved_cfg.outdir is None:
        resolved_cfg.outdir = str(Path(resolved_cfg.outdir))
    Path(Path(resolved_cfg.outdir) / "configs" ).mkdir(parents=True, exist_ok=True)
    with open(Path(resolved_cfg.outdir) / "configs" /"resolved_config.yaml", "w") as f:
        # Use mode='json' to force Pydantic to convert complex types (like Enums) to strings
        resolved_dict = resolved_cfg.model_dump(mode='json')
        # print('DUMPING RESOLVED CONFIG IN ', f)
        yaml.dump(resolved_dict, f, sort_keys=False)

    return resolved_cfg
