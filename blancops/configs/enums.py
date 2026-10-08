from enum import Enum
import operator

class Algorithm(str, Enum):
    BC = "bc"
    DQN = "dqn"
    DDQN = "ddqn"
    CQL = "cql"
    IQL = "iql"

class Network(str, Enum):
    MLP = "mlp"
    CONTEXTUAL_SCORE_MLP = "contextual_score_mlp"
    DUAL_STREAM_MLP = "dual_stream_mlp"
    AUTOREGRESSIVE = "autoregressive"

class ActionArchitecture(str, Enum):
    PURE_JOINT = "pure_joint"
    HYBRID_MARGINAL = "hybrid_marginal"
    PSEUDO_AUTOREGRESSIVE = "pseudo_ar"
    AUTOREGRESSIVE = "autoregressive"
    # MARGINAL = "marginal"

class CheckpointMetric(str, Enum):
    VAL_LOSS = "val_loss"
    ANGULAR_SEPARATION = "ang_sep"
    MAX_Q_POLICY = "q_policy"
    ACCURACY = "accuracy"

class LossFunction(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    FOCAL_LOSS = "focal_loss"
    FOCAL_LOSS_FILTER = "focal_loss_filter"
    FOCAL_LOSS_SLEW = "focal_loss_slew"
    HUBER = "huber"
    MSE = "mse"

_AUTOREGRESSIVE_NETWORKS = {Network.AUTOREGRESSIVE}

def is_autoregressive(network: Network) -> bool:
    return network in _AUTOREGRESSIVE_NETWORKS

class ActionSpace(str, Enum):
    AZEL_FILTER = 'azel_filter'
    RADEC_FILTER = 'radec_filter'
    FIELD_FILTER = 'field_filter'
    FILTER = 'filter' # Not functional
    AZEL = 'azel'
    RADEC = 'radec'


def is_field_level(action_space: str) -> bool:
    """Whether actions are (survey field, filter) pairs rather than (bin, filter) pairs."""
    return action_space == ActionSpace.FIELD_FILTER


def has_filter(action_space: str) -> bool:
    """Whether the filter is part of the action."""
    return action_space in (ActionSpace.AZEL_FILTER, ActionSpace.RADEC_FILTER,
                            ActionSpace.FIELD_FILTER, ActionSpace.FILTER)


def grid_is_azel(action_space: str) -> bool:
    """Whether the HEALPix grid and feature cache are az/el; field-level runs share the az/el global cache."""
    return action_space in (ActionSpace.AZEL_FILTER, ActionSpace.AZEL, ActionSpace.FIELD_FILTER)


class RewardTerm(str, Enum):
    TEFF = "teff"
    TEFF_ACCEPTED = "teff_accepted"  # accepted effective seconds, relative to nominal 90 s exposures
    SLEW = "slew"
    EXPERT = "expert"
    UNIFORMITY = "uniformity"


class AcceptanceRule(str, Enum):
    """Which exposures count toward the survey."""
    UNIFORM = "uniform_0.3"         # one 0.3 teff threshold for every band (legacy)
    DES_PER_BAND = "des_per_band"   # DES's per-band minimum teff (Morganson et al. 2018, Table 4)

    def require(self, wanted: "AcceptanceRule | str", source) -> None:
        """Refuse a requested rule that differs from this one, the rule ``source`` was built with.

        Parameters
        ----------
        wanted : AcceptanceRule or str
            Rule the caller asked for.
        source : Path or str
            Folder built with this rule, named in the error.

        Raises
        ------
        ValueError
            Naming both rules and the source.
        """
        wanted = AcceptanceRule(wanted)
        if wanted is not self:
            raise ValueError(f"{source} was built with acceptance '{self.value}', but '{wanted.value}' "
                             f"was asked for. Rebuild it or fix data.acceptance.")


class LookupKeys(str, Enum):
    """Convenient/consistent lookup table names."""

    FIELDS = "fields_table.json"
    TARGET_FIDFILT_COUNTS = "target_counts_per_fidfilt.pkl"
    FIDFILT_EXPTIME = "fidfilt_exptime.pkl"
    TARGET_FILT_COUNTS = "target_counts_per_filter.pkl"
    TARGET_FID_COUNTS = "target_counts_per_fid.pkl"

    # TRAIN DATA LOOKUP KEYS
    TARGET_FID2VISITS_TRAIN = "target_counts_per_fid_train.json"
    TARGET_FID2VISITS_EVAL = "target_counts_per_fid_eval.json"
    NIGHT2FID_VISIT_HIST = "night2fidvisits.pkl"
    NIGHT2FIDFILT_VISIT_HIST = "night2fidfilt_visits.pkl"
    NIGHT2FID_LAST_VISIT_TS = "night2fid_last_visit_ts.pkl"
    NIGHT2FIDFILT_LAST_VISIT_TS = "night2fidfilt_last_visit_ts.pkl"
    NIGHT2FID_LAST_VISIT_OT = "night2fid_last_visit_ot.pkl"
    NIGHT2FIDFILT_LAST_VISIT_OT = "night2fidfilt_last_visit_ot.pkl"
    NIGHT2OT_CLOCK_SECONDS = "night2observing_time_seconds.pkl"
    # TOTAL_OT_SECONDS = "total_observing_time_seconds.txt"
    HISTORIC_OBSERVATIONS = "historic_observations.json"
    ACCEPTANCE = "acceptance.json"  # acceptance rule the lookups were built with
