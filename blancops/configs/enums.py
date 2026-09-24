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

_AUTOREGRESSIVE_NETWORKS = {Network.AUTOREGRESSIVE}

def is_autoregressive(network: Network) -> bool:
    return network in _AUTOREGRESSIVE_NETWORKS

class ActionSpace(str, Enum):
    AZEL_FILTER = 'azel_filter'
    RADEC_FILTER = 'radec_filter'
    FILTER = 'filter' # Not functional
    AZEL = 'azel'
    RADEC = 'radec'


class RewardTerm(str, Enum):
    TEFF = "teff"
    SLEW = "slew"
    EXPERT = "expert"
    UNIFORMITY = "uniformity"


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
