import copy
import logging

from blancops.rl.algorithms.iql import IQL
logger = logging.getLogger(__name__)

import torch
from torch import nn

from blancops.configs.experiment_schema import ExperimentConfig
from blancops.configs.enums import (
    _AUTOREGRESSIVE_NETWORKS, Algorithm, ActionArchitecture, Network, ActionSpace,
    is_autoregressive, grid_is_azel, is_field_level
)
from blancops.rl.neural_nets.neural_nets import (
    ContextualScoreMLP,
    MLP,
    StateValueMLP,
    AutoregressiveNet,
    DualStreamMLP,
)
from blancops.rl.algorithms.bc import BehaviorCloning
from blancops.rl.algorithms.ddqn import DDQN, calculate_distance_matrix
from blancops.rl.algorithms.cql import CQL
from blancops.rl.policies.loss_function import FilterFocalLoss, FocalLoss, SlewDistanceFocalLoss
from blancops.rl.policies.q_policies import (
    QFlatPolicy,
    QAutoregressivePolicy,
)
from blancops.rl.policies.bc_policies import (
    BCAutoregressivePolicy,
    BCPureJointPolicy,
    BCHybridMarginalPolicy,
    BCPseudoAutoregressivePolicy,
)

# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #

BC_LOSS_STRAT_REGISTRY = {
    ActionArchitecture.PURE_JOINT: BCPureJointPolicy,
    ActionArchitecture.AUTOREGRESSIVE: BCAutoregressivePolicy,
    ActionArchitecture.HYBRID_MARGINAL: BCHybridMarginalPolicy,
    ActionArchitecture.PSEUDO_AUTOREGRESSIVE: BCPseudoAutoregressivePolicy,
}

# DQN, DDQN, CQL all share the same algorithm class; flags differentiate them.
ALGORITHM_REGISTRY = {
    Algorithm.BC: BehaviorCloning,
    Algorithm.DQN: DDQN,
    Algorithm.DDQN: DDQN,
    Algorithm.CQL: CQL,
    Algorithm.IQL: IQL,
}

NETWORK_REGISTRY = {
    Network.CONTEXTUAL_SCORE_MLP: ContextualScoreMLP,
    Network.MLP: MLP,
    Network.AUTOREGRESSIVE: AutoregressiveNet,
    Network.DUAL_STREAM_MLP: DualStreamMLP,
}

_ACTIVATION_REGISTRY = {
    'relu': nn.ReLU,
    'mish': nn.Mish,
    'swish': nn.SiLU,
    'gelu': nn.GELU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
}

_Q_VALUE_ALGORITHMS = {Algorithm.DQN, Algorithm.DDQN, Algorithm.CQL}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def get_activation(name: str):
    key = name.lower()
    if key not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Activation {name} not supported.")
    return _ACTIVATION_REGISTRY[key]


def get_loss_function(name: str, reduction='mean', gamma_focal=2.0, alpha=None):
    loss_map = {
        'cross_entropy':     lambda: nn.CrossEntropyLoss(reduction=reduction),
        'huber':             lambda: nn.HuberLoss(reduction=reduction),
        'mse':               lambda: nn.MSELoss(reduction=reduction),
        'focal_loss':        lambda: FocalLoss(gamma=gamma_focal, reduction=reduction, alpha=alpha),
        'focal_loss_filter': lambda: FilterFocalLoss(gamma=gamma_focal, reduction=reduction),
        'focal_loss_slew':   lambda: SlewDistanceFocalLoss(gamma=gamma_focal, reduction=reduction),
    }
    if name not in loss_map:
        raise NotImplementedError(f"Loss function '{name}' is not registered.")
    return loss_map[name]()


def _maybe_load_pretrained(net: nn.Module, cfg: ExperimentConfig, device: torch.device):
    path = getattr(cfg.metadata, 'pretrained_model_path', None)
    if path:
        net.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"Loaded pretrained model from {path}")


# --------------------------------------------------------------------------- #
# Network builder
# --------------------------------------------------------------------------- #

def build_network(cfg: ExperimentConfig) -> nn.Module:
    activation_fn = get_activation(cfg.model.activation)
    network_class = NETWORK_REGISTRY.get(cfg.model.network)
    if network_class is None:
        raise NotImplementedError(f"Network {cfg.model.network} not implemented.")

    if cfg.model.network == Network.CONTEXTUAL_SCORE_MLP:
        return network_class(
            global_dim=cfg.data.state_dim,
            cand_feat_dim=cfg.data.candidate_state_dim,
            score_dim=cfg.data.num_filters,
            hidden_dim=cfg.model.hidden_dim,
            nlayers=cfg.model.nlayers,
            global_enc_dim=cfg.model.global_enc_dim,
            layernorm=cfg.model.layernorm,
            activation=activation_fn,
            use_contextual_gating=cfg.model.contextual_gating,
        )

    if cfg.model.network == Network.DUAL_STREAM_MLP:
        layer_norm = cfg.model.algorithm != Algorithm.BC
        return network_class(
            global_dim=cfg.data.state_dim,
            cand_feat_dim=cfg.data.candidate_state_dim,
            hidden_dim=cfg.model.hidden_dim,
            score_dim=cfg.data.num_filters,
            activation=activation_fn,
            use_contextual_gating=False,
            use_layer_norm=layer_norm)

    if cfg.model.network == Network.MLP:
        # Add kwargs if/when a flat-MLP path is needed.
        return network_class()

    raise NotImplementedError(f"Network {cfg.model.network} has no builder branch.")


# --------------------------------------------------------------------------- #
# BC policy/strategy construction
# --------------------------------------------------------------------------- #

def _build_bc_policy(cfg: ExperimentConfig, core_net: nn.Module):
    # Filter-only action space short-circuits all strategy logic.
    if cfg.data.action_space == 'filter':
        loss_function = get_loss_function(
            cfg.model.loss_function,
            reduction=cfg.model.reduction,
            gamma_focal=cfg.model.gamma_focal,
            alpha=cfg.model.alpha,
        )
        return BCPureJointPolicy(core_net, loss_function, cfg.data.num_filters)

    if is_autoregressive(cfg.model.network):
        if cfg.model.loss_strategy != ActionArchitecture.AUTOREGRESSIVE:
            raise ValueError(
                f"Network {cfg.model.network} requires "
                f"LossStrategy.AUTOREGRESSIVE, got {cfg.model.loss_strategy}"
            )
        return BCAutoregressivePolicy(core_net, cfg.data.num_filters)

    # Simultaneous architecture: pick strategy.
    strategy_class = BC_LOSS_STRAT_REGISTRY.get(cfg.model.loss_strategy)
    if strategy_class is None:
        raise NotImplementedError(
            f"`{cfg.model.loss_strategy}` strategy not implemented for BC."
        )

    primary_loss = get_loss_function(
        cfg.model.loss_function,
        reduction=cfg.model.reduction,
        gamma_focal=cfg.model.gamma_focal,
        alpha=cfg.model.alpha,
    )

    if cfg.model.loss_strategy == ActionArchitecture.PURE_JOINT:
        return strategy_class(core_net, primary_loss, cfg.data.num_filters)

    if cfg.model.loss_strategy == ActionArchitecture.PSEUDO_AUTOREGRESSIVE:
        return strategy_class(
            core_net=core_net,
            num_filters=cfg.data.num_filters,
            filter_penalty=cfg.model.filter_penalty,
        )

    if cfg.model.loss_strategy == ActionArchitecture.HYBRID_MARGINAL:
        ce_loss = nn.CrossEntropyLoss(reduction=cfg.model.reduction)
        # Joint head can use focal loss; candidate/filter marginals stay CE for stability.
        joint_loss = (
            get_loss_function('focal_loss', gamma_focal=cfg.model.gamma_focal, alpha=None)
            if cfg.model.loss_function == 'focal_loss'
            else ce_loss
        )
        return strategy_class(
            core_net=core_net,
            num_filters=cfg.data.num_filters,
            candidate_loss_function=ce_loss,
            filter_loss_function=primary_loss,
            joint_loss_function=joint_loss,
            alpha_candidate=cfg.model.alpha_candidate,
            beta_filter=cfg.model.beta_filter,
            zeta_joint=cfg.model.zeta_joint,
        )

    raise NotImplementedError(f"`{cfg.model.loss_strategy}` has no construction branch.")


# --------------------------------------------------------------------------- #
# Q-net adapter selection
# --------------------------------------------------------------------------- #

def _build_q_adapter(cfg: ExperimentConfig, net: nn.Module):
    if cfg.model.network in _AUTOREGRESSIVE_NETWORKS:
        return QAutoregressivePolicy(net, cfg.data.num_filters)
    return QFlatPolicy(net, cfg.data.num_filters)


# --------------------------------------------------------------------------- #
# Top-level algorithm builder
# --------------------------------------------------------------------------- #

def build_algorithm(cfg: ExperimentConfig, device: torch.device):
    algo_class = ALGORITHM_REGISTRY.get(cfg.model.algorithm)
    if algo_class is None:
        raise ValueError(f"Algorithm {cfg.model.algorithm} unknown.")

    core_net = build_network(cfg).to(device)
    # _maybe_load_pretrained(core_net, cfg, device)

    optimizer = torch.optim.Adam(core_net.parameters(), lr=cfg.train.lr_init)

    if cfg.model.algorithm == Algorithm.BC:
        policy = _build_bc_policy(cfg, core_net)
        return BehaviorCloning(
            policy=policy,
            optimizer=optimizer,
            lr_scheduler=cfg.train.lr_scheduler,
            lr_scheduler_kwargs=cfg.train.lr_scheduler_kwargs,
            lr_scheduler_epoch_start=cfg.train.lr_sched_epoch_start,
            lr_scheduler_num_epochs=cfg.train.lr_sched_epoch_duration,
            device=device,
        )

    if cfg.model.algorithm in _Q_VALUE_ALGORITHMS:
        target_net = copy.deepcopy(core_net).to(device)
        policy = _build_q_adapter(cfg, core_net)
        target = _build_q_adapter(cfg, target_net)

        loss_function = get_loss_function(cfg.model.loss_function)
        use_double = cfg.model.algorithm in (Algorithm.DDQN, Algorithm.CQL)
        if cfg.model.algorithm in (Algorithm.DQN, Algorithm.DDQN):
            logger.info(f"Loading DDQN algorithm with gamma={cfg.model.gamma}, tau={cfg.model.tau}")
            return DDQN(
                policy=policy,
                target=target,
                optimizer=optimizer,
                gamma=cfg.model.gamma,
                tau=cfg.model.tau,
                loss_function=loss_function,
                use_double=use_double,
                lr_scheduler=cfg.train.lr_scheduler,
                lr_scheduler_kwargs=cfg.train.lr_scheduler_kwargs,
                lr_scheduler_epoch_start=cfg.train.lr_sched_epoch_start,
                lr_scheduler_num_epochs=cfg.train.lr_sched_epoch_duration,
                device=device,
            )
        elif cfg.model.algorithm == Algorithm.CQL:
            logger.info(f"Loading CQL algorithm with gamma={cfg.model.gamma}, tau={cfg.model.tau}")
            # CQL-specific scaling.
            dist_matrix = None
            dist_scaling_factor = 0.0
            if not is_field_level(cfg.data.action_space):
                dist_matrix = calculate_distance_matrix(
                    nside=cfg.data.nside,
                    is_azel=grid_is_azel(cfg.data.action_space),
                )
            q_max = 1.0 / (1.0 - cfg.model.gamma)
            dist_scaling_factor = q_max / torch.pi
            return CQL(
                policy=policy,
                target=target,
                optimizer=optimizer,
                gamma=cfg.model.gamma,
                tau=cfg.model.tau,
                loss_function=loss_function,
                use_double=use_double,
                cql_alpha=cfg.model.cql_alpha,
                cql_margin=cfg.model.cql_margin,
                dist_matrix=dist_matrix,
                dist_scaling_factor=dist_scaling_factor,
                lr_scheduler=cfg.train.lr_scheduler,
                lr_scheduler_kwargs=cfg.train.lr_scheduler_kwargs,
                lr_scheduler_epoch_start=cfg.train.lr_sched_epoch_start,
                lr_scheduler_num_epochs=cfg.train.lr_sched_epoch_duration,
                device=device,
            )

    if cfg.model.algorithm == Algorithm.IQL:
        logger.info(
            f"Loading IQL algorithm with expectile={cfg.model.expectile}, "
            f"awr_beta={cfg.model.awr_beta}, awr_clip={cfg.model.awr_clip}"
        )
        activation_fn = get_activation(cfg.model.activation)

        target_net = copy.deepcopy(core_net).to(device)
        policy_raw = build_network(cfg).to(device)   # separate net for the AWR policy
        v_net = StateValueMLP(
            glob_dim=cfg.data.state_dim,
            cand_dim=cfg.data.num_candidates * cfg.data.candidate_state_dim,
            hidden=(cfg.model.hidden_dim,) * cfg.model.nlayers,
            layernorm=cfg.model.layernorm,
            activation=activation_fn,
        ).to(device)

        q_adapter = _build_q_adapter(cfg, core_net)
        q_target_adapter = _build_q_adapter(cfg, target_net)
        policy_adapter = _build_q_adapter(cfg, policy_raw)  # QFlatPolicy for checkpointing

        iql_optimizer = torch.optim.Adam(
            [*q_adapter.parameters(), *v_net.parameters(), *policy_adapter.parameters()],
            lr=cfg.train.lr_init,
        )
        loss_function = get_loss_function('mse')  # TD regression

        return IQL(
            q_adapter=q_adapter,
            q_target_adapter=q_target_adapter,
            v_net=v_net,
            policy_net=policy_adapter,
            optimizer=iql_optimizer,
            loss_function=loss_function,
            gamma=cfg.model.gamma,
            tau=cfg.model.tau,
            expectile=cfg.model.expectile,
            awr_beta=cfg.model.awr_beta,
            awr_clip=cfg.model.awr_clip,
            lr_scheduler=cfg.train.lr_scheduler,
            lr_scheduler_kwargs=cfg.train.lr_scheduler_kwargs,
            lr_scheduler_epoch_start=cfg.train.lr_sched_epoch_start,
            lr_scheduler_num_epochs=cfg.train.lr_sched_epoch_duration,
            device=device,
        )

    raise ValueError(f"Algorithm {cfg.model.algorithm} has no construction branch.")
