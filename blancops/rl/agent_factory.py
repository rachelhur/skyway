import json
from typing import Tuple
import torch
import yaml
from pathlib import Path

# Import your domain-specific modules
from blancops.configs.paths import RunPaths, workspace
from blancops.configs.enums import Algorithm
from blancops.configs.experiment_schema import ExperimentConfig, load_and_validate
from blancops.data.norm_stats import NormStats
from blancops.rl.registry import _build_bc_policy, _build_q_adapter, build_network
from blancops.rl.agent import Agent
from blancops.rl.checkpointer import resolve_weights_path

import logging
logger = logging.getLogger(__name__)

from typing import Tuple

class AgentFactory:
    def __init__(self, base_model_dir: str | Path | None = None):
        """Factory for building scheduling agents.

        Args:
            base_model_dir (str, optional): _description_. Defaults to workspace().deployable_models.
        """
        self.base_dir = Path(base_model_dir) if base_model_dir is not None else workspace().deployable_models
        self.alias_file = self.base_dir / "aliases.yml"
        self.aliases = self._load_aliases()

    def build_agent(
        self,
        model_path_or_alias: str,
        lookups: Path,
        field_choice_method: str,
        device: str = 'cpu',
        weights_filename: str = None, # Now defaults to None for auto-detection
        action_decode: str = 'joint'
    ) -> Tuple[Agent, ExperimentConfig, NormStats | None]:

        # If model_path_or_alias is an absolute path or a path that exists, prefer that
        if isinstance(model_path_or_alias, str) and Path(model_path_or_alias).is_absolute() and Path(model_path_or_alias).exists():
            model_dir = Path(model_path_or_alias)
        elif isinstance(model_path_or_alias, Path) and model_path_or_alias.exists():
            model_dir = model_path_or_alias
        else:
            model_dir = self.resolve_model_dir(model_path_or_alias)

        config_path = model_dir / RunPaths.RESOLVED_CONFIG
        if not config_path.exists():
            config_path = RunPaths(model_dir).resolved_config

        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find resolved_config.yaml in {model_dir} or {model_dir}/configs/"
            )

        cfg = load_and_validate(config_path)

        # 1. Resolve which weights file to actually use
        weights_path = self._resolve_weights_path(model_dir, weights_filename)

        # 2. Load the policy
        loaded_policy, norm_stats = self.load_policy(weights_path, cfg, device)

        agent = Agent(
            policy=loaded_policy,
            cfg=cfg,
            lookups=lookups,
            field_choice_method=field_choice_method,
            action_decode=action_decode
        )

        return agent, cfg, norm_stats

    def _resolve_weights_path(self, model_dir: Path, filename: str = None) -> Path:
        """Resolve the weights file via the shared, machine-portable resolver."""
        return resolve_weights_path(model_dir, filename)

    @staticmethod
    def load_policy(weights_path: Path, cfg: ExperimentConfig, device: str) -> Tuple[torch.nn.Module, NormStats | None]:
        core_net = build_network(cfg)

        if cfg.model.algorithm == Algorithm.BC:
            policy = _build_bc_policy(cfg, core_net)
        elif cfg.model.algorithm in (Algorithm.DDQN, Algorithm.CQL, Algorithm.IQL):
            # For IQL, algorithm.policy is the policy_net (QFlatPolicy), not the Q-adapter.
            policy = _build_q_adapter(cfg, core_net)

        try:
            checkpoint = torch.load(weights_path, map_location=device)
        except Exception as e:
            logger.warning(f"torch.load failed with default settings when loading policy: {e}. Retrying with weights_only=False.")
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and any(
            k in checkpoint for k in ('model_state_dict', 'policy_state_dict', 'state_dict')
        ):
            # Support multiple checkpoint key names.
            for key in ('model_state_dict', 'policy_state_dict', 'state_dict'):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            norm_stats = NormStats.from_dict(checkpoint['norm_stats']) if checkpoint.get('norm_stats') else None
        else:
            # Raw state dict.
            state_dict = checkpoint
            norm_stats = None

        AgentFactory._load_state_dict_tolerant(policy, state_dict, weights_path)

        policy.eval()
        return policy.to(device), norm_stats

    @staticmethod
    def _load_state_dict_tolerant(policy: torch.nn.Module, state_dict: dict, weights_path: Path):
        """Load a state dict, tolerating a leading 'policy.' key prefix mismatch.

        Deployment artifacts strip the 'policy.' wrapper prefix while some training
        checkpoints keep it; retry once with the prefix removed before failing.
        """
        try:
            policy.load_state_dict(state_dict)
            return
        except RuntimeError:
            stripped = {
                (k[len("policy."):] if k.startswith("policy.") else k): v
                for k, v in state_dict.items()
            }
            try:
                policy.load_state_dict(stripped)
                return
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to load weights from {weights_path}: state dict keys do "
                    f"not match the policy architecture. {e}"
                ) from e

    def _load_aliases(self) -> dict:
        if self.alias_file.exists():
            with open(self.alias_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def resolve_model_dir(self, model_path_or_alias: str) -> Path:
        target_directory_name = self.aliases.get(model_path_or_alias, model_path_or_alias)
        model_path = self.base_dir / target_directory_name

        if not model_path.is_dir():
            raise FileNotFoundError(f"Resolved model directory not found: {model_path}")
        return model_path
