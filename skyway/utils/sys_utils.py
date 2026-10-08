import sys
import logging
from pathlib import Path
import importlib.resources as pkg_resources
import os

import numpy as np
import random
import torch
import yaml

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

# def load_model_config(config_path=None):
#     """Loads a custom config if provided, otherwise loads the default from the package."""
#     if config_path:
#         with open(config_path, 'r') as f:
#             return yaml.safe_load(f)
#     else:
#         # Load the default config bundled inside your package (e.g., skyway/global_config.json)
#         config_text = pkg_resources.files('skyway').joinpath('configs/default_model_config.yaml').read_text()
#         return yaml.safe_load(config_text)

# def save_config(args=None, config_dict=None, outdir=None):
#     """Saves the experiment arguments as YAML file."""
#     out_path = Path(outdir)
#     out_path.mkdir(parents=True, exist_ok=True)
    
#     # Convert argparse Namespace to nested dict
#     if args is not None:
#         config_dict = dict_to_nested(vars(args))
    
#     with open(out_path / "config.yaml", "w") as f:
#         yaml.dump(config_dict, f, indent=4)

# def dict_to_nested(data):
#     """Converts {'model.lr': 0.1} to {'model': {'lr': 0.1}}"""
#     nested = {}
#     for key, value in data.items():
#         keys = key.split('.')
#         d = nested
#         for k in keys[:-1]:
#             d = d.setdefault(k, {})
#         d[keys[-1]] = value
#     return nested

def get_system_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"   
    )
    return device
