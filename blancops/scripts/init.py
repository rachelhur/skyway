import argparse
from pathlib import Path
# import importlib.resources as pkg_resources
from importlib import resources

from blancops.configs.paths import PROFILE_POINTER_FILE, WorkspacePaths, get_workspace_dir

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    """
    Initialize a blancops workspace and save a pointer to ~/.blancops_profile. Defaults to the active workspace ($BLANCOPS_WORKSPACE, else the current pointer, else ~/.blancops).
    """
    parser = argparse.ArgumentParser(description="Initialize blancops workspace and saves a pointer to ~/.blancops_profile")
    parser.add_argument(
        '--workspace',
        '-w',
        type=Path, 
        default=get_workspace_dir(),
        help="Target directory to initialize the workspace. Defaults to the active workspace."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Overwrite existing configuration files if they already exist."
    )
    
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    logger.info(f"Initializing workspace at: {workspace}")

    # Create the necessary directory structure
    directories_to_create = WorkspacePaths(workspace).init_dirs
    
    for dir_path in directories_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  [+] Created directory: {dir_path}")

    # save workspace pointer file
    PROFILE_POINTER_FILE.write_text(str(workspace))
    logger.info(f"  [+] Saved workspace pointer to {PROFILE_POINTER_FILE}")

    logger.info("\nInitialization complete!")

if __name__ == "__main__":
    main()
