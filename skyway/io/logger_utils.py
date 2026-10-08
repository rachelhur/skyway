import logging
from pathlib import Path
import sys


def configure_logger(
    level: str = "debug",
    log_to_stdout: bool = True,
    log_to_file: bool = True,
    outdir: str | Path = None,
    filename: str = None,
    use_tqdm: bool = True,
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    parent_module: str = 'skyway',
) -> logging.Logger:
    """
    Configure logging for the skyway project.

    Policy enforced:
    - Stdout handler: INFO+
    - File handler: DEBUG+
    - tqdm-safe stdout (optional, default True)
    """

    logger = logging.getLogger(parent_module)
    logger.propagate = False

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logger.setLevel(level_map[level])

    # ---- Clear existing handlers (idempotent) ----
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    # ---- Formatter ----
    formatter = logging.Formatter(
        format,
        datefmt=datefmt,
    )

    # ---- Stdout handler (INFO+) ----
    if log_to_stdout:
        if use_tqdm:
            from tqdm import tqdm

            class TqdmHandler(logging.Handler):
                def emit(self, record):
                    
                    msg = self.format(record)
                    tqdm.write(msg)

            sh = TqdmHandler()
        else:
            sh = logging.StreamHandler(sys.stdout)

        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    # ---- File handler (DEBUG+) ----
    if log_to_file:
        assert outdir and filename, \
            "outdir and filename must be provided when log_to_file=True"

        Path(outdir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(outdir) / filename, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
