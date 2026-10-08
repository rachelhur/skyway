"""Night-level train/val/test split resolution for the offline RL pipeline.

A split is resolved once per training run from the config and persisted as
JSON so evaluation can reproduce it without re-sampling.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_SPLIT_NAMES = ('train', 'val', 'test')


@dataclass(frozen=True)
class NightSplit:
    """Resolved partition of observing nights into train, val, and test.

    Args:
        train: Night strings assigned to the training set.
        val: Night strings assigned to the validation set.
        test: Night strings assigned to the test set.
        seed: Seed used for any random draws that produced this split.
    """

    train: List[str]
    val: List[str]
    test: List[str]
    seed: int

    def nights_for(self, split: str) -> List[str]:
        """Return the night list for a split name.

        Args:
            split: One of 'train', 'val', 'test'.

        Returns:
            The night strings assigned to that split.
        """
        if split not in _SPLIT_NAMES:
            raise ValueError(f"Unknown split '{split}'. Expected one of {_SPLIT_NAMES}.")
        return list(getattr(self, split))

    @property
    def counts(self) -> dict:
        return {name: len(getattr(self, name)) for name in _SPLIT_NAMES}

    def save(self, path: Path) -> None:
        """Write the split to JSON.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
        logger.info(f"Night split saved to {path} ({self.counts})")

    @classmethod
    def load(cls, path: Path) -> 'NightSplit':
        """Read a split from JSON.

        Args:
            path: Source file path.

        Returns:
            The deserialized NightSplit.
        """
        with open(Path(path)) as f:
            d = json.load(f)
        return cls(train=list(d['train']), val=list(d['val']),
                   test=list(d['test']), seed=int(d['seed']))

    @classmethod
    def exists(cls, path: Path) -> bool:
        return Path(path).exists()


def _draw(pool: List[str], n_total: int, frac: float, rng) -> List[str]:
    """Draw a fraction of the total night count from a pool without replacement.

    The count is taken from ``n_total`` (every night in the dataset), while the
    pool is only the nights still unassigned. This keeps splits disjoint without
    shrinking later splits when an earlier one is pinned to an explicit list.

    Args:
        pool: Nights still available for assignment.
        n_total: Total number of nights in the dataset.
        frac: Fraction of n_total to draw.
        rng: A numpy Generator, or None to use the legacy global stream.

    Returns:
        The drawn night strings.
    """
    n_draw = max(1, int(n_total * frac))
    if n_draw > len(pool):
        raise ValueError(
            f"Cannot draw {n_draw} nights (frac={frac} of {n_total} total) "
            f"from a pool of {len(pool)} unassigned nights."
        )
    if rng is None:
        drawn = np.random.choice(np.asarray(pool), size=n_draw, replace=False)
    else:
        drawn = rng.choice(np.asarray(pool), size=n_draw, replace=False)
    return [str(n) for n in drawn]


def _take_explicit(explicit: Sequence[str], pool: List[str], name: str) -> List[str]:
    """Validate an explicit night list against the unassigned pool.

    Args:
        explicit: Requested night strings.
        pool: Nights still available for assignment.
        name: Split name, used in error messages.

    Returns:
        The requested nights as strings.
    """
    requested = [str(n) for n in explicit]
    pool_set = set(pool)
    missing = [n for n in requested if n not in pool_set]
    if missing:
        raise ValueError(
            f"{name}_nights contains {len(missing)} night(s) not available in the "
            f"dataset (or already assigned to an earlier split): {missing[:10]}"
        )
    return requested


def resolve_night_split(
    unique_nights: Sequence[str],
    seed: int,
    val_nights: Optional[Sequence[str]] = None,
    val_frac: Optional[float] = None,
    test_nights: Optional[Sequence[str]] = None,
    test_frac: Optional[float] = None,
) -> NightSplit:
    """Partition nights into train, val, and test.

    Val resolves first, then test from the remaining nights, then train takes
    the rest. For each of val and test: an explicit night list wins over a
    fraction (the fraction is ignored with an INFO log), a fraction draws
    ``max(1, int(n_total * frac))`` nights, and both None leaves the split
    empty.

    The val draw uses the legacy global numpy stream (``np.random.seed(seed)``
    then ``np.random.choice``) so that adding a test split to an existing config
    does not change which nights land in val. The test draw uses an independent
    ``np.random.default_rng(seed)``.

    Args:
        unique_nights: Every night present in the dataset.
        seed: Seed for random draws.
        val_nights: Explicit validation nights, or None.
        val_frac: Validation fraction of the total night count, or None.
        test_nights: Explicit test nights, or None.
        test_frac: Test fraction of the total night count, or None.

    Returns:
        The resolved NightSplit.
    """
    all_nights = [str(n) for n in unique_nights]
    n_total = len(all_nights)
    if n_total == 0:
        raise ValueError("resolve_night_split: unique_nights is empty.")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    pool = list(all_nights)

    if val_nights:
        if val_frac is not None:
            logger.info(
                "Both val_nights and val_frac supplied; using the explicit "
                "val_nights list and ignoring val_frac."
            )
        val = _take_explicit(val_nights, pool, 'val')
    elif val_frac is not None:
        val = _draw(pool, n_total, val_frac, rng=None)
    else:
        val = []
    pool = [n for n in pool if n not in set(val)]

    if test_nights:
        if test_frac is not None:
            logger.info(
                "Both test_nights and test_frac supplied; using the explicit "
                "test_nights list and ignoring test_frac."
            )
        test = _take_explicit(test_nights, pool, 'test')
    elif test_frac is not None:
        test = _draw(pool, n_total, test_frac, rng=rng)
    else:
        test = []
    pool = [n for n in pool if n not in set(test)]

    train = pool

    assert not (set(train) & set(val)), "train and val nights overlap"
    assert not (set(train) & set(test)), "train and test nights overlap"
    assert not (set(val) & set(test)), "val and test nights overlap"
    assert set(train) | set(val) | set(test) == set(all_nights), \
        "split does not partition unique_nights"
    if not train:
        raise ValueError(
            "Resolved split leaves no training nights. Check val_nights / "
            "val_frac / test_frac."
        )

    split = NightSplit(train=sorted(train), val=sorted(val), test=sorted(test), seed=int(seed))
    logger.info(f"Resolved night split: {split.counts} from {n_total} nights (seed={seed})")
    return split
