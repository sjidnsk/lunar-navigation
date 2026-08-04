"""Locked external lunar-polar data sources and deterministic split metadata."""

from .source_lock import PolarSourceLock, SourceLockError, verify_source_lock

__all__ = [
    "PolarSourceLock",
    "SourceLockError",
    "SplitCatalogRow",
    "SplitError",
    "build_split_catalog",
    "jaxa_sites",
    "nasa_windows",
    "verify_source_lock",
]


def __getattr__(name: str) -> object:
    """Keep ``python -m lunar_policy_training.polar_data.split`` warning-free."""
    if name in {
        "SplitCatalogRow",
        "SplitError",
        "build_split_catalog",
        "jaxa_sites",
        "nasa_windows",
    }:
        from . import split

        return getattr(split, name)
    raise AttributeError(name)
