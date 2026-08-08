"""Locked external lunar-polar data sources and deterministic split metadata."""

from .source_lock import PolarSourceLock, SourceLockError, verify_source_lock

__all__ = [
    "PolarSourceLock",
    "SCENARIO_MANIFEST_SCHEMA",
    "ScenarioManifestError",
    "SourceLockError",
    "SplitCatalogRow",
    "SplitError",
    "build_split_catalog",
    "build_scenario_manifest_document",
    "jaxa_sites",
    "load_scenario_manifest",
    "nasa_windows",
    "verify_source_lock",
    "write_scenario_manifest",
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
    if name in {
        "SCENARIO_MANIFEST_SCHEMA",
        "ScenarioManifestError",
        "build_scenario_manifest_document",
        "load_scenario_manifest",
        "write_scenario_manifest",
    }:
        from . import scenario_manifest

        return getattr(scenario_manifest, name)
    raise AttributeError(name)
