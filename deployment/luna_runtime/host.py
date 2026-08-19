from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HostFacts:
    """Read-only facts used to decide whether a bundle fits this host."""

    os_id: str
    os_version: str
    architecture: str
    ros_distro: str
    l4t: str | None = None
    jetpack: str | None = None
