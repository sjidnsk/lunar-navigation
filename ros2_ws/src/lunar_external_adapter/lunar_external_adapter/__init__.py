"""Replaceable external ROS interface adapter."""

from .profile import (
    InterfaceProfile,
    InterfaceProfileError,
    load_interface_profile,
)

__all__ = [
    "InterfaceProfile",
    "InterfaceProfileError",
    "load_interface_profile",
]
