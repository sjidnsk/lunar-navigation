"""Lazy contract exports keep CLI/spawn bootstrap free of numerical imports."""

__all__ = ["DecisionObservation", "MapSnapshot", "PolicyMapStore", "Pose",
           "PrivilegedState", "RewardParts", "SensorSpec", "TaskReport",
           "TaskSpec", "Transition"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from importlib import import_module
    module = import_module('.maps' if name == 'PolicyMapStore' else '.contracts', __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
