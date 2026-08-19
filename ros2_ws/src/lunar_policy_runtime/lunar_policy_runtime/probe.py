from __future__ import annotations

from pathlib import Path


class ModelProbeError(RuntimeError):
    pass


def probe_model_artifact(onnx_path: Path, backend: str) -> None:
    """Probe only when a caller explicitly asks to use a model backend.

    The first deployment runtime deliberately never calls this from fallback
    planning, because no ROS policy adapter is present yet.
    """
    if backend not in {"onnxruntime", "trtexec"}:
        raise ModelProbeError("MODEL_BACKEND_UNSUPPORTED")
    if not onnx_path.is_file():
        raise ModelProbeError("MODEL_ONNX_MISSING")
