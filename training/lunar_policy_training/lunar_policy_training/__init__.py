"""Training-side shared PPO policy package."""

import os as _os


# PyTorch requires this to be present before the first CUDA BLAS operation.
# Every formal entry point imports this package before importing ``torch``.
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .project_capability import load_project_formal_capability

__all__ = ["load_project_formal_capability"]
