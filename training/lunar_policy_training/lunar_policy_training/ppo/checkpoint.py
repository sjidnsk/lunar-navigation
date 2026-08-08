"""Explicit single-file checkpoint primitives for the imported PPO core."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


OBSERVATION_CONTRACT_VERSION = "ObservationContractV1"
CHECKPOINT_SCHEMA_VERSION = "lunar-ppo-core-checkpoint/v1"


class CheckpointError(ValueError):
    """A checkpoint cannot be safely written or restored."""


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    update_step: int
    contract_version: str
    payload_sha256: str


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    update_step: int,
    contract_version: str = OBSERVATION_CONTRACT_VERSION,
) -> TrainingCheckpoint:
    """Write the current training primitives to exactly one explicit file."""
    target = _validated_save_path(path)
    _validate_components(model, optimizer)
    if type(update_step) is not int or update_step < 0:
        raise CheckpointError("checkpoint update step must be a non-negative integer")
    _validate_contract_version(contract_version)

    model_state = _cpu_copy(model.state_dict())
    optimizer_state = _cpu_copy(optimizer.state_dict())
    _validate_finite_tensors(model_state, state_name="model")
    _validate_finite_tensors(optimizer_state, state_name="optimizer")
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "contract_version": contract_version,
        "update_step": update_step,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "rng_state": _capture_rng_state(),
    }
    payload_sha256 = _semantic_sha256(body)
    try:
        torch.save(
            {"body": body, "body_sha256": payload_sha256},
            target,
        )
    except Exception as error:
        raise CheckpointError("checkpoint file could not be written") from error
    return TrainingCheckpoint(
        update_step=update_step,
        contract_version=contract_version,
        payload_sha256=payload_sha256,
    )


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_contract_version: str = OBSERVATION_CONTRACT_VERSION,
) -> TrainingCheckpoint:
    """Validate and restore one explicit checkpoint without path discovery."""
    target = _validated_load_path(path)
    _validate_components(model, optimizer)
    _validate_contract_version(expected_contract_version)
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError(
            "checkpoint restricted loader rejected the payload"
        ) from error
    body, payload_sha256 = _validated_payload(
        payload,
        expected_contract_version=expected_contract_version,
    )
    live_model_state = _cpu_copy(model.state_dict())
    live_optimizer_state = _cpu_copy(optimizer.state_dict())
    live_rng_state = _capture_rng_state()
    try:
        model.load_state_dict(dict(body["model_state"]), strict=True)
        optimizer.load_state_dict(dict(body["optimizer_state"]))
        _restore_rng_state(body["rng_state"])
    except Exception as error:
        _rollback_training_state(
            model,
            optimizer,
            model_state=live_model_state,
            optimizer_state=live_optimizer_state,
            rng_state=live_rng_state,
        )
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError("checkpoint train state cannot be restored") from error
    return TrainingCheckpoint(
        update_step=body["update_step"],
        contract_version=body["contract_version"],
        payload_sha256=payload_sha256,
    )


def _validated_payload(
    payload: object,
    *,
    expected_contract_version: str,
) -> tuple[Mapping[object, object], str]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "body",
        "body_sha256",
    }:
        raise CheckpointError("checkpoint payload structure is invalid")
    body = payload["body"]
    payload_sha256 = payload["body_sha256"]
    if not isinstance(body, Mapping) or set(body) != {
        "schema_version",
        "contract_version",
        "update_step",
        "model_state",
        "optimizer_state",
        "rng_state",
    }:
        raise CheckpointError("checkpoint body structure is invalid")
    if not _is_sha256(payload_sha256):
        raise CheckpointError("checkpoint payload hash is invalid")
    if _semantic_sha256(body) != payload_sha256:
        raise CheckpointError("checkpoint payload hash mismatch")
    if body["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("checkpoint schema version mismatch")
    if body["contract_version"] != expected_contract_version:
        raise CheckpointError("checkpoint contract version mismatch")
    if type(body["update_step"]) is not int or body["update_step"] < 0:
        raise CheckpointError("checkpoint update step is invalid")
    if not isinstance(body["model_state"], Mapping):
        raise CheckpointError("checkpoint model state is invalid")
    if not isinstance(body["optimizer_state"], Mapping):
        raise CheckpointError("checkpoint optimizer state is invalid")
    _validate_finite_tensors(body["model_state"], state_name="model")
    _validate_finite_tensors(body["optimizer_state"], state_name="optimizer")
    _validated_rng_state(body["rng_state"])
    return body, payload_sha256


def _capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].copy()),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
        ),
    }


def _restore_rng_state(state: object) -> None:
    validated = _validated_rng_state(state)
    _apply_rng_state(validated)


def _apply_rng_state(validated: Mapping[object, object]) -> None:
    numpy_state = validated["numpy"]
    try:
        random.setstate(validated["python"])
        np.random.set_state(
            (
                numpy_state[0],
                numpy_state[1].detach().cpu().numpy().copy(),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            )
        )
        torch.set_rng_state(validated["torch_cpu"])
        cuda_states = tuple(validated["torch_cuda"])
        if cuda_states:
            if not torch.cuda.is_available():
                raise CheckpointError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            if len(cuda_states) != torch.cuda.device_count():
                raise CheckpointError("CUDA RNG device count mismatch")
            torch.cuda.set_rng_state_all(list(cuda_states))
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError("checkpoint RNG state cannot be restored") from error


def _rollback_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    model_state: object,
    optimizer_state: object,
    rng_state: Mapping[object, object],
) -> None:
    try:
        if not isinstance(model_state, Mapping) or not isinstance(
            optimizer_state, Mapping
        ):
            raise CheckpointError("checkpoint live-state backup is invalid")
        model.load_state_dict(dict(model_state), strict=True)
        optimizer.load_state_dict(dict(optimizer_state))
        _apply_rng_state(rng_state)
    except Exception as error:
        raise CheckpointError("checkpoint rollback failed") from error


def _validated_rng_state(state: object) -> Mapping[object, object]:
    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise CheckpointError("checkpoint RNG state is incomplete")
    numpy_state = state["numpy"]
    if (
        not isinstance(numpy_state, tuple)
        or len(numpy_state) != 5
        or not isinstance(numpy_state[0], str)
        or not isinstance(numpy_state[1], torch.Tensor)
        or numpy_state[1].dtype != torch.uint32
        or numpy_state[1].device.type != "cpu"
        or numpy_state[1].ndim != 1
        or type(numpy_state[2]) is not int
        or type(numpy_state[3]) is not int
        or type(numpy_state[4]) is not float
    ):
        raise CheckpointError("checkpoint NumPy RNG state is invalid")
    if not isinstance(state["torch_cpu"], torch.Tensor):
        raise CheckpointError("checkpoint Torch CPU RNG state is invalid")
    if not isinstance(state["torch_cuda"], (tuple, list)) or not all(
        isinstance(value, torch.Tensor) for value in state["torch_cuda"]
    ):
        raise CheckpointError("checkpoint Torch CUDA RNG state is invalid")
    try:
        python_validator = random.Random()
        python_validator.setstate(state["python"])
    except Exception as error:
        raise CheckpointError("checkpoint Python RNG state is invalid") from error
    try:
        numpy_validator = np.random.RandomState()
        numpy_validator.set_state(
            (
                numpy_state[0],
                numpy_state[1].detach().cpu().numpy().copy(),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            )
        )
    except Exception as error:
        raise CheckpointError("checkpoint NumPy RNG state is invalid") from error
    try:
        torch_validator = torch.Generator(device="cpu")
        torch_validator.set_state(state["torch_cpu"])
        cuda_states = tuple(state["torch_cuda"])
        if cuda_states:
            if not torch.cuda.is_available():
                raise CheckpointError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            if len(cuda_states) != torch.cuda.device_count():
                raise CheckpointError("CUDA RNG device count mismatch")
            for index, cuda_state in enumerate(cuda_states):
                cuda_validator = torch.Generator(device=f"cuda:{index}")
                cuda_validator.set_state(cuda_state)
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError("checkpoint Torch RNG state is invalid") from error
    return state


def _validate_finite_tensors(value: object, *, state_name: str) -> None:
    visited: set[int] = set()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            if not (item.is_floating_point() or item.is_complex()):
                return
            tensor = item.detach()
            try:
                if tensor.layout == torch.sparse_coo:
                    tensor = tensor.coalesce().values()
                elif tensor.layout != torch.strided:
                    tensor = tensor.values()
                is_finite = bool(torch.isfinite(tensor).all().item())
            except (RuntimeError, TypeError, NotImplementedError) as error:
                raise CheckpointError(
                    f"checkpoint {state_name} state tensor finiteness cannot be verified"
                ) from error
            if not is_finite:
                raise CheckpointError(
                    f"checkpoint {state_name} state contains non-finite tensor"
                )
            return
        if isinstance(item, float) and not math.isfinite(item):
            raise CheckpointError(
                f"checkpoint {state_name} state contains non-finite scalar"
            )
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in visited:
                return
            visited.add(identity)
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        ):
            identity = id(item)
            if identity in visited:
                return
            visited.add(identity)
            for nested in item:
                visit(nested)

    visit(value)


def _semantic_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _feed_hash(digest, value)
    return digest.hexdigest()


def _feed_hash(digest, value: object) -> None:
    if value is None:
        digest.update(b"none;")
        return
    if isinstance(value, bool):
        digest.update(b"bool:1;" if value else b"bool:0;")
        return
    if isinstance(value, int):
        digest.update(f"int:{value};".encode("ascii"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError("checkpoint payload contains non-finite scalar")
        digest.update(f"float:{value.hex()};".encode("ascii"))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        return
    if isinstance(value, bytes):
        digest.update(f"bytes:{len(value)}:".encode("ascii"))
        digest.update(value)
        return
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        tensor = tensor.contiguous()
        digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(f"mapping:{len(value)}:".encode("ascii"))
        for key in sorted(value, key=_key_sort_token):
            _feed_hash(digest, key)
            _feed_hash(digest, value[key])
        return
    if isinstance(value, tuple):
        digest.update(f"tuple:{len(value)}:".encode("ascii"))
        for nested in value:
            _feed_hash(digest, nested)
        return
    if isinstance(value, list):
        digest.update(f"list:{len(value)}:".encode("ascii"))
        for nested in value:
            _feed_hash(digest, nested)
        return
    raise CheckpointError(
        f"checkpoint payload contains unsupported type: {type(value).__name__}"
    )


def _key_sort_token(value: object) -> tuple[str, str]:
    if not isinstance(value, (str, int, float, bool)):
        raise CheckpointError("checkpoint mapping key type is unsupported")
    return type(value).__name__, repr(value)


def _cpu_copy(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(nested) for nested in value)
    if isinstance(value, list):
        return [_cpu_copy(nested) for nested in value]
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise CheckpointError(
        f"checkpoint train state contains unsupported type: {type(value).__name__}"
    )


def _validated_save_path(path: str | Path) -> Path:
    target = Path(path)
    if target.is_symlink():
        raise CheckpointError("checkpoint file must not be a symbolic link")
    if target.exists() and not target.is_file():
        raise CheckpointError("checkpoint path must name a regular file")
    if not target.parent.is_dir():
        raise CheckpointError("checkpoint parent directory is missing")
    return target


def _validated_load_path(path: str | Path) -> Path:
    target = Path(path)
    if target.is_symlink():
        raise CheckpointError("checkpoint file must not be a symbolic link")
    if not target.exists():
        raise CheckpointError("checkpoint file is missing")
    if not target.is_file():
        raise CheckpointError("checkpoint path must name a regular file")
    return target


def _validate_components(
    model: object,
    optimizer: object,
) -> None:
    if not isinstance(model, nn.Module):
        raise CheckpointError("model must be a Torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise CheckpointError("optimizer must be a Torch optimizer")


def _validate_contract_version(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise CheckpointError("checkpoint contract version must be a non-empty string")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "OBSERVATION_CONTRACT_VERSION",
    "CheckpointError",
    "TrainingCheckpoint",
    "load_training_checkpoint",
    "save_training_checkpoint",
]
