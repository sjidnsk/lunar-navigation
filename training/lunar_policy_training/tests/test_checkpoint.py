from __future__ import annotations

import pathlib
import random
import sys

import numpy as np
import pytest
import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import lunar_policy_training.ppo.checkpoint as checkpoint_module  # noqa: E402
from lunar_policy_training.ppo.checkpoint import (  # noqa: E402
    OBSERVATION_CONTRACT_VERSION,
    CheckpointError,
    TrainingCheckpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


class UnsafeCheckpointValue:
    def __init__(self) -> None:
        self.value = "must not be constructed by the restricted loader"


def _trained_components() -> tuple[torch.nn.Linear, torch.optim.Adam]:
    torch.manual_seed(17)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss = model(torch.tensor([[0.25, -0.5]], dtype=torch.float32)).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def test_checkpoint_roundtrip_restores_model_optimizer_update_and_all_rng(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if one explicit file omitted either train state or an RNG family."""
    model, optimizer = _trained_components()
    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_optimizer = optimizer.state_dict()
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)

    path = tmp_path / "training.pt"
    saved = save_training_checkpoint(path, model, optimizer, update_step=29)
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(50.0)
    optimizer.param_groups[0]["lr"] = 0.25
    random.seed(201)
    np.random.seed(202)
    torch.manual_seed(203)

    loaded = load_training_checkpoint(path, model, optimizer)

    assert saved == loaded
    assert isinstance(loaded, TrainingCheckpoint)
    assert loaded.update_step == 29
    assert loaded.contract_version == OBSERVATION_CONTRACT_VERSION
    assert len(loaded.payload_sha256) == 64
    assert all(character in "0123456789abcdef" for character in loaded.payload_sha256)
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    _assert_nested_equal(optimizer.state_dict(), expected_optimizer)
    assert random.random() == expected_random
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)


def test_checkpoint_payload_contains_only_task_one_primitives(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if scheduling, durable history, or candidate selection leaked in."""
    model, optimizer = _trained_components()
    path = tmp_path / "training.pt"

    save_training_checkpoint(path, model, optimizer, update_step=3)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert set(payload) == {"body", "body_sha256"}
    assert set(payload["body"]) == {
        "schema_version",
        "contract_version",
        "update_step",
        "model_state",
        "optimizer_state",
        "rng_state",
    }
    assert set(payload["body"]["rng_state"]) == {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }


def test_checkpoint_rejects_wrong_contract_without_mutating_model(
    tmp_path: pathlib.Path,
) -> None:
    model, optimizer = _trained_components()
    path = tmp_path / "training.pt"
    save_training_checkpoint(path, model, optimizer, update_step=1)
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(CheckpointError, match="contract version mismatch"):
        load_training_checkpoint(
            path,
            model,
            optimizer,
            expected_contract_version="DifferentObservationContract",
        )

    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_checkpoint_rejects_nonfinite_model_before_writing(
    tmp_path: pathlib.Path,
) -> None:
    model, optimizer = _trained_components()
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = torch.inf
    path = tmp_path / "training.pt"

    with pytest.raises(CheckpointError, match="non-finite tensor"):
        save_training_checkpoint(path, model, optimizer, update_step=1)

    assert not path.exists()


@pytest.mark.parametrize(
    "payload",
    (
        {"body": {}},
        {"body": {}, "body_sha256": "0" * 64, "unexpected": True},
    ),
)
def test_checkpoint_rejects_malformed_payload(
    tmp_path: pathlib.Path, payload: dict[str, object]
) -> None:
    model, optimizer = _trained_components()
    path = tmp_path / "malformed.pt"
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="structure is invalid"):
        load_training_checkpoint(path, model, optimizer)


def test_checkpoint_rejects_malformed_rng_before_mutating_model(
    tmp_path: pathlib.Path,
) -> None:
    model, optimizer = _trained_components()
    path = tmp_path / "malformed-rng.pt"
    save_training_checkpoint(path, model, optimizer, update_step=2)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["body"]["rng_state"]["python"] = ("invalid",)
    payload["body_sha256"] = checkpoint_module._semantic_sha256(payload["body"])
    torch.save(payload, path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(7.0)
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(CheckpointError, match="Python RNG state is invalid"):
        load_training_checkpoint(path, model, optimizer)

    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_checkpoint_restricted_loader_rejects_unsafe_pickle(
    tmp_path: pathlib.Path,
) -> None:
    model, optimizer = _trained_components()
    path = tmp_path / "unsafe.pt"
    torch.save(UnsafeCheckpointValue(), path)

    with pytest.raises(CheckpointError, match="restricted loader rejected"):
        load_training_checkpoint(path, model, optimizer)


def test_checkpoint_rejects_missing_explicit_file(tmp_path: pathlib.Path) -> None:
    model, optimizer = _trained_components()

    with pytest.raises(CheckpointError, match="checkpoint file is missing"):
        load_training_checkpoint(tmp_path / "missing.pt", model, optimizer)
