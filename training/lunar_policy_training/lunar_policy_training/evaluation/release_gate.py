"""Per-platform candidate and release gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ..curriculum import PLATFORMS
from .report import EvaluationReport, REQUIRED_METHODS


@dataclass(frozen=True, slots=True)
class GateRules:
    schema_version: str
    per_platform_rules: Mapping[str, float]
    required_methods: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    failed_rules: tuple[str, ...]


def load_gate_rules(path: Path) -> GateRules:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release gate config must be a mapping")
    schema = payload.get("schema_version")
    rules = payload.get("per_platform_rules")
    methods = payload.get("required_methods")
    if schema != "lunar-policy-release-gate/v1":
        raise ValueError("release gate schema mismatch")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("release gate per-platform rules are missing")
    if not isinstance(methods, list) or any(
        method not in REQUIRED_METHODS for method in methods
    ):
        raise ValueError("release gate required methods are invalid")
    return GateRules(
        schema_version=schema,
        per_platform_rules={name: float(value) for name, value in rules.items()},
        required_methods=tuple(methods),
    )


def evaluate_release_gate(
    report: EvaluationReport, rules: GateRules
) -> GateResult:
    if not isinstance(report, EvaluationReport):
        raise ValueError("release gate requires EvaluationReport")
    if not isinstance(rules, GateRules):
        raise ValueError("release gate requires GateRules")
    failed: list[str] = []
    report_methods = {method.method for method in report.methods}
    for method in rules.required_methods:
        if method not in report_methods:
            failed.append(f"required_methods.{method}")
    if "ppo_policy" not in report_methods:
        return GateResult(False, tuple(failed))
    ppo = report.method("ppo_policy")
    for platform in PLATFORMS:
        metrics = ppo.per_platform[platform]
        for rule_name, threshold in rules.per_platform_rules.items():
            if rule_name.endswith("_min"):
                metric_name = rule_name[: -len("_min")]
                passed = float(getattr(metrics, metric_name)) >= threshold
            elif rule_name.endswith("_max"):
                metric_name = rule_name[: -len("_max")]
                passed = float(getattr(metrics, metric_name)) <= threshold
            else:
                raise ValueError(f"gate rule lacks _min/_max suffix: {rule_name}")
            if not passed:
                failed.append(f"{platform}.{rule_name}")
    return GateResult(not failed, tuple(failed))


__all__ = [
    "GateResult",
    "GateRules",
    "evaluate_release_gate",
    "load_gate_rules",
]
