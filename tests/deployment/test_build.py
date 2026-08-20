from __future__ import annotations

from pathlib import Path

from deployment.luna_runtime.build import REQUIRED_PACKAGES, make_build_plan
from deployment.luna_runtime.config import load_runtime_config
from deployment.luna_runtime.state import resolve_runtime_paths


def test_default_build_selects_current_runtime_packages_and_never_training_bridge(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_runtime_config(root / "deployment/config/runtime.default.yaml")
    paths = resolve_runtime_paths("dev", home=tmp_path)

    plan = make_build_plan(config, paths, root)

    assert "lunar_planner_training_bridge" not in plan.packages
    assert plan.packages == REQUIRED_PACKAGES
    assert plan.packages[-1] == "lunar_policy_runtime"
    assert plan.build_base == paths.data / "build"
    assert all(str(base).startswith(str(paths.data)) for base in (plan.build_base, plan.install_base, plan.log_base))


def test_task3_adapted_build_includes_local_map_adapter(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_runtime_config(root / "deployment/config/task3-adapted.runtime.yaml")
    plan = make_build_plan(config, resolve_runtime_paths("dev", home=tmp_path), root)

    assert "luna_t3_map_adapter" in plan.packages


def test_enabled_wheeled_controller_is_selected_for_build(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "deployment/config/task3-adapted.runtime.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "enabled.yaml"
    config_path.write_text(text.replace("    enabled: false", "    enabled: true"), encoding="utf-8")

    plan = make_build_plan(
        load_runtime_config(config_path), resolve_runtime_paths("dev", home=tmp_path), root
    )

    assert "luna_wheeled_controller" in plan.packages
