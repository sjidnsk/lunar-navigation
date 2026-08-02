from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_GPU_DOCS = (
    "AGENTS.md",
    "README.md",
    "docs/architecture/system-ownership.md",
    "docs/superpowers/specs/2026-08-02-lunar-navigation-greenfield-ros2-jetson-design.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-greenfield-roadmap.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-1-foundation.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-4-integration-cutover.md",
)
ACTIVE_EXECUTION_PLANS = (
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-1-foundation.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-greenfield-roadmap.md",
    "docs/superpowers/plans/2026-08-02-ubuntu-handoff-task3-5.md",
)
CANONICAL_PACKAGES = (
    "lunar_navigation_msgs lunar_navigation_config lunar_planning_msgs"
)
CANONICAL_ROI_FIELDS = (
    "required_fields: [header, mission_id, revision, desired_state, roi_min_x_m, "
    "roi_min_y_m, roi_max_x_m, roi_max_y_m, science_regions]"
)


def bash_commands(text: str) -> list[str]:
    """Return executable commands from bash fences with continuations joined."""
    commands: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        pending = ""
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pending = f"{pending} {line}".strip()
            if pending.endswith("\\"):
                pending = pending[:-1].rstrip()
                continue
            commands.append(pending)
            pending = ""
        if pending:
            commands.append(pending)
    return commands


def test_active_documents_use_rtx4080_super_names():
    stale = []
    for relative in ACTIVE_GPU_DOCS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if re.search(r"RTX 4080(?! SUPER)", text) or re.search(r"rtx4080(?!_super)", text):
            stale.append(relative)
    assert stale == []


def test_external_baseline_declares_provisional_schema_authority():
    text = (ROOT / "docs/interfaces/external-input-baseline.md").read_text(encoding="utf-8")
    assert "暂定消息字段与语义的唯一权威基线" in text
    assert "Topic 数据生产者仍由外部项目拥有" in text
    assert "不得据此复制" not in text


def test_agent_rules_record_the_approved_provisional_exception():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "暂定提供同名 schema" in text
    assert "不得与上游同名包共存" in text


def test_active_plans_preserve_provisional_schema_exception():
    volume_one = (ROOT / "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-1-foundation.md").read_text(
        encoding="utf-8"
    )
    roadmap = (ROOT / "docs/superpowers/plans/2026-08-02-lunar-navigation-greenfield-roadmap.md").read_text(
        encoding="utf-8"
    )
    design = (ROOT / "docs/superpowers/specs/2026-08-02-lunar-navigation-greenfield-ros2-jetson-design.md").read_text(
        encoding="utf-8"
    )

    assert "新仓暂定提供 `lunar_navigation_msgs` schema" in volume_one
    assert "上游未定义时本仓暂定 schema" in roadmap
    assert "上游未定义时本仓暂定 schema" in design


def test_volume_one_exploration_task_lists_the_complete_canonical_roi_contract():
    """The old shortened required_fields list would omit every ROI boundary value."""
    text = (ROOT / ACTIVE_EXECUTION_PLANS[0]).read_text(encoding="utf-8")
    assert CANONICAL_ROI_FIELDS in text
    assert "required_fields: [header, mission_id, revision, desired_state, science_regions]" not in text


def test_live_interface_checker_commands_bind_the_fresh_overlay_prefix():
    """A sourced overlay alone must not let a shadow package pass the live checker."""
    for relative in ACTIVE_EXECUTION_PLANS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        checker_commands = [
            command for command in bash_commands(text)
            if command.startswith("python3 tools/check_external_interfaces.py")
        ]
        assert checker_commands, relative
        assert all("--expected-lunar-navigation-prefix" in command for command in checker_commands), relative


def test_foundation_completion_gates_build_and_test_all_three_packages():
    """The former two-package gate could pass without generated provisional messages."""
    volume_one = (ROOT / ACTIVE_EXECUTION_PLANS[0]).read_text(encoding="utf-8")
    roadmap = (ROOT / ACTIVE_EXECUTION_PLANS[1]).read_text(encoding="utf-8")
    assert volume_one.count(f"--packages-select {CANONICAL_PACKAGES}") >= 2
    assert roadmap.count(f"--packages-select {CANONICAL_PACKAGES}") >= 2


def test_active_plans_allow_exact_provisional_sources_and_forbid_a_second_provider():
    """A blanket message-source rejection would contradict the approved provisional package."""
    allowlist = (
        "LocalizationStatus.msg`、`ScienceTargetRegion.msg` 和 `ExplorationTask.msg"
    )
    forbidden_blanket = "没有复制的 `lunar_navigation_msgs/msg` 文件"
    for relative in ACTIVE_EXECUTION_PLANS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert forbidden_blanket not in text, relative
        assert allowlist in text, relative
        assert "不得与上游同名包共存" in text or "不得存在第二个同名 provider" in text, relative


def test_active_plan_commands_keep_python_and_colcon_artifacts_outside_source():
    """Default pytest/colcon caches and logs must not appear in a source checkout."""
    for relative in ACTIVE_EXECUTION_PLANS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        commands = bash_commands(text)
        pytest_commands = [command for command in commands if re.search(r"\bpytest\b", command)]
        colcon_commands = [command for command in commands if command.startswith("colcon ")]
        assert pytest_commands, relative
        assert colcon_commands, relative
        assert all(command.startswith("PYTHONDONTWRITEBYTECODE=1 ") for command in pytest_commands), relative
        assert all("-p no:cacheprovider" in command for command in pytest_commands), relative
        for command in colcon_commands:
            assert command.startswith("colcon --log-base \"$"), (relative, command)
            if re.search(r"\sbuild\s", command):
                assert "--build-base \"$" in command, (relative, command)
                assert "--install-base \"$" in command, (relative, command)
            elif re.search(r"\stest\s", command):
                assert "--build-base \"$" in command, (relative, command)
                assert "--install-base \"$" in command, (relative, command)
                assert "--test-result-base \"$" in command, (relative, command)
            elif re.search(r"\stest-result\s", command):
                assert "--test-result-base \"$" in command, (relative, command)


def test_powershell_pytest_snippets_use_powershell_environment_syntax():
    """POSIX inline assignments are not executable in the Windows review shell."""
    for relative in ACTIVE_EXECUTION_PLANS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for block in re.findall(r"```powershell\n(.*?)```", text, flags=re.DOTALL):
            if "pytest" not in block:
                continue
            assert "PYTHONDONTWRITEBYTECODE=1 python" not in block, relative
            assert '$env:PYTHONDONTWRITEBYTECODE = "1"' in block, relative
            assert all(
                "-p no:cacheprovider" in line
                for line in block.splitlines()
                if "pytest" in line
            ), relative


def test_current_task6_test_result_uses_the_external_global_log_base():
    text = (ROOT / ACTIVE_EXECUTION_PLANS[2]).read_text(encoding="utf-8")
    commands = bash_commands(text)
    assert any(
        command.startswith('colcon --log-base "$LUNAR_VOLUME1_VERIFY/log" test-result')
        and '--test-result-base "$LUNAR_VOLUME1_VERIFY/test-results"' in command
        for command in commands
    )


def test_active_plans_create_each_colcon_test_result_directory():
    """Interface-only packages may emit no result files, so colcon will not create the base."""
    for relative in ACTIVE_EXECUTION_PLANS:
        commands = bash_commands((ROOT / relative).read_text(encoding="utf-8"))
        mkdir_commands = [
            command for command in commands if command.startswith("mkdir -p ")
        ]
        test_result_commands = [
            command for command in commands if re.search(r"\stest-result\s", command)
        ]
        assert test_result_commands, relative
        for command in test_result_commands:
            match = re.search(r'--test-result-base "([^"]+)"', command)
            assert match is not None, (relative, command)
            assert any(match.group(1) in mkdir for mkdir in mkdir_commands), (
                relative,
                match.group(1),
            )


def test_external_handoff_source_is_not_presented_as_an_in_repository_file():
    text = (ROOT / "docs/interfaces/external-input-baseline.md").read_text(encoding="utf-8")
    assert "外部/legacy 交接来源（不复制入本仓）" in text
    assert "只读输入：`docs/外部输入/" not in text
    assert "a4c2db0a6647d59fa7cee5cf8048d18f7bca7c7b11a591a32d33d237c0c06e78" in text


def test_frozen_plans_use_the_correct_public_rosidl_dependency_sets():
    navigation = "`geometry_msgs`、`std_msgs` 为 `<depend>`"
    planning = (
        "`action_msgs`、`builtin_interfaces`、`geometry_msgs`、`nav_msgs`、"
        "`std_msgs`、`trajectory_msgs` 为 `<depend>`"
    )
    for relative in (ACTIVE_EXECUTION_PLANS[0], ACTIVE_EXECUTION_PLANS[2]):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert navigation in text, relative
        assert planning in text, relative
