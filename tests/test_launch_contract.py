"""Static contracts for the isolated pure-planner launch surface."""

import ast
from pathlib import Path
import re

import pytest
import yaml


PURE_PLANNER_ROOT = Path(__file__).resolve().parents[1]
PARAMETERS_PATH = PURE_PLANNER_ROOT / "config" / "pure_planner.yaml"
LAUNCH_PATH = PURE_PLANNER_ROOT / "launch" / "pure_planner.launch.py"
README_PATH = PURE_PLANNER_ROOT / "README.md"
VERIFICATION_PATH = PURE_PLANNER_ROOT / "VERIFICATION.md"
CMAKE_PATH = (
    PURE_PLANNER_ROOT / "ros2_ws" / "src" / "lunar_pure_planner_ros" / "CMakeLists.txt"
)
PACKAGE_PATH = (
    PURE_PLANNER_ROOT / "ros2_ws" / "src" / "lunar_pure_planner_ros" / "package.xml"
)
SERVER_HEADER_PATH = (
    PURE_PLANNER_ROOT
    / "ros2_ws"
    / "src"
    / "lunar_pure_planner_ros"
    / "include"
    / "lunar_pure_planner_ros"
    / "pure_plan_motion_server.hpp"
)
SERVER_SOURCE_PATH = (
    PURE_PLANNER_ROOT
    / "ros2_ws"
    / "src"
    / "lunar_pure_planner_ros"
    / "src"
    / "pure_plan_motion_server.cpp"
)
SERVER_SEAM_PATH = SERVER_SOURCE_PATH.with_name(
    "pure_plan_motion_server_test_seam.hpp"
)

EXPECTED_TOPICS = {
    "global_map_topic": "/Car/T3/mapping/global_overview",
    "local_map_topic": "/Car/T3/mapping/grid_map",
    "odometry_topic": "/Car/T3/localization/odometry",
    "tf_topic": "/tf",
    "action_name": "/Car/T4/plan_motion",
    "diagnostics_topic": "/Car/T4/planning/diagnostics",
    "wheeled_reference_topic": "/Car/T4/planning/wheeled_reference",
    "wheeled_path_topic": "/Car/T4/planning/wheeled_path",
    "wheeled_global_path_topic": "/Car/T4/planning/wheeled_global_path",
    "wheeled_timed_path_topic": "/Car/T4/planning/wheeled_path_timing",
}
EXPECTED_PARAMETERS = {
    "platform_type",
    "platform_config",
    "global_occupancy_threshold_percent",
    "local_occupancy_threshold",
    "wheel_planner_mode",
    "rolling_surface_enabled",
    "rolling_horizon_m",
    "rolling_poll_period_ms",
    "rolling_min_replan_interval_ms",
    "rolling_max_deviation_m",
    *EXPECTED_TOPICS,
}
FORBIDDEN_PARAMETER_FRAGMENTS = {
    "sqlite",
    "revision",
    "status",
    "freshness",
    "covariance",
    "observation",
    "timestamp",
    "tf_static",
}


def load_parameters() -> dict[str, object]:
    """Read the one ROS parameters document and return its shared defaults."""
    document = yaml.safe_load(PARAMETERS_PATH.read_text(encoding="utf-8"))
    return document["/**"]["ros__parameters"]


def _markdown_section(text: str, heading: str) -> str:
    """Return the unique level-two Markdown section named by heading."""
    matches = list(
        re.finditer(rf"^## {re.escape(heading)}\s*$", text, flags=re.MULTILINE)
    )
    assert len(matches) == 1, heading
    next_heading = re.search(r"^## ", text[matches[0].end():], flags=re.MULTILINE)
    end = (
        matches[0].end() + next_heading.start()
        if next_heading is not None
        else len(text)
    )
    return text[matches[0].end():end].strip()


def _paragraph_starting(section: str, prefix: str) -> str:
    """Return one normalized paragraph from a previously isolated section."""
    paragraphs = [
        "".join(line.strip() for line in paragraph.splitlines())
        for paragraph in re.split(r"\n\s*\n", section)
        if paragraph.strip()
    ]
    matches = [paragraph for paragraph in paragraphs if paragraph.startswith(prefix)]
    assert len(matches) == 1, prefix
    return matches[0]


def _bullet_list_after(section: str, prefix: str) -> list[str]:
    """Parse the first contiguous Markdown bullet list after a local paragraph."""
    lines = section.splitlines()
    paragraph_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().startswith(prefix)
        ),
        None,
    )
    assert paragraph_index is not None, prefix
    first_bullet = next(
        index
        for index in range(paragraph_index + 1, len(lines))
        if lines[index].strip().startswith("- ")
    )
    bullets = []
    for line in lines[first_bullet:]:
        stripped = line.strip()
        if not stripped.startswith("- "):
            break
        bullets.append(stripped[2:].strip().strip("`"))
    return bullets


def _markdown_table_after(
    section: str, prefix: str
) -> tuple[list[str], list[list[str]]]:
    """Parse the first Markdown table after a local paragraph."""
    lines = section.splitlines()
    paragraph_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().startswith(prefix)
        ),
        None,
    )
    assert paragraph_index is not None, prefix
    first_row = next(
        index
        for index in range(paragraph_index + 1, len(lines))
        if lines[index].strip().startswith("|")
    )
    raw_rows = []
    for line in lines[first_row:]:
        if not line.strip().startswith("|"):
            break
        raw_rows.append(
            [cell.strip() for cell in line.strip().strip("|").split("|")]
        )
    assert len(raw_rows) >= 3
    assert all(set(cell) <= {"-", ":"} for cell in raw_rows[1])
    return raw_rows[0], raw_rows[2:]


def _clause_containing(paragraph: str, token: str) -> str:
    """Bind one field token to its local Chinese semicolon/full-stop clause."""
    matches = [
        clause.strip()
        for clause in re.split(r"[；。]", paragraph)
        if token in clause
    ]
    assert len(matches) == 1, token
    return matches[0]


EXPECTED_EVIDENCE_STATUSES = {
    "bag_metadata": "SEARCHED_ZERO_CANDIDATES",
    "real_bag_replay": "NOT_RUN_REAL_BAG_NOT_LOCATED",
    "lava_tube_replay": "NOT_RUN_REAL_BAG_NOT_LOCATED",
    "lunar_surface_e2e": "NOT_RUN_MISSING_GLOBAL_INPUT",
    "global_producer": "NOT_RUN",
    "jetson_agx": "NOT_RUN",
    "historical_missing_global_bag": "HISTORICAL_SUBSET_ONLY",
    "synthetic_global": "FORBIDDEN",
}

EXPECTED_EVIDENCE_NOTES = {
    "bag_metadata": (
        "已搜索 `/home/kai`（maxdepth 8）下的 `metadata.yaml`，候选数为 0。"
    ),
    "real_bag_replay": (
        "metadata 搜索已完成且候选数为 0；没有候选可执行 `ros2 bag info`，"
        "也未回放任何真实 bag。"
    ),
    "lava_tube_replay": (
        "未定位真实 bag，未执行 `ros2 bag info` 或回放，不能声明 "
        "`LAVA_TUBE` 已验证。"
    ),
    "lunar_surface_e2e": (
        "缺少真实 `/Car/T3/mapping/global_overview` producer 或包含该 Topic 的 bag，"
        "不能证明 `LUNAR_SURFACE` 全局加局部端到端成功。"
    ),
    "global_producer": "未完成真实 producer 集成。",
    "jetson_agx": (
        "未执行 Jetson AGX Orin 原生构建、性能、功耗或稳定性验收。"
    ),
    "historical_missing_global_bag": (
        "历史记录中的 bag 缺少 `/Car/T3/mapping/global_overview`，"
        "仅可作为 `LAVA_TUBE` 局部输入子集；本轮未找到其 metadata，未实际回放。"
    ),
    "synthetic_global": (
        "禁止生成或使用假全局图；不得用合成、零填充或局部图替代真实 "
        "`/Car/T3/mapping/global_overview`；月表验收仍需实机在线全局总览，"
        "或重新录制并核验包含该 Topic 的 rosbag。"
    ),
}


def _parse_readme_evidence_contract(text: str) -> dict[str, tuple[str, str]]:
    """Parse the terminal evidence section as one closed Markdown table."""
    heading = "## 当前证据边界"
    assert text.splitlines().count(heading) == 1, heading
    _, evidence = text.split(f"{heading}\n", maxsplit=1)
    lines = [line.strip() for line in evidence.splitlines() if line.strip()]
    assert len(lines) == 10, "evidence section must contain only one eight-row table"
    assert lines[0] == "| key | status | 边界 |"
    assert lines[1] == "| --- | --- | --- |"

    records: dict[str, tuple[str, str]] = {}
    for line in lines[2:]:
        assert line.startswith("|") and line.endswith("|"), line
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        assert len(cells) == 3, line
        key_match = re.fullmatch(r"`([a-z0-9_]+)`", cells[0])
        status_match = re.fullmatch(r"`([A-Z_]+)`", cells[1])
        assert key_match is not None, cells[0]
        assert status_match is not None, cells[1]
        key = key_match.group(1)
        assert key not in records, f"duplicate evidence key: {key}"
        assert cells[2], f"missing evidence boundary: {key}"
        records[key] = (status_match.group(1), cells[2])

    assert {key: status for key, (status, _) in records.items()} == (
        EXPECTED_EVIDENCE_STATUSES
    )
    for key, canonical_note in EXPECTED_EVIDENCE_NOTES.items():
        assert records[key][1] == canonical_note, (
            f"{key} evidence boundary must match canonical note"
        )
    return records


def _replace_once(text: str, old: str, new: str) -> str:
    """Apply one deliberate README mutation without masking fixture drift."""
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


@pytest.mark.parametrize("parser", [_bullet_list_after, _markdown_table_after])
def test_readme_parsers_report_a_missing_anchor_with_its_prefix(parser) -> None:
    """A missing local anchor must name the contract prefix, not leak StopIteration."""
    prefix = "missing README anchor"
    with pytest.raises(AssertionError, match=prefix):
        parser("one paragraph\n\n- one bullet\n\n| a | b |", prefix)


def test_default_topics_are_exact_car_contracts() -> None:
    """A changed external topic must not silently redirect the pure planner."""
    params = load_parameters()
    for key, expected_topic in EXPECTED_TOPICS.items():
        assert params[key] == expected_topic


def test_parameters_are_limited_to_the_pure_planner_contract() -> None:
    """Excluded readiness and legacy inputs must not become launch parameters."""
    params = load_parameters()
    assert set(params) == EXPECTED_PARAMETERS
    assert params["global_occupancy_threshold_percent"] == 50
    assert params["local_occupancy_threshold"] == 0.5
    assert params["platform_config"] == ""
    forbidden = {
        key
        for key in params
        if any(fragment in key.lower() for fragment in FORBIDDEN_PARAMETER_FRAGMENTS)
    }
    assert not forbidden


def _node_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == "Node")
             or (isinstance(node.func, ast.Attribute) and node.func.attr == "Node"))
    ]


def _keyword_value(call: ast.Call, key: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == key)


def test_launch_starts_only_the_isolated_pure_planner_node() -> None:
    """The launch file must not compose the new action server with old launchers."""
    text = LAUNCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = _node_calls(tree)
    assert len(calls) == 1
    node = calls[0]
    assert ast.literal_eval(_keyword_value(node, "package")) == "lunar_pure_planner_ros"
    assert ast.literal_eval(_keyword_value(node, "executable")) == "lunar_pure_planner_node"
    assert ast.literal_eval(_keyword_value(node, "name")) == "pure_planner"
    assert ast.literal_eval(_keyword_value(node, "namespace")) == ""
    assert "IncludeLaunchDescription" not in text
    assert "Lifecycle" not in text
    assert "lunar_planner" not in text


def test_launch_passes_the_shared_parameter_file_and_explicit_overrides() -> None:
    """The launch path must keep platform choice explicit without duplicating config."""
    tree = ast.parse(LAUNCH_PATH.read_text(encoding="utf-8"))
    node = _node_calls(tree)[0]
    parameters = _keyword_value(node, "parameters")
    assert isinstance(parameters, ast.List)
    assert len(parameters.elts) == 2
    assert isinstance(parameters.elts[0], ast.Name)
    assert parameters.elts[0].id == "params_file"
    assert isinstance(parameters.elts[1], ast.Dict)
    assert [ast.literal_eval(key) for key in parameters.elts[1].keys] == [
        "platform_type", "wheel_planner_mode", "rolling_surface_enabled"
    ]
    assert [value.id for value in parameters.elts[1].values if isinstance(value, ast.Name)] == [
        "platform_type", "wheel_planner_mode", "rolling_surface_enabled"
    ]


def test_cmake_installs_the_single_top_level_config_and_launch_sources() -> None:
    """Installed runtime files must originate at pure_planner/, not a copied package tree."""
    cmake = CMAKE_PATH.read_text(encoding="utf-8")
    assert "${CMAKE_CURRENT_SOURCE_DIR}/../../../config" in cmake
    assert "${CMAKE_CURRENT_SOURCE_DIR}/../../../launch" in cmake
    assert "DESTINATION share/${PROJECT_NAME}/config" in cmake
    assert "DESTINATION share/${PROJECT_NAME}/launch" in cmake


def test_test_only_action_dependency_and_direct_launch_runtime_dependencies() -> None:
    """Test helpers must not inflate runtime linkage; launch imports stay declared."""
    cmake = CMAKE_PATH.read_text(encoding="utf-8")
    testing = cmake.index("if(BUILD_TESTING)")
    assert cmake.index("find_package(action_msgs REQUIRED)") > testing

    package = PACKAGE_PATH.read_text(encoding="utf-8")
    assert "<test_depend>action_msgs</test_depend>" in package
    assert "<exec_depend>launch</exec_depend>" in package
    assert "<exec_depend>launch_ros</exec_depend>" in package


def test_production_server_header_and_translation_unit_exclude_test_seams() -> None:
    """Installed ABI and the production TU must not carry test-only branches."""
    forbidden = {
        "ServerExecutionHooks",
        "PurePlanMotionServerTestFactory",
        "pure_plan_motion_server_test_seam",
        "before_terminal_primitive",
        "after_terminal_primitive",
        "before_cancel_response_return",
    }
    public_header = SERVER_HEADER_PATH.read_text(encoding="utf-8")
    production_source = SERVER_SOURCE_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in public_header
        assert token not in production_source
    assert not SERVER_SEAM_PATH.exists()
    assert "friend class" not in public_header
    # Public constructor, destructor, and deleted copy constructor only.
    assert public_header.count("PurePlanMotionServer(") == 3


def test_readme_cancel_instructions_use_the_supported_foreground_action_cli() -> None:
    """The Humble runbook must not direct operators to a nonexistent cancel verb."""
    text = README_PATH.read_text(encoding="utf-8")
    assert "ros2 action cancel" not in text
    assert "ros2 action send_goal" in text
    assert "Ctrl+C" in text
    assert "SIGINT" in text
    assert "CancelGoal" in text


def test_task16_docs_distinguish_verified_baselines_from_the_uncommitted_report() -> None:
    """A committed report must not keep describing itself as an uncommitted patch."""
    readme = README_PATH.read_text(encoding="utf-8")
    verification = VERIFICATION_PATH.read_text(encoding="utf-8")

    assert "生产源码/前置测试基线为" in readme
    assert "验证源码基线为" not in readme
    assert "生产源码/前置测试基线" in verification
    assert "本报告所在 Task 16 change set，验证时尚未提交" in verification
    assert "git log -1 -- pure_planner/VERIFICATION.md" in verification
    assert "current patch" not in verification


def test_readme_timing_table_maps_each_output_field_to_its_exact_unit() -> None:
    """Swapping Action seconds with Topic milliseconds must break the runbook."""
    results = _markdown_section(
        README_PATH.read_text(encoding="utf-8"), "结果与执行语义"
    )
    header, rows = _markdown_table_after(results, "计时和调用次数的实际输出位置")
    assert header == ["输出", "字段", "单位与含义"]

    canonical_units = {
        "seconds": "seconds",
        "milliseconds": "milliseconds",
        "次数": "count",
    }
    timing_contract: dict[tuple[str, str], tuple[str, str]] = {}
    for output, fields, semantics in rows:
        unit, _, meaning = semantics.partition("；")
        assert unit in canonical_units
        for field in fields.split("、"):
            timing_contract[(output, field.strip("`"))] = (
                canonical_units[unit],
                meaning,
            )

    assert set(timing_contract) == {
        ("Action Result", "diagnostics.elapsed_s"),
        ("diagnostics Topic", "global_elapsed_ms"),
        ("diagnostics Topic", "local_elapsed_ms"),
        ("diagnostics Topic", "total_elapsed_ms"),
        ("diagnostics Topic", "global_call_count"),
        ("diagnostics Topic", "local_call_count"),
        ("TimedPath Topic", "planning_time"),
    }
    action_unit, action_meaning = timing_contract[
        ("Action Result", "diagnostics.elapsed_s")
    ]
    assert action_unit == "seconds"
    assert "总耗时" in action_meaning
    for field in ("global_elapsed_ms", "local_elapsed_ms", "total_elapsed_ms"):
        assert timing_contract[("diagnostics Topic", field)][0] == "milliseconds"
    for field in ("global_call_count", "local_call_count"):
        assert timing_contract[("diagnostics Topic", field)][0] == "count"
    timed_path_unit, timed_path_meaning = timing_contract[("TimedPath Topic", "planning_time")]
    assert timed_path_unit == "seconds"
    assert "sec + nanosec" in timed_path_meaning


def test_readme_failure_list_is_exact_and_keeps_success_separate() -> None:
    """Dropping a failure or adding a seventh reason must break the runbook."""
    results = _markdown_section(
        README_PATH.read_text(encoding="utf-8"), "结果与执行语义"
    )
    preface = _paragraph_starting(results, "Action Result 和 diagnostics Topic")
    success = re.fullmatch(
        r"Action Result 和 diagnostics Topic 的生产成功原因码为 `([^`]+)`。"
        r"失败时 `reason_code` 才限定为以下六类：",
        preface,
    )
    assert success is not None
    assert success.group(1) == "PLAN_FOUND"

    failures = _bullet_list_after(results, "Action Result 和 diagnostics Topic")
    assert len(failures) == 6
    assert set(failures) == {
        "INVALID_INPUT",
        "GOAL_OUTSIDE_LOCAL_MAP",
        "NO_PATH",
        "TIMEOUT",
        "REQUEST_CANCELED",
        "PLANNER_ERROR",
    }
    assert success.group(1) not in failures


def test_readme_removed_admission_fields_are_bound_to_non_gating_semantics() -> None:
    """Each removed readiness input must stay explicitly outside request admission."""
    results = _markdown_section(
        README_PATH.read_text(encoding="utf-8"), "结果与执行语义"
    )
    admission = _paragraph_starting(results, "下列兼容字段不是准入条件：")

    stamp = _clause_containing(admission, "输入消息的 stamp")
    assert stamp == (
        "下列兼容字段不是准入条件：输入消息的 stamp 不用于排序、新鲜度或时间差判断，"
        "只在 Result 中把本次快照的 `global_map_stamp`、`local_map_stamp` 和 "
        "`state_stamp` 原样回填"
    )

    mission_revision = _clause_containing(admission, "`mission_revision`")
    assert mission_revision == (
        "Goal 的`mission_revision` 原样回传，`mission_id` 不被规划算法读取"
    )

    covariance = _clause_containing(admission, "covariance")
    assert covariance == "Odometry 的 covariance 字段不读取、不拒绝请求，也不回传"

    absent_runtime_inputs = _clause_containing(
        admission, "freshness/revision/version"
    )
    absent_match = re.fullmatch(
        r"运行时地图 ([^、]+)、定位 ([^、]+)、([^、]+)、(.+)"
        r"没有对应 pure planner 订阅或参数，完全不在请求准入面",
        absent_runtime_inputs,
    )
    assert absent_match is not None
    absent_fields = set(absent_match.group(1).split("/")) | {
        absent_match.group(2),
        absent_match.group(3),
        absent_match.group(4),
    }
    assert absent_fields == {
        "freshness",
        "revision",
        "version",
        "status",
        "任务",
        "execution feedback",
    }

    capability_exception = _clause_containing(admission, "`capability_version`")
    assert capability_exception == (
        "这里的 map version 不包括启动时加载并核对的静态平台 `capability_version`"
    )
    assert (
        "后者是固定平台配置合同，不是运行时地图质量门禁" in admission
    )


def test_readme_evidence_section_preserves_each_not_run_boundary() -> None:
    """The evidence boundary is one closed table with exact current states."""
    records = _parse_readme_evidence_contract(
        README_PATH.read_text(encoding="utf-8")
    )
    assert len(records) == 8


@pytest.mark.parametrize(
    "claim",
    [
        pytest.param("NOT_RUN: Jetson VERIFIED", id="not-run-verified"),
        pytest.param("NOT RUN, but PASSED", id="not-run-but-passed"),
        pytest.param("未运行但已验收", id="not-run-but-accepted-zh"),
    ],
)
def test_readme_evidence_contract_rejects_appended_claims(claim: str) -> None:
    """No free-form sentence may be appended after the closed evidence table."""
    text = README_PATH.read_text(encoding="utf-8")
    _parse_readme_evidence_contract(text)
    with pytest.raises(AssertionError, match="only one eight-row table"):
        _parse_readme_evidence_contract(f"{text.rstrip()}\n\n{claim}\n")


@pytest.mark.parametrize(
    ("key", "contradiction"),
    [
        pytest.param("jetson_agx", "Jetson 已验收。", id="jetson-accepted"),
        pytest.param(
            "historical_missing_global_bag",
            "本轮已回放。",
            id="historical-replayed",
        ),
        pytest.param(
            "synthetic_global",
            "已使用合成图。",
            id="synthetic-used",
        ),
    ],
)
def test_readme_evidence_contract_rejects_note_contradictions(
    key: str, contradiction: str
) -> None:
    """Canonical notes must reject contrary facts appended inside a table cell."""
    text = README_PATH.read_text(encoding="utf-8")
    records = _parse_readme_evidence_contract(text)
    note = records[key][1]
    mutated = _replace_once(text, note, f"{note} {contradiction}")
    with pytest.raises(AssertionError, match=f"{key}.*canonical note"):
        _parse_readme_evidence_contract(mutated)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            "`real_bag_replay`",
            "`bag_metadata`",
            id="duplicate-key",
        ),
        pytest.param(
            "`bag_metadata`",
            "`unknown_evidence`",
            id="unknown-key",
        ),
        pytest.param(
            "| `jetson_agx` | `NOT_RUN` |",
            "| `jetson_agx` | `MISSING_GLOBAL` |",
            id="status-mismatch",
        ),
        pytest.param(
            "| `bag_metadata` | `SEARCHED_ZERO_CANDIDATES` |",
            "| `bag_metadata` | `NOT_RUN` |",
            id="metadata-search-demoted-to-not-run",
        ),
        pytest.param(
            "| `real_bag_replay` | `NOT_RUN_REAL_BAG_NOT_LOCATED` |",
            "| `real_bag_replay` | `NOT_RUN` |",
            id="real-replay-loses-not-located-reason",
        ),
        pytest.param(
            "| `lava_tube_replay` | `NOT_RUN_REAL_BAG_NOT_LOCATED` |",
            "| `lava_tube_replay` | `NOT_RUN` |",
            id="lava-replay-loses-not-located-reason",
        ),
        pytest.param(
            "| `lunar_surface_e2e` | `NOT_RUN_MISSING_GLOBAL_INPUT` |",
            "| `lunar_surface_e2e` | `MISSING_GLOBAL` |",
            id="surface-loses-not-run-boundary",
        ),
    ],
)
def test_readme_evidence_contract_rejects_key_and_status_mutations(
    old: str, new: str
) -> None:
    """Duplicate, unknown, or incorrectly promoted records must fail closed."""
    text = README_PATH.read_text(encoding="utf-8")
    _parse_readme_evidence_contract(text)
    with pytest.raises(AssertionError):
        _parse_readme_evidence_contract(_replace_once(text, old, new))


def test_readme_evidence_contract_requires_synthetic_global_prohibition() -> None:
    """Removing the synthetic-global row or its prohibition must fail closed."""
    text = README_PATH.read_text(encoding="utf-8")
    _parse_readme_evidence_contract(text)
    synthetic_row = next(
        line for line in text.splitlines() if line.startswith("| `synthetic_global`")
    )
    without_row = _replace_once(text, f"{synthetic_row}\n", "")
    without_prohibition = _replace_once(
        text, "禁止生成或使用假全局图", "假全局图"
    )
    with pytest.raises(AssertionError):
        _parse_readme_evidence_contract(without_row)
    with pytest.raises(AssertionError, match="synthetic_global.*canonical note"):
        _parse_readme_evidence_contract(without_prohibition)
