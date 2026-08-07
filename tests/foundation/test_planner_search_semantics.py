from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_hierarchical_and_wheel_search_have_no_fixed_work_quotas() -> None:
    search_sources = (
        ROOT
        / "ros2_ws/src/lunar_planner_core/src/hierarchical/grid_search.cpp",
        ROOT / "ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp",
    )
    forbidden_quota_names = (
        "maximum_expanded_states",
        "maximum_reopened_states",
        "maximum_generated_candidates",
        "maximum_open_states",
        "maximum_memory_bytes",
    )
    for source in search_sources:
        text = source.read_text(encoding="utf-8")
        for quota_name in forbidden_quota_names:
            assert quota_name not in text, (source, quota_name)


def test_search_termination_distinguishes_exhaustion_cancel_and_allocation() -> None:
    global_search = (
        ROOT
        / "ros2_ws/src/lunar_planner_core/src/hierarchical/grid_search.cpp"
    ).read_text(encoding="utf-8")
    wheel_search = (
        ROOT / "ros2_ws/src/lunar_planner_core/src/wheel/wheel_lattice.cpp"
    ).read_text(encoding="utf-8")

    assert "while (!open.empty())" in global_search
    assert "GLOBAL_NO_KNOWN_SAFE_ROUTE" in global_search
    assert "GLOBAL_SEARCH_ALLOCATION_FAILED" in global_search
    assert "while (!open.empty())" in wheel_search
    assert "WHEEL_NO_KNOWN_SAFE_ROUTE" in wheel_search
    assert "WHEEL_SEARCH_ALLOCATION_FAILURE" in wheel_search
    assert "REQUEST_CANCELED" in global_search
    assert "REQUEST_CANCELED" in wheel_search
