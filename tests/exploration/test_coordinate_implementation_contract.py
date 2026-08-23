from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
OCCUPANCY = ROOT / "ros2_ws/src/lunar_pure_exploration_core/src/occupancy_grid.cpp"
RASTER = ROOT / "ros2_ws/src/lunar_pure_exploration_core/src/task_raster.cpp"
CANDIDATE_HEADER = ROOT / (
    "ros2_ws/src/lunar_pure_exploration_core/include/"
    "lunar_pure_exploration_core/candidate_generator.hpp"
)


def _body(source: str, qualified_name: str) -> str:
    match = re.search(rf"{re.escape(qualified_name)}\s*\([^)]*\)\s*const\s*\{{", source)
    assert match, f"missing body for {qualified_name}"
    start = match.end()
    depth = 1
    index = start
    while depth and index < len(source):
        depth += (source[index] == "{") - (source[index] == "}")
        index += 1
    assert depth == 0
    return source[start:index - 1]


def _struct_body(source: str, name: str) -> str:
    match = re.search(rf"struct\s+{re.escape(name)}\s*\{{", source)
    assert match, f"missing struct {name}"
    start = match.end()
    depth = 1
    index = start
    while depth and index < len(source):
        depth += (source[index] == "{") - (source[index] == "}")
        index += 1
    assert depth == 0
    return source[start:index - 1]


def _assert_shared_center_body(body: str) -> None:
    assert "OccupancyGridView::GridToWorld" in body
    for forbidden in ("std::sin", "std::cos", "origin_x", "origin_y", "resolution"):
        assert forbidden not in body


def test_both_cell_centers_use_the_shared_static_transform():
    occupancy = OCCUPANCY.read_text(encoding="utf-8")
    raster = RASTER.read_text(encoding="utf-8")
    _assert_shared_center_body(_body(occupancy, "OccupancyGridView::CellCenter"))
    _assert_shared_center_body(_body(raster, "TaskRaster::CellCenter"))


def test_task_raster_and_map_instances_delegate_without_coordinate_math():
    occupancy = OCCUPANCY.read_text(encoding="utf-8")
    raster = RASTER.read_text(encoding="utf-8")
    assert "std::sin" not in raster
    assert "std::cos" not in raster
    for name, shared_call in (
        ("TaskRaster::WorldToGrid", "OccupancyGridView::WorldToGrid"),
        ("TaskRaster::GridToWorld", "OccupancyGridView::GridToWorld"),
        ("TaskRaster::WorldToCell", "OccupancyGridView::WorldToCell"),
        ("TaskRaster::CellCornersInGrid", "OccupancyGridView::CellCornersInGrid"),
        ("OccupancyGridView::WorldToGrid", "OccupancyGridView::WorldToGrid"),
        ("OccupancyGridView::GridToWorld", "OccupancyGridView::GridToWorld"),
        ("OccupancyGridView::WorldToCell", "OccupancyGridView::WorldToCell"),
    ):
        source = raster if name.startswith("TaskRaster") else occupancy
        assert shared_call in _body(source, name)


def test_guard_detects_a_locally_duplicated_transform_without_rewriting_sources():
    valid = "return *OccupancyGridView::GridToWorld(geometry_, grid_point);"
    _assert_shared_center_body(valid)
    mutated = valid.replace(
        "*OccupancyGridView::GridToWorld(geometry_, grid_point)",
        "Vec2{geometry_.origin_x + std::cos(geometry_.origin_yaw) * grid_point.x * geometry_.resolution, 0.0}",
    )
    try:
        _assert_shared_center_body(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutation guard accepted duplicated coordinate mathematics")


def _assert_candidate_view_uses_shared_immutable_frontier_key(body: str) -> None:
    shared_owner = (
        "std::shared_ptr<const std::vector<std::int64_t>> "
        "frontier_canonical_key{};"
    )
    assert "std::size_t frontier_index" in body
    assert body.count(shared_owner) == 1
    remainder = body.replace(shared_owner, "")
    assert "canonical_key" not in remainder
    assert "std::vector" not in remainder


def test_candidate_view_keeps_snapshot_index_and_one_shared_immutable_key_owner():
    header = CANDIDATE_HEADER.read_text(encoding="utf-8")
    _assert_candidate_view_uses_shared_immutable_frontier_key(
        _struct_body(header, "CandidateView")
    )


def test_candidate_view_guard_rejects_per_view_full_key_copy():
    valid = """
      std::size_t frontier_index{};
      std::shared_ptr<const std::vector<std::int64_t>> frontier_canonical_key{};
    """
    _assert_candidate_view_uses_shared_immutable_frontier_key(valid)
    mutant = valid.replace(
        "std::shared_ptr<const std::vector<std::int64_t>>",
        "std::vector<std::int64_t>",
    )
    try:
        _assert_candidate_view_uses_shared_immutable_frontier_key(mutant)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutation guard accepted a per-view full-key copy")
