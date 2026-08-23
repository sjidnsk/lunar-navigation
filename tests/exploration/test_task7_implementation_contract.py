from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros2_ws/src/lunar_pure_exploration_core"
PUBLIC_HEADER = PACKAGE / "include/lunar_pure_exploration_core/coverage.hpp"
CPP = PACKAGE / "src/coverage.cpp"
DETAIL = PACKAGE / "src/detail/coverage_statistics.hpp"
RANK_HEADER = PACKAGE / "include/lunar_pure_exploration_core/candidate_ranker.hpp"
RANK_CPP = PACKAGE / "src/candidate_ranker.cpp"
FAILURE_HEADER = PACKAGE / "include/lunar_pure_exploration_core/failure_memory.hpp"
FAILURE_CPP = PACKAGE / "src/failure_memory.cpp"
CMAKE = PACKAGE / "CMakeLists.txt"


def _body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for offset in range(brace, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : offset]
    raise AssertionError(f"unterminated body for {signature}")


def _compact(source: str) -> str:
    return re.sub(r"\s+", " ", source).strip()


def _code_identifiers(source: str) -> list[str]:
    code_only = re.sub(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',
        " ",
        source,
        flags=re.DOTALL,
    )
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code_only)


def _case_body(switch_body: str, label: str, next_label: str) -> str:
    start = switch_body.index(label) + len(label)
    end = switch_body.index(next_label, start)
    return switch_body[start:end]


def _validate(public: str, cpp: str, detail: str) -> None:
    stats = _body(public, "struct CoverageStats")
    fields = re.findall(r"\bdouble\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", stats)
    assert fields == [
        "polygon_area_m2",
        "task_raster_area_m2",
        "known_free_area_m2",
        "known_occupied_area_m2",
        "unknown_area_m2",
        "outside_map_area_m2",
        "coverage_ratio",
    ]
    assert "CoverageStats CalculateCoverage(const TaskRaster& raster);" in public
    assert re.findall(
        r"^[A-Za-z_][^{}]*;$", public, re.MULTILINE
    ) == ["CoverageStats CalculateCoverage(const TaskRaster& raster);"]

    forbidden = re.compile(
        r"completed|completion|success|terminal|threshold", re.IGNORECASE
    )
    assert forbidden.search(public) is None
    assert forbidden.search(cpp) is None
    assert forbidden.search(detail) is None

    calculate = _compact(_body(cpp, "CoverageStats CalculateCoverage("))
    assert "for (const GridIndex cell : raster.task_cells())" in calculate
    assert (
        "detail::AccumulateCoverageState(raster.Classify(cell), counts);"
        in calculate
    )
    assert "return detail::FinalizeCoverage(" in calculate
    assert "switch" not in calculate
    assert re.search(r"coverage_ratio\s*(?:[<>]=?|==|!=)", calculate) is None

    accumulator = _compact(
        _body(detail, "inline void AccumulateCoverageState(")
    )
    assert "switch (state)" in accumulator
    outside = _case_body(
        accumulator,
        "case CellState::kOutsideTask:",
        "case CellState::kOutsideMap:",
    )
    assert "throw std::logic_error(" in outside
    assert "CheckedIncrement(" not in outside
    for member in ("outside_map", "unknown", "free", "occupied"):
        assert f"CheckedIncrement(counts.{member});" in accumulator

    checked_area = _compact(
        _body(detail, "inline double CheckedCoverageArea(")
    )
    assert checked_area.count("static_cast<long double>(resolution_m)") == 2
    assert "static_cast<long double>(cell_count)" in checked_area
    assert "area_ld > 0.0L" in checked_area
    assert "const double area = static_cast<double>(area_ld);" in checked_area
    assert "!std::isfinite(area)" in checked_area
    assert re.search(r"area\s*<=\s*0\.0", checked_area)

    finalize = _compact(_body(detail, "inline CoverageStats FinalizeCoverage("))
    for count in (
        "counts.free",
        "counts.occupied",
        "counts.unknown",
        "counts.outside_map",
    ):
        assert f"CheckedCoverageArea({count}, resolution_m)" in finalize
    assert "CheckedCoverageArea(task_cell_count, resolution_m)" in finalize
    assert "static_cast<double>(known_cell_count)" in finalize
    assert "static_cast<double>(task_cell_count)" in finalize
    assert re.search(r"coverage_ratio\s*<\s*0\.0", finalize)
    assert re.search(r"coverage_ratio\s*>\s*1\.0", finalize)
    assert re.search(r"coverage_ratio\s*(?:[<>]=?|==|!=)\s*0\.[1-9]", finalize) is None


def _current_sources() -> tuple[str, str, str]:
    return (
        PUBLIC_HEADER.read_text(encoding="utf-8"),
        CPP.read_text(encoding="utf-8"),
        DETAIL.read_text(encoding="utf-8"),
    )


def _validate_ownership_contract(
    rank_header: str,
    rank_cpp: str,
    failure_header: str,
    failure_cpp: str,
) -> None:
    assert _compact(_body(rank_header, "struct CandidateGain")) == (
        "std::size_t candidate_index; double information_gain_m2;"
    )
    assert _compact(_body(rank_header, "struct PlannedCandidate")) == (
        "std::size_t candidate_index; double path_length_m;"
    )
    assert _compact(_body(rank_header, "struct RankedCandidate")) == (
        "std::size_t candidate_index; double information_gain_m2; "
        "double euclidean_distance_m; double path_length_m; "
        "double heading_change_rad; double revisit_penalty; "
        "double rank_value;"
    )

    ranker_private = _body(rank_header, "class CandidateRanker").split(
        "private:", 1
    )[1]
    assert _compact(ranker_private) == (
        "ScoreWeights weights_; double platform_length_m_;"
    )

    failure_entry = _compact(_body(failure_header, "struct Entry"))
    assert failure_entry == (
        "CandidateKey key; Vec2 world_center; double failure_radius_m; "
        "GridGeometry geometry; std::vector<PatchCell> patch;"
    )
    failure_reason = _compact(
        _body(failure_header, "enum class PersistentFailureReason")
    )
    assert failure_reason == "kExecutionReplansExhausted,"

    failure_private = _body(failure_header, "class FailureMemory").split(
        "private:", 1
    )[1]
    assert "CandidateView" not in failure_private
    assert "std::span" not in failure_private
    assert "std::reference_wrapper" not in failure_private
    assert re.search(
        r"\b(?:display|candidate_id|frontier_id|stamp|revision|version)\b",
        failure_private,
    ) is None

    for source in (rank_header, rank_cpp, failure_header, failure_cpp):
        for identifier in _code_identifiers(source):
            lowered = identifier.lower()
            assert lowered not in {
                "id",
                "frontier_id",
                "candidate_id",
                "display_id",
                "display_hash",
            }
            assert all(
                forbidden not in lowered
                for forbidden in ("stamp", "revision", "version")
            )


def test_coverage_implementation_contract() -> None:
    _validate(*_current_sources())


def test_rank_and_failure_ownership_contract() -> None:
    _validate_ownership_contract(
        RANK_HEADER.read_text(encoding="utf-8"),
        RANK_CPP.read_text(encoding="utf-8"),
        FAILURE_HEADER.read_text(encoding="utf-8"),
        FAILURE_CPP.read_text(encoding="utf-8"),
    )


def test_ranked_candidate_display_field_mutation_is_rejected() -> None:
    rank_header = RANK_HEADER.read_text(encoding="utf-8")
    ranked = _body(rank_header, "struct RankedCandidate")
    mutant = rank_header.replace(
        ranked, ranked + "\n  std::uint64_t display_id;", 1
    )
    assert mutant != rank_header
    with pytest.raises(AssertionError):
        _validate_ownership_contract(
            mutant,
            RANK_CPP.read_text(encoding="utf-8"),
            FAILURE_HEADER.read_text(encoding="utf-8"),
            FAILURE_CPP.read_text(encoding="utf-8"),
        )


def test_ranker_retained_candidate_span_mutation_is_rejected() -> None:
    rank_header = RANK_HEADER.read_text(encoding="utf-8")
    mutant = rank_header.replace(
        "ScoreWeights weights_;",
        "std::span<const CandidateView> retained_;\n  ScoreWeights weights_;",
        1,
    )
    assert mutant != rank_header
    with pytest.raises(AssertionError):
        _validate_ownership_contract(
            mutant,
            RANK_CPP.read_text(encoding="utf-8"),
            FAILURE_HEADER.read_text(encoding="utf-8"),
            FAILURE_CPP.read_text(encoding="utf-8"),
        )


def test_failure_entry_retained_candidate_mutation_is_rejected() -> None:
    failure_header = FAILURE_HEADER.read_text(encoding="utf-8")
    entry = _body(failure_header, "struct Entry")
    mutant = failure_header.replace(
        entry, entry + "\n    CandidateView* retained_candidate;", 1
    )
    assert mutant != failure_header
    with pytest.raises(AssertionError):
        _validate_ownership_contract(
            RANK_HEADER.read_text(encoding="utf-8"),
            RANK_CPP.read_text(encoding="utf-8"),
            mutant,
            FAILURE_CPP.read_text(encoding="utf-8"),
        )


def test_display_id_tie_break_mutation_is_rejected() -> None:
    rank_cpp = RANK_CPP.read_text(encoding="utf-8")
    mutant = rank_cpp.replace(
        "const CandidateView& right = candidates[right_index];",
        (
            "const CandidateView& right = candidates[right_index];\n"
            "  if (left.id != right.id) { return left.id < right.id; }"
        ),
        1,
    )
    assert mutant != rank_cpp
    with pytest.raises(AssertionError):
        _validate_ownership_contract(
            RANK_HEADER.read_text(encoding="utf-8"),
            mutant,
            FAILURE_HEADER.read_text(encoding="utf-8"),
            FAILURE_CPP.read_text(encoding="utf-8"),
        )


def test_equivalent_forbidden_identifier_mutations_are_rejected() -> None:
    rank_header = RANK_HEADER.read_text(encoding="utf-8")
    rank_cpp = RANK_CPP.read_text(encoding="utf-8")
    failure_header = FAILURE_HEADER.read_text(encoding="utf-8")
    failure_cpp = FAILURE_CPP.read_text(encoding="utf-8")
    mutations = [
        (
            rank_header,
            rank_cpp + "\nconst auto display_probe = candidate_ptr->id;\n",
            failure_header,
            failure_cpp,
        ),
        (
            rank_header,
            rank_cpp + "\nconst auto member_probe = &CandidateView::frontier_id;\n",
            failure_header,
            failure_cpp,
        ),
        (
            rank_header,
            rank_cpp + "\nconst int map_version = 1;\n",
            failure_header,
            failure_cpp,
        ),
        (
            rank_header,
            rank_cpp,
            failure_header,
            failure_cpp + "\nconst int snapshot_revision = 1;\n",
        ),
        (
            rank_header,
            rank_cpp,
            failure_header.replace(
                "std::size_t total_patch_cells_{0U};",
                "std::size_t total_patch_cells_{0U};\n  double map_stamp_{0.0};",
                1,
            ),
            failure_cpp,
        ),
    ]
    for mutant in mutations:
        with pytest.raises(AssertionError):
            _validate_ownership_contract(*mutant)


@pytest.mark.parametrize(
    "call",
    ["detail::AccumulateCoverageState(", "detail::FinalizeCoverage("],
)
def test_removing_production_shared_call_is_rejected(call: str) -> None:
    public, cpp, detail = _current_sources()
    mutant = cpp.replace(call, "detail::RemovedSharedCall(", 1)
    assert mutant != cpp
    with pytest.raises(AssertionError):
        _validate(public, mutant, detail)


def test_counting_outside_task_as_an_ordinary_state_is_rejected() -> None:
    public, cpp, detail = _current_sources()
    accumulator = _body(detail, "inline void AccumulateCoverageState(")
    compact_accumulator = _compact(accumulator)
    outside = _case_body(
        compact_accumulator,
        "case CellState::kOutsideTask:",
        "case CellState::kOutsideMap:",
    )
    mutant_body = compact_accumulator.replace(
        outside, " CheckedIncrement(counts.outside_map); break; ", 1
    )
    assert mutant_body != accumulator
    mutant = detail.replace(accumulator, mutant_body, 1)
    with pytest.raises(AssertionError):
        _validate(public, cpp, mutant)


def test_deleting_positive_area_underflow_check_is_rejected() -> None:
    public, cpp, detail = _current_sources()
    checked = _body(detail, "inline double CheckedCoverageArea(")
    mutant_body = checked.replace("area <= 0.0", "false", 1)
    assert mutant_body != checked
    mutant = detail.replace(checked, mutant_body, 1)
    with pytest.raises(AssertionError):
        _validate(public, cpp, mutant)


def test_adding_a_coverage_ratio_gate_is_rejected() -> None:
    public, cpp, detail = _current_sources()
    calculate = _body(cpp, "CoverageStats CalculateCoverage(")
    mutant_body = calculate.replace(
        "return detail::FinalizeCoverage(",
        "const auto stats = detail::FinalizeCoverage(",
        1,
    ).replace(
        "raster.geometry().resolution, counts);",
        (
            "raster.geometry().resolution, counts);\n"
            "  if (stats.coverage_ratio >= 0.95) {}\n"
            "  return stats;"
        ),
        1,
    )
    assert mutant_body != calculate
    mutant = cpp.replace(calculate, mutant_body, 1)
    with pytest.raises(AssertionError):
        _validate(public, mutant, detail)


@pytest.mark.parametrize(
    "field",
    [
        "bool completed;",
        "bool success;",
        "int terminal_state;",
        "double completion_threshold;",
    ],
)
def test_completion_or_threshold_public_field_is_rejected(field: str) -> None:
    public, cpp, detail = _current_sources()
    stats = _body(public, "struct CoverageStats")
    mutant = public.replace(stats, stats + "\n  " + field, 1)
    with pytest.raises(AssertionError):
        _validate(mutant, cpp, detail)


def test_additional_stable_public_function_is_rejected() -> None:
    public, cpp, detail = _current_sources()
    mutant = public.replace(
        "CoverageStats CalculateCoverage(const TaskRaster& raster);",
        (
            "CoverageStats CalculateCoverage(const TaskRaster& raster);\n"
            "bool IsCoverageKnown(const CoverageStats& stats);"
        ),
        1,
    )
    with pytest.raises(AssertionError):
        _validate(mutant, cpp, detail)


def test_private_detail_header_is_not_installed_or_used_outside_owner() -> None:
    cmake = CMAKE.read_text(encoding="utf-8")
    assert "src/detail/coverage_statistics.hpp" not in cmake
    assert "install(DIRECTORY src" not in cmake

    users = []
    for path in PACKAGE.rglob("*"):
        if path.is_file() and path.suffix in {".hpp", ".cpp"}:
            source = path.read_text(encoding="utf-8")
            if '"detail/coverage_statistics.hpp"' in source:
                users.append(path.relative_to(PACKAGE).as_posix())
    assert sorted(users) == ["src/coverage.cpp", "test/test_coverage.cpp"]
