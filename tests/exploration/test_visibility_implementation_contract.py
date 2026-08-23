from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "ros2_ws/src/lunar_pure_exploration_core/src/information_gain.cpp"
DETAIL = ROOT / "ros2_ws/src/lunar_pure_exploration_core/src/detail/visibility_traversal.hpp"
EXPECTED_TRACE_BODY_SHA256 = (
    "e8ca1801dc2edc9fe32d98798a14608898b1d91f94f3cc306f70ea7379786db5"
)


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


def _validate(evaluate_source: str, detail_source: str) -> None:
    evaluate = _body(evaluate_source, "InformationGainEvaluator::Evaluate(")
    for call in (
        "ConsumeVisibilityWork(",
        "detail::TraceClosedSegment(",
        "ClassifyPreTargetContact(",
        "CheckedVisibleIncrement(",
    ):
        assert call in evaluate

    trace = _body(detail_source, "TraceSummary TraceClosedSegment(")
    compact = re.sub(r"\s+", " ", trace)
    for required in (
        "ConsumeVisibilityWork(",
        "visitor(",
        "ExactFromDouble(",
        "traversal_internal::CheckedMultiply(",
        "traversal_internal::GridUnitProductStride(",
        "Compare(x_event_product, y_event_product)",
        "CheckedAdd(x_event_product, x_product_stride)",
        "CheckedAdd(y_event_product, y_product_stride)",
        "std::unordered_set<TraceCell",
        "std::array<TraceCell, kMaximumTieGroupSize> group",
        "while (!stopped && (x_active || y_active))",
    ):
        assert required in compact
    assert "next_t_" not in compact
    assert "std::vector" not in compact
    assert ".reserve(" not in compact
    assert ".resize(" not in compact
    assert len(re.findall(r"\bwhile\s*\(", compact)) == 1
    assert re.search(r"\b(?:new|malloc|calloc|realloc)\b", compact) is None

    loop_headers = re.findall(r"for\s*\(([^)]*)\)", compact)
    assert loop_headers
    assert all("kMaximumTieGroupSize" in header for header in loop_headers)
    assert hashlib.sha256(compact.strip().encode()).hexdigest() == (
        EXPECTED_TRACE_BODY_SHA256
    )


def test_production_uses_streaming_visibility_seams() -> None:
    _validate(CPP.read_text(encoding="utf-8"), DETAIL.read_text(encoding="utf-8"))


@pytest.mark.parametrize("call", [
    "ConsumeVisibilityWork(",
    "detail::TraceClosedSegment(",
    "ClassifyPreTargetContact(",
    "CheckedVisibleIncrement(",
])
def test_missing_evaluate_seam_is_rejected(call: str) -> None:
    cpp = CPP.read_text(encoding="utf-8").replace(call, "removed_call(", 1)
    with pytest.raises(AssertionError):
        _validate(cpp, DETAIL.read_text(encoding="utf-8"))


def test_bbox_double_loop_mutant_is_rejected() -> None:
    detail = DETAIL.read_text(encoding="utf-8")
    marker = _body(detail, "TraceSummary TraceClosedSegment(")
    mutant = marker + "\nfor (int x=0; x<width; ++x) { for (int y=0; y<height; ++y) {} }"
    detail = detail.replace(marker, mutant, 1)
    with pytest.raises(AssertionError):
        _validate(CPP.read_text(encoding="utf-8"), detail)


def test_missing_trace_consume_mutant_is_rejected() -> None:
    detail = DETAIL.read_text(encoding="utf-8")
    trace = _body(detail, "TraceSummary TraceClosedSegment(")
    mutant = trace.replace("ConsumeVisibilityWork(", "removed_consume(")
    detail = detail.replace(trace, mutant, 1)
    with pytest.raises(AssertionError):
        _validate(CPP.read_text(encoding="utf-8"), detail)


@pytest.mark.parametrize("call", [
    "traversal_internal::CheckedMultiply(",
    "traversal_internal::GridUnitProductStride(",
    "Compare(x_event_product, y_event_product)",
    "CheckedAdd(x_event_product, x_product_stride)",
    "CheckedAdd(y_event_product, y_product_stride)",
])
def test_missing_exact_incremental_call_is_rejected(call: str) -> None:
    detail = DETAIL.read_text(encoding="utf-8")
    trace = _body(detail, "TraceSummary TraceClosedSegment(")
    mutant = trace.replace(call, "removed_exact_call(", 1)
    assert mutant != trace
    detail = detail.replace(trace, mutant, 1)
    with pytest.raises(AssertionError):
        _validate(CPP.read_text(encoding="utf-8"), detail)


@pytest.mark.parametrize("allocation", [
    "std::vector<GridIndex> bad(width * height);",
    "cells.reserve(x_extent * y_extent);",
    "cells.resize(width * height);",
    "const auto cell_count = width * height; std::vector<TraceCell> bad(cell_count);",
    "std::vector<TraceCell> bad(static_cast<std::size_t>(width) * height);",
])
def test_bbox_area_allocation_mutant_is_rejected(allocation: str) -> None:
    detail = DETAIL.read_text(encoding="utf-8")
    trace = _body(detail, "TraceSummary TraceClosedSegment(")
    detail = detail.replace(trace, trace + "\n" + allocation, 1)
    with pytest.raises(AssertionError):
        _validate(CPP.read_text(encoding="utf-8"), detail)


def test_equivalent_row_column_bbox_scan_mutant_is_rejected() -> None:
    detail = DETAIL.read_text(encoding="utf-8")
    trace = _body(detail, "TraceSummary TraceClosedSegment(")
    mutation = (
        "\nfor (int row=0; row<rows; ++row) {"
        " for (int column=0; column<columns; ++column) {} }"
    )
    detail = detail.replace(trace, trace + mutation, 1)
    with pytest.raises(AssertionError):
        _validate(CPP.read_text(encoding="utf-8"), detail)


@pytest.mark.parametrize("mutation", [
    "while (row < rows) { ++row; }",
    "while(row < rows) { ++row; }",
    "auto* bad = new TraceCell[width * height];",
    "auto* bad = malloc(width * height * sizeof(TraceCell));",
])
def test_unapproved_loop_and_raw_allocation_mutants_are_rejected(
    mutation: str,
) -> None:
    detail = DETAIL.read_text(encoding="utf-8")
    trace = _body(detail, "TraceSummary TraceClosedSegment(")
    detail = detail.replace(trace, trace + "\n" + mutation, 1)
    with pytest.raises(AssertionError):
        _validate(CPP.read_text(encoding="utf-8"), detail)


@pytest.mark.parametrize("mutation", [
    "std::unordered_set<TraceCell> bad(width * height);",
    (
        "std::for_each(cells.begin(), cells.end(), "
        "[](const auto& cell) { classify(cell); });"
    ),
    "auto bad = std::make_unique<TraceCell[]>(width * height);",
])
def test_semantically_equivalent_area_work_mutants_are_rejected(
    mutation: str,
) -> None:
    detail = DETAIL.read_text(encoding="utf-8")
    trace = _body(detail, "TraceSummary TraceClosedSegment(")
    detail = detail.replace(trace, trace + "\n" + mutation, 1)
    with pytest.raises(AssertionError):
        _validate(CPP.read_text(encoding="utf-8"), detail)
