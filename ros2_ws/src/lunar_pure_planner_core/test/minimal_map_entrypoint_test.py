"""Source boundary checks for the minimal map snapshot entrypoints."""

from pathlib import Path
import re


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MAP_SNAPSHOT_HEADER = PACKAGE_ROOT / "src" / "shared" / "map_snapshot.hpp"
MAP_SNAPSHOT_SOURCE = PACKAGE_ROOT / "src" / "shared" / "map_snapshot.cpp"
GLOBAL_PROJECTION_FILES = (
    PACKAGE_ROOT / "src" / "shared" / "global_occupancy_projection.hpp",
    PACKAGE_ROOT / "src" / "shared" / "global_occupancy_projection.cpp",
)


def test_legacy_copy_entrypoint_is_removed() -> None:
    text = MAP_SNAPSHOT_HEADER.read_text(encoding="utf-8")
    assert not re.search(
        r"static\s+MapSnapshotBuildResult\s+Create\(\s*const\s+GridMap&"
        r"\s+\w+\s*\)",
        text,
    )
    source = MAP_SNAPSHOT_SOURCE.read_text(encoding="utf-8")
    assert not re.search(
        r"MapSnapshot::Create\(\s*const\s+GridMap&\s+\w+\s*\)",
        source,
    )


def test_minimal_production_entrypoints_use_map_contract_overload() -> None:
    for path in (MAP_SNAPSHOT_SOURCE, *GLOBAL_PROJECTION_FILES):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"MapSnapshot::Create\(([^;]*?)\);", text, re.DOTALL):
            expression = match.group(1)
            if "const GridMap&" in expression or "GridMap map" in expression:
                continue
            assert "MapContract::" in expression, (
                f"unqualified map snapshot construction in {path.name}: {expression}"
            )


def test_elevation_sampling_does_not_consult_valid_mask() -> None:
    text = MAP_SNAPSHOT_SOURCE.read_text(encoding="utf-8")
    sample_start = text.index("MapSnapshot::SampleElevationBilinear")
    sample_end = text.index("Vec3 MapSnapshot::CellCenter", sample_start)
    assert "valid_mask" not in text[sample_start:sample_end]
