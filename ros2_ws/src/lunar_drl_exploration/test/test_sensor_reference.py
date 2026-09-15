"""Independent exhaustive center-ray oracle and effective measurement parity."""

import math
import numpy as np
import pytest
from lunar_drl_exploration.config import load_platform_config
from lunar_drl_exploration.contracts import Pose, SensorSpec, TaskSpec
from lunar_drl_exploration.scene import TerrainGrid
from lunar_drl_exploration.sensor import SensorModel
from lunar_drl_exploration.reference import CoverageReference
from lunar_drl_exploration.geometry import world_to_cell, polygon_mask


def oracle_line(b, source, target):
    # Independent continuous segment versus every closed cell square. Ties at
    # corners touch both cells; the first-hit target itself is not an occluder.
    sy, sx = source
    ty, tx = target
    for y, x in np.argwhere(b == 2):
        if (y, x) in (source, target):
            continue
        lo, hi = 0.0, 1.0
        for a, d, lower in ((sx + 0.5, tx - sx, x), (sy + 0.5, ty - sy, y)):
            if d == 0:
                if not lower <= a <= lower + 1:
                    hi = -1
                    break
            else:
                p, q = (lower - a) / d, (lower + 1 - a) / d
                lo, hi = max(lo, min(p, q)), min(hi, max(p, q))
        if lo <= hi and hi > 0 and lo < 1:
            return False
    return True


def oracle_reachable(m, start):
    result = {start} if m[start] == 1 else set()
    pending = list(result)
    while pending:
        y, x = pending.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = y + dy, x + dx
                if (
                    q not in result
                    and 0 <= q[0] < m.shape[0]
                    and 0 <= q[1] < m.shape[1]
                    and m[q] == 1
                    and m[y, q[1]] == 1
                    and m[q[0], x] == 1
                ):
                    result.add(q)
                    pending.append(q)
    return result


def oracle_union(terrain, start, sensor):
    r = oracle_reachable(terrain.navigation, start)
    result = np.zeros(terrain.shape, bool)
    for source in r:
        for target in np.ndindex(terrain.shape):
            if (
                math.dist(source, target) * terrain.resolution_m
                <= sensor.range_m + 1e-10
                and np.isfinite(terrain.heights[target])
                and terrain.intrinsic[target] != 0
                and oracle_line(terrain.intrinsic, source, target)
            ):
                result[target] = True
    return result


def terrain_fixture():
    h = np.zeros((14, 16), np.float32)
    h[2:12, 8] = 2.0
    h[6:8, 8] = 0.0  # visible opening too narrow for the footprint
    return TerrainGrid.from_heights(h, 0.5, (-3.17, -2.39), load_platform_config())


def task_all(t):
    x, y = t.origin
    h, w = t.shape
    r = t.resolution_m
    return TaskSpec(
        "all", "map", [[x, y], [x + w * r, y], [x + w * r, y + h * r], [x, y + h * r]]
    )


def test_native_union_matches_independent_exhaustive_and_each_heading():
    t = terrain_fixture()
    p = Pose(*t.cell_center(3, 6), 0.173)
    s = SensorSpec(range_m=4.0, fov_deg=90)
    reference = CoverageReference.build(t, p, task_all(t), s)
    expected = oracle_union(t, (6, 3), s)
    assert np.array_equal(reference.mask(), expected)
    union = np.zeros(t.shape, bool)
    for y, x in oracle_reachable(t.navigation, (6, 3)):
        for yaw in (
            0.173,
            0.173 + math.pi / 2,
            0.173 + math.pi,
            0.173 + 3 * math.pi / 2,
        ):
            measurement = SensorModel.observe(t, Pose(*t.cell_center(x, y), yaw), s)
            classified = t.intrinsic[measurement.rows, measurement.cols] != 0
            assert np.all(
                reference.mask()[
                    measurement.rows[classified], measurement.cols[classified]
                ]
            )
            union |= measurement.mask & (t.intrinsic != 0)
    assert np.array_equal(union, expected)
    assert np.any(expected & (t.navigation != 1) & (t.intrinsic == 1))
    assert np.any(expected & (t.intrinsic == 2))
    assert np.any(~expected & (t.intrinsic == 2))


def test_origin_fractional_floor_actual_yaw_translation_rejected():
    t = terrain_fixture()
    x, y = t.cell_center(3, 6)
    assert world_to_cell(t.origin[0], t.origin[1], t.origin, t.resolution_m) == (0, 0)
    assert world_to_cell(t.origin[0] - 0.01, t.origin[1] - 0.01, t.origin, 0.5) == (
        -1,
        -1,
    )
    s = SensorSpec(range_m=3, fov_deg=20, offset_yaw_rad=0.1)
    a = SensorModel.observe(t, Pose(x - 0.2, y + 0.2, 0.25), s)
    b = SensorModel.observe(t, Pose(x, y, 0.25), s)
    assert np.array_equal(a.mask, b.mask)
    c = SensorModel.observe(t, Pose(x, y, 1.25), s)
    assert not np.array_equal(a.mask, c.mask)
    with pytest.raises(ValueError, match="translation"):
        SensorModel.observe(t, Pose(x, y, 0), SensorSpec(offset_x_m=0.01))
    with pytest.raises(ValueError, match="translation"):
        CoverageReference.build(
            t, Pose(x, y, 0), task_all(t), SensorSpec(offset_y_m=0.01)
        )


def test_first_hit_exact_corner_and_range():
    t = terrain_fixture()
    # Pick native-produced blockers, then independently check all center rays.
    p = Pose(*t.cell_center(3, 6), 0)
    got = SensorModel.observe(t, p, SensorSpec(range_m=4, fov_deg=360)).mask
    want = np.array(
        [
            [
                math.dist((6, 3), (y, x)) * 0.5 <= 4 + 1e-10
                and oracle_line(t.intrinsic, (6, 3), (y, x))
                for x in range(t.shape[1])
            ]
            for y in range(t.shape[0])
        ]
    )
    assert np.array_equal(got, want)


def test_external_stance_sensor_rebuild_packed_intersection_and_zero_area():
    t = terrain_fixture()
    p = Pose(*t.cell_center(3, 6), 0)
    x, y = t.cell_center(5, 5)
    task = TaskSpec("inner", "map", [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]])
    a = CoverageReference.build(t, p, task, SensorSpec(range_m=1))
    b = CoverageReference.build(t, p, task, SensorSpec(range_m=4))
    assert a.reference_id != b.reference_id
    assert b.area_m2 > 0
    assert b.covered_area(b.packed_mask) == b.area_m2
    assert np.array_equal(b.unpack(b.pack(b.mask())), b.mask())
    empty = TaskSpec("outside", "map", [[100, 100], [101, 100], [101, 101], [100, 101]])
    z = CoverageReference.build(t, p, empty, SensorSpec())
    assert z.area_m2 == 0 and z.coverage_ratio(z.packed_mask) == 0


def test_effective_stats_float32_roundtrip_and_hidden_neighbors():
    t = terrain_fixture()
    p = Pose(*t.cell_center(3, 6), 0)
    o = SensorModel.observe(t, p, SensorSpec(range_m=2, fov_deg=30))
    assert o.stats.dtype == np.float32 and o.heights.dtype == np.float32
    partial = np.full(t.shape, np.nan, np.float32)
    partial[o.rows, o.cols] = o.heights
    stats = np.full((*t.shape, 4), np.nan, np.float32)
    stats[o.rows, o.cols] = o.stats
    measured = TerrainGrid.from_heights(
        partial, t.resolution_m, t.origin, load_platform_config(), stats=stats
    )
    assert np.array_equal(
        measured.intrinsic[o.rows, o.cols], t.intrinsic[o.rows, o.cols]
    )
    assert np.count_nonzero(np.isfinite(partial)) == len(o.indices)
    assert not np.any(
        measured.navigation == 1
    )  # thin partial FOV cannot support footprint


@pytest.mark.parametrize(
    "field,key",
    [
        (0, "maximum_slope_rad"),
        (1, "maximum_local_obstacle_relief_m"),
        (2, "minimum_underbody_clearance_m"),
    ],
)
def test_threshold_statistics_quantized_before_truth_classification(
    field, key, native_producer_oracle
):
    platform = load_platform_config()
    limit = platform.capability[key]
    values = [
        np.nextafter(np.float32(limit), np.float32(-np.inf)),
        np.float32(limit),
        np.nextafter(np.float32(limit), np.float32(np.inf)),
    ]
    for value in values:
        stats = np.zeros((9, 9, 4), np.float64)
        stats[..., 3] = 1
        stats[..., field] = float(value) + 1e-13
        t = TerrainGrid.from_heights(
            np.zeros((9, 9), np.float32), 0.5, (-0.13, -0.27), platform, stats=stats
        )
        assert t.intrinsic[4, 4] == (
            2 if float(np.float32(stats[4, 4, field])) > limit else 1
        )
        b, m, observed = producer_roundtrip(native_producer_oracle, t)
        assert np.array_equal(b, t.intrinsic) and np.array_equal(m, t.navigation)
        assert np.all(observed)


def test_polygon_concave_closed_boundary():
    polygon = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], np.float32)
    got = polygon_mask((4, 4), (-0.5, -0.5), 1.0, polygon)
    assert got[1, 1] and got[0, 3] and not got[2, 2]


def test_shared_measured_visibility_helper_and_bounded_observation():
    from lunar_drl_exploration.sensor import visible_cells

    t = terrain_fixture()
    p = Pose(*t.cell_center(3, 6), 0.24)
    s = SensorSpec(range_m=2)
    rows, cols = visible_cells(t.intrinsic, t.origin, t.resolution_m, p, s)
    observed = SensorModel.observe(t, p, s)
    assert np.array_equal(rows, observed.rows) and np.array_equal(cols, observed.cols)
    # Same local measured patch translated into a much larger map: no change
    # in range, FOV or blockers. Returned coordinates remain global grid cells.
    large = np.zeros((1000, 1000), np.uint8)
    large[400:414, 300:316] = t.intrinsic
    origin = (t.origin[0] - 150, t.origin[1] - 200)
    rr, cc = visible_cells(large, origin, 0.5, p, s)
    assert np.array_equal(rr - 400, rows) and np.array_equal(cc - 300, cols)


def test_exact_corner_cannot_cut_blocker_and_disconnected_free_is_not_blocked():
    import lunar_drl_terrain_native as native

    b = np.zeros((5, 5), np.uint8)
    b[1, 2] = 2
    out = np.empty_like(b)
    native.observe(b, 1, 1, 4.0, math.pi / 4, math.pi * 2, out)
    assert out[1, 2] and not out[2, 2]  # side cell hit at the exact shared corner
    m = np.zeros((4, 4), np.uint8)
    m[1, 1] = 1
    m[2, 2] = 1
    r = np.empty_like(m)
    native.reachable(m, 1, 1, r)
    assert r[1, 1] and not r[2, 2] and m[2, 2] == 1


def test_interior_viewpoints_needed_ring_only_reference_fails():
    # Closed room. Its interior targets are visible from interior legal stances,
    # while a stance ring at the room's outer boundary is physically illegal.
    h = np.zeros((21, 21), np.float32)
    h[1:3, :] = 2
    h[-3:-1, :] = 2
    h[:, 1:3] = 2
    h[:, -3:-1] = 2
    t = TerrainGrid.from_heights(h, 0.5, (-5.25, -5.25), load_platform_config())
    p = Pose(*t.cell_center(10, 10), 0)
    s = SensorSpec(range_m=1, fov_deg=360)
    reference = CoverageReference.build(t, p, task_all(t), s)
    rr = oracle_reachable(t.navigation, (10, 10))
    ring = {
        q
        for q in rr
        if any(
            (q[0] + dy, q[1] + dx) not in rr
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
        )
    }
    ring_union = np.zeros(t.shape, bool)
    for y, x in ring:
        ring_union |= SensorModel.observe(t, Pose(*t.cell_center(x, y), 0), s).mask
    assert reference.mask()[10, 10] and not ring_union[10, 10]


def test_only_external_legal_stances_can_see_small_target():
    t = terrain_fixture()
    p = Pose(*t.cell_center(3, 6), 0)
    # Select a visible intrinsically free yet nonstandable opening cell.
    observed = SensorModel.observe(t, p, SensorSpec(range_m=4, fov_deg=360)).mask
    yy, xx = np.argwhere(observed & (t.intrinsic == 1) & (t.navigation != 1))[0]
    x, y = t.cell_center(int(xx), int(yy))
    r = t.resolution_m * 0.49
    task = TaskSpec(
        "nonstandable",
        "map",
        [[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]],
    )
    ref = CoverageReference.build(t, p, task, SensorSpec(range_m=4))
    assert ref.area_m2 == t.resolution_m**2
    assert not np.any(
        (t.navigation == 1)
        & polygon_mask(t.shape, t.origin, t.resolution_m, task.polygon)
    )


@pytest.fixture(scope="module")
def native_producer_oracle(tmp_path_factory):
    import subprocess
    from pathlib import Path
    import lunar_drl_terrain_native  # library mapped before inspecting its path

    root = Path(__file__).resolve().parents[1]
    library = next(
        Path(line.split()[-1])
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "liblunar_incremental_navigation_core.so" in line
    )
    executable = tmp_path_factory.mktemp("terrain-oracle") / "roundtrip"
    subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-O2",
            str(root / "test/native_roundtrip.cpp"),
            "-I" + str(root.parent / "lunar_incremental_navigation_core/include"),
            "-L" + str(library.parent),
            "-Wl,-rpath," + str(library.parent),
            "-llunar_incremental_navigation_core",
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def producer_roundtrip(executable, t):
    import struct, subprocess

    c = load_platform_config().capability
    by, bx = np.argwhere(np.isfinite(t.heights))[0]
    header = struct.pack(
        "=4i7d",
        t.shape[1],
        t.shape[0],
        int(bx),
        int(by),
        t.resolution_m,
        *t.origin,
        c["maximum_slope_rad"],
        c["maximum_local_obstacle_relief_m"],
        c["minimum_underbody_clearance_m"],
        c["minimum_clearance_m"]
    )
    header += np.asarray(c["footprint_xy_m"], dtype=np.float64).tobytes()
    result = subprocess.run(
        [str(executable)],
        input=header + t.heights.tobytes() + t.stats.tobytes(),
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, np.uint8).reshape(3, *t.shape)


def test_actual_persistent_producer_roundtrip_fractional_origin_partial_fov(
    native_producer_oracle,
):
    truth = terrain_fixture()
    p = Pose(*truth.cell_center(3, 6), 0)
    for fov in (20, 90, 360):
        o = SensorModel.observe(truth, p, SensorSpec(range_m=3, fov_deg=fov))
        heights = np.full(truth.shape, np.nan, np.float32)
        heights[o.rows, o.cols] = o.heights
        stats = np.full((*truth.shape, 4), np.nan, np.float32)
        stats[o.rows, o.cols] = o.stats
        partial = TerrainGrid.from_heights(
            heights, 0.5, truth.origin, load_platform_config(), stats=stats
        )
        b, m, observed = producer_roundtrip(native_producer_oracle, partial)
        assert np.array_equal(b, partial.intrinsic)
        assert np.array_equal(m, partial.navigation)
        assert np.array_equal(observed, np.isfinite(heights))
        assert np.array_equal(b[o.rows, o.cols], truth.intrinsic[o.rows, o.cols])


@pytest.mark.parametrize("seed", range(8))
def test_random_small_native_union_independent_ray_and_reachability_oracle(seed):
    import lunar_drl_terrain_native as native

    rng = np.random.default_rng(seed)
    b = np.asarray(rng.random((6, 7)) < 0.2, np.uint8) * 2
    b[2, 2] = 0
    m = np.asarray(b == 0, np.uint8)
    rr = np.empty_like(m)
    native.reachable(m, 2, 2, rr)
    sources = oracle_reachable(m, (2, 2))
    assert np.count_nonzero(rr) == len(sources)
    output = np.empty_like(m)
    native.visible_union(b, rr, np.ones_like(m), 2.5, output)
    want = np.zeros(m.shape, bool)
    for source in sources:
        for target in np.ndindex(m.shape):
            if math.dist(source, target) <= 2.5 and oracle_line(b, source, target):
                want[target] = True
    assert np.array_equal(output, want)


def test_deployment_geometry_import_does_not_construct_privileged_modules():
    import subprocess, sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lunar_drl_exploration.sensor; "
            'assert "lunar_drl_exploration.scene" not in sys.modules; '
            'assert "lunar_drl_exploration.reference" not in sys.modules',
        ],
        check=True,
    )


def test_fractional_lattice_exact_corners_and_neighboring_cells():
    origin = (-3.125, -2.375)
    r = 0.25
    for y in range(-3, 7):
        for x in range(-3, 7):
            wx, wy = origin[0] + x * r, origin[1] + y * r
            assert world_to_cell(wx, wy, origin, r) == (x, y)
            assert world_to_cell(wx - 0.001, wy - 0.001, origin, r) == (x - 1, y - 1)


@pytest.mark.parametrize("interior_hole", [False, True])
def test_reference_counts_classified_centers_not_finite_partial_measurements(
    interior_hole,
):
    heights = np.zeros((13, 13) if interior_hole else (9, 9), np.float32)
    if interior_hole:
        heights[8, 8] = np.nan
    terrain = TerrainGrid.from_heights(
        heights, 0.5, (-0.375, -0.125), load_platform_config()
    )
    pose = Pose(*terrain.cell_center(3, 3), 0)
    sensor = SensorSpec(range_m=100, fov_deg=360)
    reference = CoverageReference.build(terrain, pose, task_all(terrain), sensor)
    measurements = SensorModel.observe(terrain, pose, sensor)
    classified = terrain.intrinsic != 0
    assert np.any(measurements.mask & ~classified)  # retain real partial raw centers
    assert np.array_equal(reference.mask(), classified)
    assert reference.covered_area(reference.pack(measurements.mask & ~classified)) == 0
    assert (
        reference.covered_area(reference.pack(measurements.mask)) == reference.area_m2
    )
    if interior_hole:
        assert terrain.intrinsic[8, 7] == 0 and np.isfinite(terrain.heights[8, 7])
        assert not reference.mask()[8, 7] and not measurements.mask[8, 8]
    else:
        assert np.count_nonzero(reference.mask()) == 49
        assert reference.area_m2 == 49 * 0.25
