import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import lunar_drl_terrain_native as native
from beam_oracle import observe as oracle_observe
from lunar_drl_exploration.contracts import Pose, SensorSpec
from lunar_drl_exploration.sensor import visible_cells, target_visibility

PROVEN = np.array(
    [
        [1, 2, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 2, 1, 1],
        [1, 1, 1, 1, 2, 1, 1],
        [2, 1, 1, 1, 1, 1, 2],
        [1, 1, 1, 1, 1, 2, 1],
        [1, 1, 2, 2, 1, 1, 1],
        [1, 1, 1, 1, 1, 2, 1],
    ],
    np.uint8,
)


def native_observe(b, source, radius, yaw=0.0, fov=2 * math.pi):
    out = np.empty_like(b)
    native.observe(b, *source, radius, yaw, fov, out)
    return out != 0


@pytest.mark.parametrize(
    "fov,yaw,mount", [(360, 0, 0), (90, 0.173, 0.219), (17, -0.44, 0.39)]
)
def test_fixed_pose_self_closure_proven_and_seeded(fov, yaw, mount):
    for seed in range(9):
        b = (
            PROVEN
            if seed == 0
            else np.where(
                np.random.default_rng(seed).random((7, 7)) < 0.25, 2, 1
            ).astype(np.uint8)
        )
        b = b.copy()
        b[3, 1] = 1
        truth = native_observe(b, (1, 3), 10.0, yaw + mount, math.radians(fov))
        measured = np.where(truth, b, 0).astype(np.uint8)
        prediction = native_observe(
            measured, (1, 3), 10.0, yaw + mount, math.radians(fov)
        )
        assert np.array_equal(truth, prediction), (
            seed,
            np.argwhere(prediction & ~truth),
        )
        assert np.array_equal(
            truth, oracle_observe(b, (1, 3), 10.0, yaw + mount, math.radians(fov))
        )


@pytest.mark.parametrize("radius", [0.9, 1.0, math.sqrt(2), 2.5, 4.2])
def test_native_all_targets_and_eight_headings_match_independent_beams(radius):
    from lunar_drl_exploration.sensor import directional_visibility, source_visibility

    headings = np.arange(8) * math.pi / 4
    for seed in range(5):
        b = np.where(np.random.default_rng(seed).random((5, 6)) < 0.3, 2, 1).astype(
            np.uint8
        )
        source = (2, 2)
        b[2, 2] = 1
        targets = np.argwhere(np.ones_like(b))[:, ::-1].copy()
        expected = oracle_observe(b, source, radius)
        assert np.array_equal(native_observe(b, source, radius), expected)
        assert np.array_equal(
            target_visibility(b, source, targets, radius), expected.ravel()
        )
        source_expected = [oracle_observe(b, tuple(s), radius)[2, 2] for s in targets]
        assert np.array_equal(
            source_visibility(b, targets, source, radius), source_expected
        )
        for fov, mount in [(90, 0.0), (90, 0.173), (11, -0.211), (360, 0.3)]:
            got = directional_visibility(
                b, source, targets, radius, headings + mount, math.radians(fov)
            )
            for j, angle in enumerate(headings + mount):
                want = oracle_observe(b, source, radius, angle, math.radians(fov))
                assert np.array_equal(got[:, j], want.ravel())
                assert np.array_equal(
                    native_observe(b, source, radius, angle, math.radians(fov)), want
                )


def test_simultaneous_side_group_and_beam_fov_not_target_center():
    b = np.ones((5, 5), np.uint8)
    b[1, 2] = 2
    out = native_observe(b, (1, 1), 2.0, math.pi / 4, 0.01)
    assert out[1, 2] and out[2, 1] and not out[2, 2]
    assert np.array_equal(out, oracle_observe(b, (1, 1), 2.0, math.pi / 4, 0.01))
    b[2, 1] = 2
    assert np.array_equal(out, native_observe(b, (1, 1), 2.0, math.pi / 4, 0.01))


def test_map_clipping_keeps_nominal_beams_and_threadsafe_cache():
    b = PROVEN.copy()
    b[3, 0] = 1  # the clipped-boundary source must transmit

    def run(r):
        return native_observe(b, (0, 3), r, 0.32, 0.1)

    radii = [0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0] * 2
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, radii))
    for r, got in zip(radii, results):
        assert np.array_equal(got, oracle_observe(b, (0, 3), r, 0.32, 0.1))
    info = native.visibility_cache_info()
    assert info["templates"] <= 8
    assert info["bytes"] <= info["limit_bytes"] == 128 * 1024 * 1024


def test_optical_component_prefilter_preserves_exact_all_stance_union():
    from lunar_drl_exploration.sensor import optical_candidate_mask

    for seed in range(20):
        rng = np.random.default_rng(seed + 123)
        b = rng.choice([0, 1, 2], (7, 8), p=[0.2, 0.45, 0.35]).astype(np.uint8)
        r = ((b != 2) & (rng.random(b.shape) < 0.15)).astype(np.uint8)
        expected = np.zeros_like(b, bool)
        for y, x in np.argwhere(r):
            expected |= oracle_observe(b, (x, y), 3.5)
        candidate = optical_candidate_mask(b, r).astype(np.uint8)
        out = np.empty_like(b)
        native.visible_union(b, r, candidate, 3.5, out)
        assert np.array_equal(out, expected), seed


def test_small_closed_native_cave_actual_measurements_exhaust_task(
    native_producer_oracle,
):
    from lunar_drl_exploration.scene import Scene, TerrainGrid
    from lunar_drl_exploration.reference import CoverageReference
    from lunar_drl_exploration.task_analysis import TaskAnalyzer
    from test_task_analysis import snapshot
    from test_sensor_reference import producer_roundtrip

    scene = Scene(20260915, "cave", 40, 0.5)
    truth = TerrainGrid.from_scene(scene)
    pose = scene.initial_pose(truth)
    sensor = SensorSpec()
    reachable = truth.reachable(pose)
    raw = np.empty(truth.shape, np.uint8)
    native.visible_union(truth.intrinsic, reachable, np.ones_like(raw), 20.0, raw)
    hit = (raw != 0) & np.isfinite(truth.heights)
    measured = TerrainGrid.from_heights(
        np.where(hit, truth.heights, np.nan),
        0.5,
        truth.origin,
        scene.platform,
        stats=np.where(hit[:, :, None], truth.stats, np.nan),
    )
    b, m, observed = producer_roundtrip(native_producer_oracle, measured)
    assert np.array_equal(b, measured.intrinsic)
    assert np.array_equal(m, measured.navigation)
    assert np.array_equal(observed != 0, hit)
    assert np.array_equal(measured.reachable(pose), reachable)
    snap = snapshot(m, b, b, pose, 0.5, (*truth.origin, 0.0))
    report = TaskAnalyzer(scene.task, sensor).update(snap)
    reference = CoverageReference.build(truth, pose, scene.task, sensor)
    assert report.known_area_m2 == reference.area_m2
    assert report.available and report.exhausted
    assert not len(report.frontier_cells)


# Reuse the real producer executable fixture, not a second Python classifier.
from test_sensor_reference import native_producer_oracle


def test_exact_fov_boundary_across_negative_pi_keeps_left_axis_beam():
    b = np.ones((3, 3), np.uint8)
    yaw, fov = 5 * math.pi / 4, math.pi / 2
    expected = oracle_observe(b, (1, 1), 1.0, yaw, fov)
    assert expected[1, 0]
    assert np.array_equal(native_observe(b, (1, 1), 1.0, yaw, fov), expected)
