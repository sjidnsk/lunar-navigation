# Isaac→ROS Action Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an external, deterministic Isaac Sim→ROS 2 Humble snapshot bridge, three proxy capability packages, and six isolated `/plan_motion` positive/negative Action regressions.

**Architecture:** Isaac Sim 6.0.1 executes one read-only collector in its Python 3.12 process and atomically writes a ROS-neutral JSON/NPZ snapshot. A standalone external ROS 2 Humble package running on system Python 3.10 validates and publishes that snapshot, qualifies immutable scene-backed fixtures, starts a fresh Lifecycle planner for each case, validates each Action result, and emits JSON/JUnit evidence.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, Isaac Sim 6.0.1 Python 3.12, `pxr.Usd/UsdGeom/UsdPhysics`, NumPy, `rclpy`, `grid_map_msgs`, ROS 2 Lifecycle and Action APIs, pytest, colcon.

## Global Constraints

- External implementation root is exactly `/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression`; no implementation, build product, snapshot, log, or report enters `/mnt/data/WS/lunar-navigation`.
- The external implementation root is a standalone local Git repository with no remote. Never nest, link, vendor, or import it into the main repository.
- Preserve the current active USD byte-for-byte: no stage open, timeline play, prim edit, layer save, or derivative USD export.
- Isaac Sim work runs in Python 3.12 through loopback `isaacsim.code_editor.python_server`; ROS work runs only after sourcing `/opt/ros/humble/setup.bash` and confirming system Python 3.10.
- Never load the project’s Python 3.10 ROS interface modules into Isaac Sim Python 3.12.
- Use the existing absolute `/plan_motion` Action and `/lunar_planner` Lifecycle node unchanged; execute six fresh planner sessions sequentially.
- Publish the exact ten required GridMap layers and the six frozen external topics with their existing frames and QoS.
- Capability files remain `proxy-v1` validation inputs and do not become production capability claims or external interface baselines.
- Do not use Nav2, controllers, execution feedback, `/clock`, synthetic hazards, downloaded assets, broad `pkill`, recursive deletion, or automatic Isaac restart.
- Runtime directories are unique per run under external `build/`, `install/`, `log/`, `snapshots/`, and `artifacts/`; do not delete earlier runs.
- Preserve every unrelated main-repository working-tree entry present at execution start, including `.vscode/`; never restore, stage, or alter one for this task.
- The 2026-08-04 approved semantic authority is main commit `a02601a8112f3417f57de9fb7df4d3cf2237b705`; production planner source remains read-only.
- Replace the reviewed external `lunar-scenario-lock/v1` only through the explicit rebaseline path; the replacement is `lunar-scenario-lock/v2`, while the final USD and snapshot hashes remain unchanged.
- Wheel and legged Action requests retain the continuous platform pose, but their trajectory-start oracle uses the independently recomputed local-grid projection. Hopper-positive alone uses a `0.75 m` point-goal tolerance; all other cases retain `0.50 m`.

Command convention: define `LUNAR_VALIDATION_ROOT=/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression` and `LUNAR_PACKAGE_ROOT=$LUNAR_VALIDATION_ROOT/ros2_ws/src/lunar_isaac_validation` at the start of execution. Relative `test/` and `lunar_isaac_validation/` paths run from `LUNAR_PACKAGE_ROOT`; relative `scripts/`, `isaac/`, `snapshots/`, and `artifacts/` paths run from `LUNAR_VALIDATION_ROOT`.

---

## File Structure

Main repository:

- Create: `docs/superpowers/plans/2026-08-03-isaac-ros-action-regression.md` — this plan only.
- Preserve: `docs/superpowers/specs/2026-08-03-isaac-ros-action-regression-design.md` — approved design authority.
- Preserve: all ROS source and interface files — the harness consumes them without modification.

External standalone Git root:

- Create: `.gitignore` — excludes tokens and every runtime/build artifact.
- Create: `README.md` — exact build, collect, qualify, run, and evidence commands.
- Create: `isaac/collect_stage_snapshot.py` — read-only live-stage extraction entry point.
- Create: `scripts/preflight.py` — environment, Git, stage-server, and path checks.
- Create: `scripts/generate_proxy_meshes.py` — deterministic STL generator.
- Create: `scripts/build_external.sh` — unique external colcon build.
- Create: `scripts/collect_snapshot.py` — Python Server client entry point.
- Create: `scripts/qualify_fixtures.sh` — explicit scenario-lock creation entry point.
- Create: `scripts/run_action_regression.sh` — formal six-case entry point.
- Create: `ros2_ws/src/lunar_isaac_validation/package.xml` — ament package metadata.
- Create: `ros2_ws/src/lunar_isaac_validation/setup.py` — package install and console entry points.
- Create: `ros2_ws/src/lunar_isaac_validation/setup.cfg` — ROS executable install location.
- Create: `ros2_ws/src/lunar_isaac_validation/resource/lunar_isaac_validation` — ament index marker.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/constants.py` — schemas, layer names, platform ids, and exit codes.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/snapshot_contract.py` — dataclasses, validation, hashing, and atomic snapshot IO.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/rasterizer.py` — deterministic terrain and collision rasterization.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/python_server_client.py` — authenticated JSON-envelope TCP client.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/capability_contract.py` — capability/provenance validation.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/scenario_qualifier.py` — map-only candidate selection and immutable lock writer.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/planner_semantics.py` — action-independent grid projection shared by qualification and assertions.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/grid_map_codec.py` — GridMap message layout encoder.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/bridge_node.py` — six-topic ROS publisher and readiness service.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/trajectory_checks.py` — wheel and legged reference checks.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/hop_checks.py` — hopper ballistic checks.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/action_assertions.py` — exact outcome/result dispatch.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/process_manager.py` — explicit managed-child lifecycle.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/reports.py` — JSON/JUnit evidence.
- Create: `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/regression_runner.py` — Lifecycle and Action orchestration.
- Create: `ros2_ws/src/lunar_isaac_validation/config/capabilities/{wheeled,legged,hopper}.yaml` — three proxy capability sources.
- Create: `ros2_ws/src/lunar_isaac_validation/config/observation.json` — shared observation capability.
- Create: `ros2_ws/src/lunar_isaac_validation/config/capability_provenance.json` — source class for every field.
- Create: `ros2_ws/src/lunar_isaac_validation/config/planner/{wheeled,legged,hopper}.yaml` — exact Lifecycle parameters.
- Create: `ros2_ws/src/lunar_isaac_validation/urdf/{wheeled,legged,hopper}_proxy.urdf` — loader geometry documents.
- Create: `ros2_ws/src/lunar_isaac_validation/meshes/{wheeled,legged,hopper}_proxy_collision.stl` — deterministic proxy collision meshes.
- Create during qualification: `ros2_ws/src/lunar_isaac_validation/scenarios/scenario_lock.json` — reviewed immutable fixture lock.
- Create: focused tests under `ros2_ws/src/lunar_isaac_validation/test/` matching each Python module.

---

### Task 1: Establish the external Git root and ROS package contract

**Files:**
- Create: external `.gitignore`, `README.md`
- Create: external `ros2_ws/src/lunar_isaac_validation/{package.xml,setup.py,setup.cfg,resource/lunar_isaac_validation}`
- Create: external `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/{__init__.py,constants.py}`
- Test: external `ros2_ws/src/lunar_isaac_validation/test/test_constants.py`

**Interfaces:**
- Consumes: approved design path and exact external root.
- Produces: importable `lunar_isaac_validation.constants` with `REQUIRED_LAYERS`, platform ids, schema strings, topic names, and exit codes.

- [ ] **Step 1: Initialize only the external standalone Git root**

Run:

```bash
mkdir -p /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression init --initial-branch=main
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression remote -v
```

Expected: an empty local repository and no remotes. Do not run `git init` anywhere below or above the external root.

- [ ] **Step 2: Write the failing constants contract test**

```python
from lunar_isaac_validation.constants import (
    EXIT_ACTION,
    EXIT_ENVIRONMENT,
    PLATFORM_IDS,
    REQUIRED_LAYERS,
    SNAPSHOT_SCHEMA,
)


def test_frozen_contract_values() -> None:
    assert SNAPSHOT_SCHEMA == "isaac-ros-planning-snapshot/v1"
    assert REQUIRED_LAYERS == (
        "elevation", "valid_mask", "obstacle", "obstacle_height",
        "observation_age_s", "observation_quality",
        "elevation_variance", "obstacle_variance",
        "observation_count", "forbidden",
    )
    assert PLATFORM_IDS == {
        "wheel": "proxy-wheeled-scout-v1",
        "legged": "proxy-legged-scout-v1",
        "hopper": "proxy-hopper-scout-v1",
    }
    assert EXIT_ENVIRONMENT == 10
    assert EXIT_ACTION == 50
```

- [ ] **Step 3: Run the test to verify the package is absent**

Run:

```bash
cd /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/ros2_ws/src/lunar_isaac_validation
python3 -m pytest -q test/test_constants.py
```

Expected: FAIL with `ModuleNotFoundError: lunar_isaac_validation`.

- [ ] **Step 4: Create package metadata, ignore rules, and exact constants**

`.gitignore` must contain:

```gitignore
build/
install/
log/
artifacts/
snapshots/
*.token
*.secret
__pycache__/
*.py[cod]
.pytest_cache/
```

`constants.py` must define:

```python
SNAPSHOT_SCHEMA = "isaac-ros-planning-snapshot/v1"
SCENARIO_SCHEMA = "isaac-ros-action-scenarios/v1"
REQUIRED_LAYERS = (
    "elevation", "valid_mask", "obstacle", "obstacle_height",
    "observation_age_s", "observation_quality", "elevation_variance",
    "obstacle_variance", "observation_count", "forbidden",
)
PLATFORM_IDS = {
    "wheel": "proxy-wheeled-scout-v1",
    "legged": "proxy-legged-scout-v1",
    "hopper": "proxy-hopper-scout-v1",
}
PLATFORM_PRIMS = {
    "wheel": "/World/LunarExplorationPlatforms/WheeledScout",
    "legged": "/World/LunarExplorationPlatforms/LeggedScout",
    "hopper": "/World/LunarExplorationPlatforms/HopperScout",
}
TOPICS = {
    "global_map": "/environment/map_global",
    "local_map": "/environment/map_local",
    "odometry": "/localization/odometry",
    "localization_status": "/localization/status",
    "tf": "/tf",
    "mission": "/mission/exploration_task",
}
EXIT_ENVIRONMENT, EXIT_ISAAC, EXIT_FIXTURE = 10, 20, 30
EXIT_ROS, EXIT_ACTION, EXIT_CLEANUP = 40, 50, 60
```

`package.xml` must use `ament_python`, declare runtime dependencies on `rclpy`, `action_msgs`, `geometry_msgs`, `grid_map_msgs`, `lifecycle_msgs`, `lunar_navigation_msgs`, `lunar_planning_msgs`, `nav_msgs`, `std_srvs`, `tf2_msgs`, `trajectory_msgs`, `python3-numpy`, and `python3-yaml`, and declare `python3-pytest` as a test dependency. `setup.py` must install package metadata plus `config`, `urdf`, `meshes`, and any existing `scenarios/*.json`; do not install runtime artifacts.

The initial README must state that the harness is validation-only, lives outside the main repository, does not command platform motion, and will gain executable commands task-by-task; it must link the approved design by absolute local path.

- [ ] **Step 5: Run the constants test and metadata sanity checks**

Run:

```bash
python3 -m pytest -q test/test_constants.py
python3 setup.py --name
```

Expected: PASS and `lunar_isaac_validation`.

- [ ] **Step 6: Commit the external package contract**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add .gitignore README.md ros2_ws
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "chore: scaffold Isaac ROS validation harness"
```

Expected: only external source files are committed; no main-repository path appears in the commit.

---

### Task 2: Implement the neutral snapshot contract and atomic IO

**Files:**
- Create: external `lunar_isaac_validation/snapshot_contract.py`
- Test: external `test/test_snapshot_contract.py`

**Interfaces:**
- Consumes: `SNAPSHOT_SCHEMA`, `REQUIRED_LAYERS`.
- Produces: `GridDescriptor`, `PlatformSnapshot`, `SnapshotBundle`, `sha256_file(path)`, `write_snapshot_atomic(output_dir, manifest, arrays)`, and `load_snapshot(manifest_path)`.

- [ ] **Step 1: Write failing round-trip, corruption, and partial-write tests**

```python
def test_snapshot_round_trip_rejects_hash_corruption(tmp_path: Path) -> None:
    arrays = make_valid_arrays(width=2, height=2)
    manifest = make_valid_manifest(width=2, height=2)
    manifest_path = write_snapshot_atomic(tmp_path, manifest, arrays)
    bundle = load_snapshot(manifest_path)
    assert bundle.manifest["schema_version"] == SNAPSHOT_SCHEMA
    assert bundle.arrays["global__elevation"].shape == (2, 2)

    with bundle.arrays_path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(SnapshotContractError, match="SNAPSHOT_ARRAY_HASH_MISMATCH"):
        load_snapshot(manifest_path)


def test_atomic_writer_leaves_no_visible_manifest_on_failure(tmp_path: Path) -> None:
    with pytest.raises(SnapshotContractError, match="SNAPSHOT_LAYER_MISSING"):
        write_snapshot_atomic(tmp_path, make_valid_manifest(), {})
    assert not (tmp_path / "snapshot_manifest.json").exists()
```

- [ ] **Step 2: Run the tests and verify missing symbols fail**

Run: `python3 -m pytest -q test/test_snapshot_contract.py`

Expected: FAIL importing `snapshot_contract`.

- [ ] **Step 3: Implement typed descriptors and strict validation**

Use these public types:

```python
@dataclass(frozen=True)
class GridDescriptor:
    key: str
    frame_id: str
    origin_xy_m: tuple[float, float]
    resolution_m: float
    width: int
    height: int


@dataclass(frozen=True)
class PlatformSnapshot:
    key: str
    platform_id: str
    prim_path: str
    root_position_m: tuple[float, float, float]
    planning_position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    custom_data: dict[str, object]


@dataclass(frozen=True)
class SnapshotBundle:
    manifest_path: Path
    arrays_path: Path
    manifest: dict[str, object]
    grids: dict[str, GridDescriptor]
    platforms: dict[str, PlatformSnapshot]
    arrays: dict[str, np.ndarray]
```

Validation must enforce exact schema, 64-character lower-case SHA-256 values, positive dimensions/resolutions, finite poses, unit quaternions within `1e-6`, exactly `global`, `wheel`, `legged`, and `hopper` grid descriptors, every `<grid>__<layer>` array, identical shapes per grid, finite float layers, binary `{0,1}` masks, unit-interval quality, nonnegative variance/age/height/count, and integer-valued counts.

- [ ] **Step 4: Implement atomic NPZ then manifest publication**

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_snapshot_atomic(
    output_dir: Path,
    manifest: dict[str, object],
    arrays: Mapping[str, np.ndarray],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_arrays(manifest, arrays)
    arrays_tmp = output_dir / ".snapshot_arrays.npz.tmp"
    with arrays_tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    arrays_final = output_dir / "snapshot_arrays.npz"
    os.replace(arrays_tmp, arrays_final)
    document = dict(manifest)
    document["arrays_file"] = arrays_final.name
    document["arrays_sha256"] = sha256_file(arrays_final)
    manifest_tmp = output_dir / ".snapshot_manifest.json.tmp"
    manifest_tmp.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_tmp, output_dir / "snapshot_manifest.json")
    return output_dir / "snapshot_manifest.json"
```

Do not overwrite an already published manifest directory; each collection uses a new run directory.

- [ ] **Step 5: Run focused tests and UTF-8 validation**

Run:

```bash
python3 -m pytest -q test/test_snapshot_contract.py
python3 -m py_compile lunar_isaac_validation/snapshot_contract.py
```

Expected: all tests PASS.

- [ ] **Step 6: Commit the snapshot contract**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add ros2_ws/src/lunar_isaac_validation
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "feat: define atomic planning snapshot contract"
```

---

### Task 3: Implement deterministic terrain and collision rasterization

**Files:**
- Create: external `lunar_isaac_validation/rasterizer.py`
- Test: external `test/test_rasterizer.py`

**Interfaces:**
- Consumes: world-space terrain triangles, collision triangles, and `GridRequest`.
- Produces: `TriangleIndex`, `triangulate_faces(points_xyz, face_vertex_counts, face_vertex_indices)`, and `build_grid_layers(terrain_triangles, collision_meshes, request) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write failing flat-terrain and rock-obstacle tests**

```python
def test_flat_square_with_one_rock_produces_ten_layers() -> None:
    terrain = np.array([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
    ])
    rock = box_triangles((0.75, 0.75, 0.0), (1.25, 1.25, 0.5))
    request = GridRequest("test", "map", (0.0, 0.0), 0.5, 4, 4)
    layers = build_grid_layers(terrain, [rock], request)
    assert tuple(layers) == REQUIRED_LAYERS
    assert np.all(layers["valid_mask"] == 1)
    assert layers["obstacle"].sum() == 4
    assert np.max(layers["obstacle_height"]) == pytest.approx(0.5)
    assert np.all(layers["observation_quality"] == 1.0)
    assert np.all(layers["forbidden"] == 0)
```

Also test a triangular terrain boundary, a 20° planar slope, non-triangular face triangulation, deterministic repeated output, and collision geometry outside the grid.

- [ ] **Step 2: Run the tests and verify import failure**

Run: `python3 -m pytest -q test/test_rasterizer.py`

Expected: FAIL importing `GridRequest`.

- [ ] **Step 3: Implement face triangulation and a uniform XY triangle index**

```python
@dataclass(frozen=True)
class GridRequest:
    key: str
    frame_id: str
    origin_xy_m: tuple[float, float]
    resolution_m: float
    width: int
    height: int


def triangulate_faces(
    points_xyz: np.ndarray,
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
) -> np.ndarray:
    triangles: list[np.ndarray] = []
    cursor = 0
    for count in face_vertex_counts:
        face = face_vertex_indices[cursor:cursor + count]
        cursor += count
        for index in range(1, count - 1):
            triangles.append(points_xyz[[face[0], face[index], face[index + 1]]])
    return np.asarray(triangles, dtype=np.float64)
```

`TriangleIndex` must bin each triangle by its XY AABB using a bin width no larger than the requested grid resolution. Vertical queries use barycentric XY weights and accept points whose weights are each at least `-1e-9`; select the highest finite intersection.

- [ ] **Step 4: Implement five-sample elevation and collision rasterization**

For cell `(x,y)`, sample offsets `[(0.5,0.5),(0.25,0.25),(0.75,0.25),(0.25,0.75),(0.75,0.75)] * resolution`. Use the median and population variance of valid terrain heights. Set invalid elevation/variance/count to zero and `valid_mask=0`. Rasterize each collision mesh by triangle/cell XY overlap, compute the maximum collision Z over the cell, and set obstacle height relative to the terrain median. Emit float32 arrays in `REQUIRED_LAYERS` order.

- [ ] **Step 5: Run deterministic and range tests**

Run:

```bash
python3 -m pytest -q test/test_rasterizer.py
python3 -m pytest -q test/test_snapshot_contract.py test/test_rasterizer.py
```

Expected: PASS, with byte-identical arrays for repeated input.

- [ ] **Step 6: Commit the rasterizer**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add ros2_ws/src/lunar_isaac_validation
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "feat: rasterize USD terrain and rock collisions"
```

---

### Task 4: Implement the Python Server client and read-only Isaac collector

**Files:**
- Create: external `lunar_isaac_validation/python_server_client.py`
- Create: external `isaac/collect_stage_snapshot.py`
- Create: external `scripts/collect_snapshot.py`
- Test: external `test/test_python_server_client.py`

**Interfaces:**
- Consumes: loopback host/port, optional token, collector path, unique snapshot directory.
- Produces: `send_envelope(host, port, envelope, timeout_s, token)`, `server_status(host, port, token)`, `collect_live_stage(host, port, token, collector_path, output_dir)`, and a validated `snapshot_manifest.json`.

- [ ] **Step 1: Write a failing fake-server protocol test**

```python
def test_send_envelope_half_closes_and_parses_json() -> None:
    async def scenario() -> None:
        received = b""

        async def handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            nonlocal received
            received = await reader.read()
            writer.write(b'{"status":"ok","result":{"contexts":[]}}')
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        response = await send_envelope(
            "127.0.0.1", port, {"introspect": "status"}, timeout_s=1.0
        )
        server.close()
        await server.wait_closed()
        assert json.loads(received)["introspect"] == "status"
        assert response["status"] == "ok"

    asyncio.run(scenario())
```

Add tests for token injection, timeout, malformed JSON, `status=error`, and refusal of non-loopback hosts.

- [ ] **Step 2: Run the protocol tests and verify failure**

Run: `python3 -m pytest -q test/test_python_server_client.py`

Expected: FAIL importing `send_envelope`.

- [ ] **Step 3: Implement the JSON-envelope client**

```python
async def send_envelope(
    host: str,
    port: int,
    envelope: Mapping[str, object],
    timeout_s: float,
    token: str | None = None,
) -> dict[str, object]:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise PythonServerError("ISAAC_SERVER_NOT_LOOPBACK")
    reader, writer = await asyncio.open_connection(host, port)
    request = dict(envelope)
    if token:
        request["auth_token"] = token
    writer.write(json.dumps(request).encode("utf-8"))
    await writer.drain()
    writer.write_eof()
    raw = await asyncio.wait_for(reader.read(), timeout=timeout_s)
    writer.close()
    await writer.wait_closed()
    response = json.loads(raw.decode("utf-8"))
    if response.get("status") != "ok":
        raise PythonServerError(str(response.get("evalue", "ISAAC_SERVER_ERROR")))
    return response
```

`collect_live_stage` must submit the expression below in named context `lunar_snapshot_v1` with timeout 120 seconds and return only the JSON-serializable `result`:

```python
__import__('runpy').run_path(collector_path)['collect_current_stage'](output_dir)
```

- [ ] **Step 4: Implement the read-only collector**

`collect_current_stage(output_dir: str) -> dict[str, object]` must obtain `omni.usd.get_context().get_stage()`, reject an absent stage, require `metersPerUnit == 1` and Z-up, hash the root layer file before extraction, and use `UsdGeom.XformCache(Usd.TimeCode.Default())` to transform mesh points into world coordinates. It must require one terrain mesh, every `LunarPolarRock_*` mesh to have `UsdPhysics.CollisionAPI`, and all three platform roots/custom data.

At module import, derive the package source without a hard-coded duplicate root and expose only pure shared modules to Isaac Python:

```python
PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "lunar_isaac_validation"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))
from lunar_isaac_validation.rasterizer import GridRequest, build_grid_layers
from lunar_isaac_validation.snapshot_contract import write_snapshot_atomic
```

Neither imported module may import `rclpy` or generated ROS message packages.

Create these grid requests:

```python
GLOBAL_RESOLUTION_M = 0.5
LOCAL_RESOLUTION_M = 0.25
LOCAL_EXTENT_M = {"wheel": 12.0, "legged": 12.0, "hopper": 8.0}
PLANNING_Z_OFFSETS_M = {"wheel": 0.0, "legged": 0.55, "hopper": 0.60}
```

The global request uses frame `map`, origin at the terrain XY minimum, and `ceil((maximum-minimum)/0.5)` cells on each axis. Platform-local requests use frame `odom`, stay centered on the planning XY when possible, and use `ceil(extent/0.25)` cells on each axis. Clamp local windows to the terrain XY bounds without moving the platform outside its own local grid. Build arrays using `rasterizer.build_grid_layers`, re-hash the root USD after extraction, fail with `USD_CHANGED_DURING_COLLECTION` on mismatch, and publish the bundle through `write_snapshot_atomic`. Do not call any stage-open, timeline, edit-target, save, export, or authoring API.

- [ ] **Step 5: Run local tests and syntax checks**

Run:

```bash
python3 -m pytest -q test/test_python_server_client.py
python3 -m py_compile isaac/collect_stage_snapshot.py scripts/collect_snapshot.py
```

Expected: PASS.

- [ ] **Step 6: Probe the live server without collecting or modifying the stage**

Run:

```bash
python3 scripts/collect_snapshot.py --status-only --host 127.0.0.1 --port 8226
```

Expected: status `ok`, server uptime, and no executed collector context. If authentication is enabled, read the token from `LUNAR_ISAAC_SERVER_TOKEN_FILE`; never print it.

- [ ] **Step 7: Commit the client and collector**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add isaac scripts/collect_snapshot.py ros2_ws/src/lunar_isaac_validation
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "feat: collect read-only Isaac planning snapshots"
```

---

### Task 5: Create the three capabilities, provenance, URDFs, and meshes

**Files:**
- Create: external `lunar_isaac_validation/capability_contract.py`
- Create: external `config/capabilities/{wheeled,legged,hopper}.yaml`
- Create: external `config/{observation.json,capability_provenance.json}`
- Create: external `config/planner/{wheeled,legged,hopper}.yaml`
- Create: external `urdf/{wheeled,legged,hopper}_proxy.urdf`
- Create: external `meshes/{wheeled,legged,hopper}_proxy_collision.stl`
- Create: external `scripts/generate_proxy_meshes.py`
- Test: external `test/test_capability_files.py`

**Interfaces:**
- Consumes: approved proxy values and current loader schema.
- Produces: three package-share-loadable capability sources and `validate_capability_bundle(package_root) -> dict[str, object]`.

- [ ] **Step 1: Write failing exact-value and provenance tests**

```python
EXPECTED = {
    "wheel": {"platform_id": "proxy-wheeled-scout-v1",
              "maximum_slope_rad": math.radians(18.0),
              "maximum_obstacle_height_m": 0.18,
              "maximum_forward_speed_mps": 0.70},
    "legged": {"platform_id": "proxy-legged-scout-v1",
               "maximum_slope_rad": math.radians(28.0),
               "maximum_roughness_m": 0.12,
               "maximum_step_height_m": 0.22},
    "hopper": {"platform_id": "proxy-hopper-scout-v1",
               "maximum_landing_slope_rad": math.radians(15.0),
               "minimum_landing_region_area_m2": 1.327322,
               "maximum_launch_speed_mps": 2.20},
}


def test_all_required_values_geometry_and_provenance_are_present() -> None:
    report = validate_capability_bundle(PACKAGE_ROOT)
    assert report["observation"] == {"sensor_range_m": 30.0,
                                     "sensor_fov_deg": 120.0}
    assert report["missing_provenance"] == []
    assert report["mesh_paths"] == {
        "wheel": "meshes/wheeled_proxy_collision.stl",
        "legged": "meshes/legged_proxy_collision.stl",
        "hopper": "meshes/hopper_proxy_collision.stl",
    }
```

- [ ] **Step 2: Run the test and verify missing files fail**

Run: `python3 -m pytest -q test/test_capability_files.py`

Expected: FAIL with `CAPABILITY_FILE_MISSING`.

- [ ] **Step 3: Author the exact capability and planner YAML files**

Use these complete scalar sections without widening any bound:

```yaml
wheeled:
  footprint_xy_m: [[-0.61, -0.54], [0.61, -0.54], [0.61, 0.54], [-0.61, 0.54]]
  minimum_body_z_m: 0.0
  maximum_body_z_m: 0.555
  maximum_slope_rad: 0.3141592653589793
  maximum_obstacle_height_m: 0.18
  maximum_forward_speed_mps: 0.70
  maximum_reverse_speed_mps: 0.35
  maximum_spin_rate_radps: 0.60
  maximum_acceleration_mps2: 0.40
  maximum_braking_deceleration_mps2: 0.50
  maximum_yaw_acceleration_radps2: 0.60
  maximum_lateral_acceleration_mps2: 0.35
  maximum_curvature_per_m: 1.4285714285714286
  minimum_clearance_m: 0.22

legged:
  reference_point: base_link
  body_half_extent_m: [0.475, 0.325, 0.14]
  maximum_slope_rad: 0.4886921905584123
  maximum_roughness_m: 0.12
  maximum_step_height_m: 0.22
  maximum_gap_width_m: 0.35
  minimum_confidence: 0.80
  minimum_body_clearance_m: 0.25
  body_height_m: [0.50, 0.60]
  forward_speed_mps: [-0.20, 0.45]
  lateral_speed_mps: [-0.30, 0.30]
  vertical_speed_mps: [-0.15, 0.15]
  yaw_rate_radps: [-0.60, 0.60]
  maximum_linear_acceleration_mps2: 0.35
  maximum_yaw_acceleration_radps2: 0.60

hopper:
  body_half_extent_m: [0.275, 0.275, 0.60]
  platform_mass_kg: 20.0
  gravity_mps2: [0.0, 0.0, -1.62]
  maximum_landing_slope_rad: 0.2617993877991494
  maximum_landing_roughness_m: 0.08
  maximum_plane_residual_m: 0.05
  minimum_overhead_clearance_m: 0.10
  minimum_lateral_clearance_m: 0.10
  minimum_landing_region_area_m2: 1.327322
  maximum_launch_speed_mps: 2.20
  maximum_launch_impulse_newton_seconds: 44.0
  minimum_flight_time_s: 0.50
  maximum_flight_time_s: 3.00
  maximum_landing_speed_mps: 2.20
  minimum_downward_impact_speed_mps: 0.10
  minimum_landing_clearance_m: 0.25
  maximum_angular_speed_radps: 0.80
  maximum_angular_acceleration_radps2: 1.00
  maximum_initial_angular_speed_radps: 0.10
  minimum_settle_guard_s: 1.00
  actuator_or_impulse_profile: {profile_id: proxy_impulse_v1}
```

Freeze the motion primitives as follows:

- Wheeled `forward-025`: `FORWARD`, pose `([0.25,0,0],[1,0,0,0])`, `2.0 s`; `reverse-025`: `REVERSE`, pose `([-0.25,0,0],[1,0,0,0])`, `2.0 s`.
- Wheeled `arc-left-045`: `FORWARD_ARC`, pose `([0.4949747468305832,0.2050252531694167,0],[0.9238795325112867,0,0,0.3826834323650898])`, `3.0 s`; `arc-right-045` mirrors Y and quaternion Z, also `3.0 s`. Both use radius `0.70 m` and yaw magnitude `π/4`.
- Wheeled `spin-cw-045`/`SPIN_CLOCKWISE` and `spin-ccw-045`/`SPIN_COUNTERCLOCKWISE`: zero translation, yaw quaternion Z respectively `-0.3826834323650898` and `0.3826834323650898`, `3.0 s`; `stop-and-switch`/`STOP_AND_SWITCH`: identity pose, `0.5 s`.
- Legged `forward-025`/`FORWARD`, `backward-025`/`BACKWARD`, `left-025`/`LATERAL_LEFT`, and `right-025`/`LATERAL_RIGHT`: displacements `[0.25,0,0]`, `[-0.25,0,0]`, `[0,0.25,0]`, `[0,-0.25,0]`, zero yaw, each `2.0 s`; `spin-left-045`/`SPIN` and `spin-right-045`/`SPIN`: zero displacement, yaw `±0.7853981633974483`, `3.0 s`.
- Hopper contains only primitive id `nominal-hop`.

Each file uses schema `platform-control-capability-source/v1`, `capability_version: proxy-v1`, `base_frame_id: base_link`, the approved platform id/type, and respectively `urdf/wheeled_proxy.urdf`, `urdf/legged_proxy.urdf`, or `urdf/hopper_proxy.urdf` as `geometry_source.urdf_file`.

Provenance is also frozen: ids/type/version/frame, mass, and scalar limits present in platform `customData` use `usd_custom_data`; footprint/body extents and body-Z bounds use `usd_collision_geometry`; degree-to-radian slopes, inverse turn radius, PhysicsScene gravity, `πr²` landing area, and `mass×launch_speed` impulse use `derived`. Observation range/FOV, every detailed primitive entry, wheel reverse/spin/braking/yaw/lateral limits, legged gap/confidence/speed intervals/yaw acceleration, and hopper plane-residual/overhead/lateral/minimum-flight/landing-speed/downward-impact/angular limits use `validation_only_assumption`. Any mixed field containing an assumed bound, such as the legged forward interval, uses `validation_only_assumption` for the complete field.

Each planner YAML targets `lunar_planner.ros__parameters` and sets:

```yaml
global_map_max_age: 2.0
local_map_max_age: 2.0
odometry_max_age: 0.5
localization_status_max_age: 0.5
tf_max_age: 0.5
max_pairwise_skew: 0.25
degraded_pose_covariance_limit: 0.5
degraded_twist_covariance_limit: 0.5
maximum_transform_samples: 256
capability_package: lunar_isaac_validation
observation_capability_file: config/observation.json
```

Set `platform_capability_file` to the matching platform YAML.

- [ ] **Step 4: Generate deterministic proxy STL meshes and matching URDFs**

`generate_proxy_meshes.py` must use deterministic primitive tessellation. The following is the required core; `write_ascii_stl` sorts triangle tuples before writing facets so repeated runs are byte-identical:

```python
Triangle = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def box_triangles(center: tuple[float, float, float],
                  size: tuple[float, float, float]) -> list[Triangle]:
    cx, cy, cz = center
    hx, hy, hz = (value / 2.0 for value in size)
    vertices = (
        (cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz),
        (cx+hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz-hz),
        (cx-hx, cy-hy, cz+hz), (cx+hx, cy-hy, cz+hz),
        (cx+hx, cy+hy, cz+hz), (cx-hx, cy+hy, cz+hz),
    )
    faces = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    )
    return [tuple(vertices[index] for index in face) for face in faces]


def cylinder_triangles(center: tuple[float, float, float], radius: float,
                       height: float, axis: str, segments: int = 24) -> list[Triangle]:
    if axis not in {"x", "y", "z"} or segments < 3:
        raise ValueError("invalid cylinder tessellation")
    cx, cy, cz = center
    half = height / 2.0

    def orient(x: float, y: float, z: float) -> tuple[float, float, float]:
        local = {"x": (z, x, y), "y": (x, z, y), "z": (x, y, z)}[axis]
        return (cx + local[0], cy + local[1], cz + local[2])

    low = [orient(radius * math.cos(math.tau*i/segments),
                  radius * math.sin(math.tau*i/segments), -half)
           for i in range(segments)]
    high = [orient(radius * math.cos(math.tau*i/segments),
                   radius * math.sin(math.tau*i/segments), half)
            for i in range(segments)]
    low_center, high_center = orient(0.0, 0.0, -half), orient(0.0, 0.0, half)
    triangles: list[Triangle] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.extend((
            (low[index], low[nxt], high[nxt]),
            (low[index], high[nxt], high[index]),
            (low_center, low[nxt], low[index]),
            (high_center, high[index], high[nxt]),
        ))
    return triangles


def uv_sphere_triangles(center: tuple[float, float, float], radius: float,
                        rings: int = 12, segments: int = 24) -> list[Triangle]:
    if rings < 2 or segments < 3:
        raise ValueError("invalid sphere tessellation")
    cx, cy, cz = center
    rows: list[list[tuple[float, float, float]]] = []
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        rows.append([
            (cx + radius * math.sin(phi) * math.cos(math.tau*i/segments),
             cy + radius * math.sin(phi) * math.sin(math.tau*i/segments),
             cz + radius * math.cos(phi))
            for i in range(segments)
        ])
    top, bottom = (cx, cy, cz + radius), (cx, cy, cz - radius)
    triangles = []
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.append((top, rows[0][index], rows[0][nxt]))
        for row in range(len(rows) - 1):
            triangles.extend((
                (rows[row][index], rows[row+1][index], rows[row+1][nxt]),
                (rows[row][index], rows[row+1][nxt], rows[row][nxt]),
            ))
        triangles.append((bottom, rows[-1][nxt], rows[-1][index]))
    return triangles
```

Compose the exact USD proxy envelopes in `base_link` coordinates:

- Wheeled: box center `(0,0,0.38)`, size `(1.20,0.85,0.35)`; four Y-axis cylinders at `(-0.43,-0.48,0.18)`, `(-0.43,0.48,0.18)`, `(0.43,-0.48,0.18)`, `(0.43,0.48,0.18)`, each radius `0.18` and length `0.12`.
- Legged: box center `(0,0,0.55)`, size `(0.95,0.65,0.28)`; six leg boxes centered at `(-0.34,-0.36,0.28)`, `(-0.34,0.36,0.28)`, `(0,-0.42,0.28)`, `(0,0.42,0.28)`, `(0.34,-0.36,0.28)`, `(0.34,0.36,0.28)`, each size `(0.10,0.10,0.50)`; six Z-axis foot cylinders at the same XY and `z=0.05`, each radius/height `0.10`.
- Hopper: sphere center `(0,0,0.60)`, radius `0.275`; leg box center `(0,0,0.30)`, size `(0.12,0.12,0.45)`; Z-axis foot cylinder center `(0,0,0.06)`, radius `0.22`, height `0.12`.

Every URDF has one `base_link` whose visual and collision elements both use the matching URI from `package://lunar_isaac_validation/meshes/wheeled_proxy_collision.stl`, `package://lunar_isaac_validation/meshes/legged_proxy_collision.stl`, or `package://lunar_isaac_validation/meshes/hopper_proxy_collision.stl`, with scale `1 1 1`.

- [ ] **Step 5: Implement capability/provenance validation**

The validator must reject unknown/missing top-level schema fields, unsafe geometry paths, missing mesh files, empty motion primitive lists, duplicate ids, non-unit quaternions, nonfinite or out-of-range values, mismatched platform ids/types, and any scalar capability field without one provenance class from `{usd_custom_data, usd_collision_geometry, derived, validation_only_assumption}`.

- [ ] **Step 6: Run generation and tests**

Run:

```bash
python3 scripts/generate_proxy_meshes.py
python3 -m pytest -q test/test_capability_files.py
python3 -m py_compile scripts/generate_proxy_meshes.py lunar_isaac_validation/capability_contract.py
```

Expected: PASS; rerunning mesh generation produces identical SHA-256 values.

- [ ] **Step 7: Commit capabilities and geometry**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add scripts/generate_proxy_meshes.py ros2_ws/src/lunar_isaac_validation
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "feat: add three proxy capability bundles"
```

---

### Task 6: Implement scene-backed fixture qualification and immutable locking

**Files:**
- Create: external `lunar_isaac_validation/scenario_qualifier.py`
- Create: external `scripts/qualify_fixtures.sh`
- Test: external `test/test_scenario_qualifier.py`

**Interfaces:**
- Consumes: `SnapshotBundle` and validated capability dictionaries.
- Produces: `ScenarioCase`, `ScenarioLock`, `qualify_snapshot(bundle, capabilities)`, `write_scenario_lock(path, lock, allow_rebaseline)`, and an evidence report.

- [ ] **Step 1: Write failing synthetic qualification tests**

```python
def test_qualifier_selects_all_six_cases_without_mutating_layers(tmp_path: Path) -> None:
    bundle = synthetic_bundle_with_flat_route_steep_band_rock_and_landing_zone()
    original = {name: value.copy() for name, value in bundle.arrays.items()}
    lock, evidence = qualify_snapshot(bundle, capability_documents())
    assert [case.case_id for case in lock.cases] == [
        "wheel-positive", "wheel-negative", "legged-positive",
        "legged-negative", "hopper-positive", "hopper-negative",
    ]
    assert lock.stage_sha256 == bundle.manifest["stage_sha256"]
    assert lock.cases[0].expected_reason == "WHEEL_PLAN_AVAILABLE"
    assert lock.cases[-1].expected_reason == "HOPPER_GOAL_INFEASIBLE"
    for name, value in original.items():
        np.testing.assert_array_equal(bundle.arrays[name], value)


def test_lock_writer_refuses_existing_lock(tmp_path: Path) -> None:
    path = tmp_path / "scenario_lock.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ScenarioError, match="SCENARIO_LOCK_EXISTS"):
        write_scenario_lock(path, valid_lock(), allow_rebaseline=False)
```

- [ ] **Step 2: Run tests and verify missing qualifier fails**

Run: `python3 -m pytest -q test/test_scenario_qualifier.py`

Expected: FAIL importing `ScenarioCase`.

- [ ] **Step 3: Implement shared terrain metrics and connected-route selection**

Define slope from central elevation differences, roughness as `sqrt(elevation_variance)`, maximum neighbor step over eight neighbors, and obstacle clearance by deterministic eight-neighbor Dijkstra distance. Build platform-specific hard-feasible masks without modifying source arrays.

Use stable priority tuples `(distance_from_start, y_index, x_index)` and fixed neighbor order. Wheel-positive selects a 2–4 m reachable target on slope ≤18°. Legged-positive selects a 2–4 m reachable target whose reconstructed path contains at least one slope in `(18°,28°]`. Hopper-positive selects a 1–2 m target with a contiguous axis-aligned safe rectangle of area ≥1.327322 m².

- [ ] **Step 4: Implement exact negative selection and lock schema**

Wheel-negative tolerance cells must all have obstacle height >0.18 m. Legged-negative tolerance cells must all exceed 28° or 0.22 m step. Hopper-negative tolerance cells must all fail slope, roughness, plane residual, or clearance before area expansion. Store exact expected triples:

```python
EXPECTED = {
    "wheel-positive": (0, 0, "WHEEL_PLAN_AVAILABLE"),
    "wheel-negative": (3, 2, "WHEEL_GOAL_INFEASIBLE"),
    "legged-positive": (0, 0, "LEGGED_BODY_PLAN_AVAILABLE"),
    "legged-negative": (3, 2, "LEGGED_GOAL_INFEASIBLE"),
    "hopper-positive": (0, 0, "HOPPER_FIRST_HOP_AVAILABLE"),
    "hopper-negative": (3, 2, "HOPPER_GOAL_INFEASIBLE"),
}
```

Here enum integers match `PlanMotion.Result`: new reference 0, goal infeasible 3, activate 0, hold 2. Lock JSON must include schema, both hashes, start/goal/tolerance, evidence metrics, and expected values.

- [ ] **Step 5: Implement explicit first-lock and rebaseline behavior**

Normal invocation creates the lock only if absent. `--rebaseline --reason <nonempty text>` writes a timestamped old/new comparison report under `artifacts/<run-id>/` before replacing the external lock; it never derives expected values from Action output.

- [ ] **Step 6: Run qualification unit tests**

Run:

```bash
python3 -m pytest -q test/test_scenario_qualifier.py
python3 -m pytest -q test/test_snapshot_contract.py test/test_rasterizer.py test/test_scenario_qualifier.py
```

Expected: PASS and deterministic identical lock JSON across repeated synthetic input.

- [ ] **Step 7: Commit qualifier source without a live lock yet**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add scripts/qualify_fixtures.sh ros2_ws/src/lunar_isaac_validation
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "feat: qualify immutable scene-backed scenarios"
```

---

### Task 7: Implement GridMap encoding and the six-topic ROS bridge

**Files:**
- Create: external `lunar_isaac_validation/grid_map_codec.py`
- Create: external `lunar_isaac_validation/bridge_node.py`
- Modify: external `setup.py` — add `lunar_isaac_bridge` console script.
- Test: external `test/test_grid_map_codec.py`, `test/test_bridge_node.py`

**Interfaces:**
- Consumes: validated snapshot/lock and one `scenario_id`.
- Produces: `encode_grid_map(layers, frame_id, origin_xy, resolution, stamp) -> GridMap`, `SnapshotBridge(Node)`, and `/lunar_isaac_validation/ready` (`std_srvs/Trigger`).

- [ ] **Step 1: Write failing GridMap unwrap compatibility test**

```python
def test_codec_matches_planner_adapter_layout() -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    message = encode_grid_map(
        layers=ten_layers_with_elevation(values),
        frame_id="odom", origin_xy=(0.0, 0.0), resolution=1.0,
        stamp=Time(sec=10),
    )
    assert message.data[0].layout.dim[0].label == "column_index"
    assert message.data[0].layout.dim[0].size == 2
    assert message.data[0].layout.dim[1].label == "row_index"
    assert message.data[0].layout.dim[1].size == 3
    assert message.outer_start_index == 0
    assert message.inner_start_index == 0
    assert decode_like_cpp_adapter(message.data[0], 3, 2) == pytest.approx(values.ravel())
```

- [ ] **Step 2: Run the codec test under a fresh external development overlay and verify failure**

Run:

```bash
source /opt/ros/humble/setup.bash
export LUNAR_VALIDATION_ROOT=/home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression
colcon --log-base "$LUNAR_VALIDATION_ROOT/log/dev" build \
  --base-paths /mnt/data/WS/lunar-navigation/ros2_ws/src "$LUNAR_VALIDATION_ROOT/ros2_ws/src" \
  --build-base "$LUNAR_VALIDATION_ROOT/build/dev" \
  --install-base "$LUNAR_VALIDATION_ROOT/install/dev" \
  --packages-up-to lunar_isaac_validation
source "$LUNAR_VALIDATION_ROOT/install/dev/setup.bash"
python3 -m pytest -q "$LUNAR_VALIDATION_ROOT/ros2_ws/src/lunar_isaac_validation/test/test_grid_map_codec.py"
```

Expected: FAIL importing `encode_grid_map`.

- [ ] **Step 3: Implement the exact circular-buffer layout**

For `outer_start_index=inner_start_index=0`, encode logical `[y,x]` into physical index:

```python
physical_row = width - 1 - x
physical_column = height - 1 - y
physical_index = physical_column * width + physical_row
```

Set outer dim `column_index,size=height,stride=width*height`, inner dim `row_index,size=width,stride=width`, and GridMap pose center to `(origin_x + width*resolution/2, origin_y + height*resolution/2)`. Preserve required layer order and set basic layers to `elevation` and `valid_mask`.

- [ ] **Step 4: Write failing bridge QoS, frame, and readiness tests**

Use a `rclpy` test node to subscribe with the planner’s exact QoS, spin until three generations, then assert global frame `map`, local/Odometry/status frame `odom`, child `base_link`, identity `map→odom`, matching Odometry/TF planning pose, mission state `ACTIVE`, revision 1, and one shared nonzero stamp per generation. Assert readiness is false with zero subscribers and true only after all six publisher subscription counts are nonzero and three generations have been emitted.

- [ ] **Step 5: Implement `SnapshotBridge`**

```python
class SnapshotBridge(Node):
    def __init__(self, manifest_path: Path, lock_path: Path, scenario_id: str) -> None:
        super().__init__("lunar_isaac_bridge")
        self.bundle = load_snapshot(manifest_path)
        self.case = load_scenario_lock(lock_path).case(scenario_id)
        self.generation = 0
        self.create_timer(0.1, self._publish_generation)

    def _publish_generation(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._publish_dynamic_inputs(stamp)
        self._publish_maps_and_mission(stamp)
        self.generation += 1
```

Use reliable transient-local depth 1 for maps/mission, SensorDataQoS for Odometry, reliable depth 10 for localization, and best-effort depth 100 for TF. Publish all six inputs at 10 Hz with the one shared generation stamp; this exceeds the design's minimum map rate and guarantees pairwise skew below 0.25 seconds. Pose/twist covariance diagonals are `0.01`; all off-diagonals are zero. Readiness becomes true only after three complete six-topic generations and nonzero subscriber counts for all publishers. Do not publish `/clock`.

- [ ] **Step 6: Run codec and bridge tests**

Run:

```bash
python3 -m pytest -q test/test_grid_map_codec.py test/test_bridge_node.py
```

Expected: PASS without a planner process.

- [ ] **Step 7: Commit the ROS bridge**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add ros2_ws/src/lunar_isaac_validation
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "feat: publish synchronized planner inputs"
```

---

### Task 8: Implement platform-specific Action result and geometry assertions

**Files:**
- Create: external `lunar_isaac_validation/{trajectory_checks.py,hop_checks.py,action_assertions.py}`
- Test: external `test/{test_trajectory_checks.py,test_hop_checks.py,test_action_assertions.py}`

**Interfaces:**
- Consumes: `ScenarioCase`, `PlanMotion.Result`, `SnapshotBundle`, and parsed capability data.
- Produces: `assert_action_result(case, result, bundle, capabilities) -> dict[str, object]` and `normalize_reference(result) -> dict[str, object]`.

- [ ] **Step 1: Write failing exact-result tests**

```python
def test_negative_requires_exact_reason_and_no_reference() -> None:
    case = make_case("wheel-negative", outcome=3, directive=2,
                     reason="WHEEL_GOAL_INFEASIBLE")
    result = PlanMotion.Result()
    result.planning_outcome = 3
    result.execution_directive = 2
    result.reason_code = "WHEEL_GOAL_INFEASIBLE"
    result.has_reference = False
    evidence = assert_action_result(case, result, bundle(), capabilities())
    assert evidence["passed"] is True


def test_wrong_reason_fails_even_if_outcome_matches() -> None:
    result = make_negative_result(reason="WHEEL_NO_KNOWN_SAFE_ROUTE")
    with pytest.raises(ActionAssertionError, match="ACTION_REASON_MISMATCH"):
        assert_action_result(wheel_negative_case(), result, bundle(), capabilities())
```

Add positive fixtures for a bounded wheel trajectory, legged body trajectory, one valid hop, timestamp/revision mismatch, NaN, nonmonotonic time, over-speed, collision, and impulse violation.

- [ ] **Step 2: Run tests and verify assertion modules are missing**

Run:

```bash
python3 -m pytest -q test/test_trajectory_checks.py test/test_hop_checks.py test/test_action_assertions.py
```

Expected: FAIL importing `assert_action_result`.

- [ ] **Step 3: Implement wheel and legged trajectory checks**

Require `reference.platform_type` to match, no hop entries, joint names exactly `['base_link']`, equal nonempty path/trajectory counts, one transform and velocity per point, finite normalized quaternions, nondecreasing stamps and strictly increasing `time_from_start` after the first point. Check start within `1e-3 m`, terminal XY within goal tolerance, wheel linear/spin/acceleration bounds, legged forward/lateral/vertical/yaw intervals, body height interval, and both acceleration bounds.

Sample each segment at spacing no larger than `0.125 m`; transform the wheel footprint or legged body rectangle into each sample pose and reject overlap with invalid, obstacle, forbidden, slope, roughness, step, or clearance violations.

- [ ] **Step 4: Implement hopper ballistic checks**

Require exactly one hop, no MultiDOF trajectory points, nonempty segment id, at least four landing polygon points, and positive flight tube radius. Recompute:

```python
landing_position = launch_position + launch_velocity * t + 0.5 * gravity * t * t
landing_velocity = launch_velocity + gravity * t
impulse = mass_kg * np.linalg.norm(launch_velocity - initial_velocity)
downward_speed = float(np.dot(landing_velocity, gravity) / np.linalg.norm(gravity))
```

Assert flight time, launch speed, impulse, landing speed, downward speed, landing goal tolerance, polygon area, and landing slope/roughness/clearance against the hopper capability.

- [ ] **Step 5: Implement exact Action dispatch and normalized summaries**

First compare expected outcome, directive, reason, mission revision, nonzero result stamps, and `has_reference`. Negative cases return evidence only after proving reference fields are empty. Positive cases dispatch by platform. Normalization excludes elapsed time and ROS publication stamps but retains outcome, directive, reason, platform, point/hop count, quantized coordinates at `1e-6`, and warning-code set.

- [ ] **Step 6: Run all assertion tests**

Run:

```bash
python3 -m pytest -q test/test_trajectory_checks.py test/test_hop_checks.py test/test_action_assertions.py
```

Expected: PASS.

- [ ] **Step 7: Commit Action assertions**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add ros2_ws/src/lunar_isaac_validation
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "test: validate platform Action references"
```

---

### Task 9: Implement managed processes, Lifecycle/Action orchestration, and reports

**Files:**
- Create: external `lunar_isaac_validation/{process_manager.py,reports.py,regression_runner.py}`
- Modify: external `setup.py` — add `lunar_isaac_regression` console script.
- Test: external `test/{test_process_manager.py,test_reports.py,test_regression_runner.py}`

**Interfaces:**
- Consumes: installed overlay, snapshot, scenario lock, planner parameter files.
- Produces: `ManagedProcess`, `CaseResult`, `run_case(case, paths)`, `run_regression(cases, paths)`, JSON report, and JUnit XML.

- [ ] **Step 1: Write failing managed-child cleanup tests**

```python
def test_manager_terminates_only_its_recorded_process(tmp_path: Path) -> None:
    process = ManagedProcess.start(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        log_path=tmp_path / "child.log",
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        process.stop(term_timeout_s=1.0)
        assert process.returncode is not None
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
```

Also test natural exit, timeout escalation to the recorded process group, and cleanup error code 60.

- [ ] **Step 2: Write failing report and fake-ROS runner tests**

Test six ordered case results, continuation after one Action failure, final nonzero status, token redaction, normalized-repeat comparison, JSON schema fields, and one JUnit testcase per scenario.

- [ ] **Step 3: Run tests and verify missing modules fail**

Run:

```bash
python3 -m pytest -q test/test_process_manager.py test/test_reports.py test/test_regression_runner.py
```

Expected: FAIL importing `ManagedProcess`.

- [ ] **Step 4: Implement explicit managed processes**

Use `subprocess.Popen(argv, stdout=log_stream, stderr=subprocess.STDOUT, start_new_session=True)` and store PID, process-group id, argv, start time, and log path. `stop()` first calls the Lifecycle shutdown callback when supplied, then sends `SIGTERM` only to the recorded group, waits, and sends `SIGKILL` only to that same group on timeout. Never enumerate system processes or call `pkill`.

- [ ] **Step 5: Implement Lifecycle and Action clients**

`RegressionRunner(Node)` must create clients for `/lunar_planner/change_state`, `/lunar_planner/get_state`, `/lunar_isaac_validation/ready`, and `/plan_motion`. For each case:

1. Start `lunar_isaac_bridge --manifest <manifest_path> --lock <lock_path> --scenario <case_id>` using the exact paths held by the current `RegressionPaths` object.
2. Start `ros2 run lunar_planner_ros lunar_planner_node --ros-args --params-file <platform.yaml>`.
3. Wait for services, call configure id 1 and verify inactive id 2; call activate id 3 and verify active id 3.
4. Wait for bridge readiness success and Action server readiness.
5. Send deterministic request/mission/goal ids with `replace_active_request=false`.
6. Require ROS Action terminal status `STATUS_SUCCEEDED`, then validate the planning result.
7. Call deactivate id 4, cleanup id 2, shutdown id 5; stop recorded children.

Construct each goal in the global `map` frame from the locked case, using the most recent complete bridge-generation stamp:

```python
goal = PlanMotion.Goal()
goal.request_id = f"{case.case_id}-request"
goal.mission_id = f"{case.case_id}-mission"
goal.mission_revision = 1
goal.replace_active_request = False
goal.goal.header.frame_id = "map"
goal.goal.header.stamp = generation_stamp
goal.goal.goal_id = f"{case.case_id}-goal"
goal.goal.goal_type = goal.goal.POINT
goal.goal.point.x, goal.goal.point.y, goal.goal.point.z = case.goal_position_m
goal.goal.position_tolerance_m = case.goal_tolerance_m
goal.goal.has_yaw_constraint = False
goal.goal.yaw_rad = 0.0
goal.goal.yaw_tolerance_rad = 0.0
```

The bridge mission uses the same mission id, header frame/stamp, revision 1, state `ACTIVE`, and global-grid bounds as ROI. The readiness service message encodes the latest complete generation stamp as JSON so the runner never sends a zero or cross-generation goal stamp.

Lifecycle operations timeout at 30 seconds; Action result timeout at 60 seconds. Continue to the next case only after cleanup completes.

- [ ] **Step 6: Implement JSON/JUnit reports and exit grouping**

```python
@dataclass(frozen=True)
class CaseResult:
    case_id: str
    platform: str
    passed: bool
    duration_s: float
    expected: dict[str, object]
    actual: dict[str, object]
    assertions: list[dict[str, object]]
    error_code: int | None
    error_detail: str | None
```

Write `environment.json`, a manifest copy, `scenario_results.json`, `junit.xml`, child logs, and `summary.json`. Recursively redact keys containing `token`, `secret`, or `auth`. Overall exit code is the first nonzero category in preflight order, otherwise 50 for any case failure, 60 for cleanup failure, or 0.

- [ ] **Step 7: Run process, report, and fake runner tests**

Run:

```bash
python3 -m pytest -q test/test_process_manager.py test/test_reports.py test/test_regression_runner.py
```

Expected: PASS with no lingering test child process.

- [ ] **Step 8: Commit orchestration and reports**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add ros2_ws/src/lunar_isaac_validation
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "feat: orchestrate isolated planner Action cases"
```

---

### Task 10: Add safe entry points, collect the live snapshot, and lock real fixtures

**Files:**
- Create: external `scripts/{preflight.py,build_external.sh,run_action_regression.sh}`
- Modify: external `scripts/{collect_snapshot.py,qualify_fixtures.sh}`
- Create at runtime: external `snapshots/<run-id>/{snapshot_manifest.json,snapshot_arrays.npz}`
- Create and commit after review: external `ros2_ws/src/lunar_isaac_validation/scenarios/scenario_lock.json`
- Test: external `test/test_cli_contract.py`

**Interfaces:**
- Consumes: current GUI stage, current main-repository source, and Tasks 1–9.
- Produces: a qualified current-stage lock, a fresh external install overlay, and safe one-command entry points.

- [ ] **Step 1: Write failing CLI safety-contract tests**

```python
def test_shell_entrypoints_never_target_main_repo_for_outputs() -> None:
    for name in ("build_external.sh", "qualify_fixtures.sh", "run_action_regression.sh"):
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "rm -rf" not in text
        assert "pkill" not in text
        assert "--build-base" in text or name != "build_external.sh"
        assert "/mnt/data/WS/lunar-navigation/build" not in text
        assert "source /opt/ros/humble/setup.bash" in text
```

Preflight tests must also cover wrong ROS distro, Python not 3.10, non-loopback host, missing design commit, unwritable external root, existing lock without rebaseline, and secret redaction.

- [ ] **Step 2: Run CLI tests and verify missing entry points fail**

Run: `python3 -m pytest -q test/test_cli_contract.py`

Expected: FAIL because entry points are absent.

- [ ] **Step 3: Implement preflight and unique build directories**

`preflight.py` verifies Ubuntu 22.04, `ROS_DISTRO=humble`, Python 3.10, main repo path and commit, external Git root without remote, writable output root, Python Server status, active stage path, and no existing main-repo runtime directories created by this run. It records but does not clean main `git status --short`.

`build_external.sh` uses a UTC run id and executes:

```bash
source /opt/ros/humble/setup.bash
python3 scripts/preflight.py --phase build
colcon --log-base "$LUNAR_VALIDATION_ROOT/log/$RUN_ID" build \
  --base-paths /mnt/data/WS/lunar-navigation/ros2_ws/src "$LUNAR_VALIDATION_ROOT/ros2_ws/src" \
  --build-base "$LUNAR_VALIDATION_ROOT/build/$RUN_ID" \
  --install-base "$LUNAR_VALIDATION_ROOT/install/$RUN_ID" \
  --packages-up-to lunar_isaac_validation \
  --cmake-args -DBUILD_TESTING=ON
```

It writes the install path to `artifacts/<run-id>/build.json`; it does not replace a shared `current` link.

- [ ] **Step 4: Run the full external unit suite before live collection**

Run:

```bash
source /opt/ros/humble/setup.bash
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test
```

Expected: PASS.

- [ ] **Step 5: Collect one read-only live-stage snapshot**

Run:

```bash
source /opt/ros/humble/setup.bash
python3 scripts/collect_snapshot.py \
  --host 127.0.0.1 --port 8226 \
  --output-root /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/snapshots
```

Expected: manifest schema v1, 1 m/Z-up, all three platform ids, every lunar rock independently collidable, four grids with ten valid layers, equal before/after USD SHA-256, and no stage save or timeline transition.

- [ ] **Step 6: Qualify and inspect the six real scene cases**

Run:

```bash
source /opt/ros/humble/setup.bash
bash scripts/qualify_fixtures.sh \
  --manifest snapshots/<run-id>/snapshot_manifest.json
```

Expected: six cases in fixed order and every case meets its map-only evidence predicate. Inspect the generated candidate evidence and confirm expected triples are design constants rather than observed Action results. If any case is unavailable, stop and report the failed qualification; do not inject or loosen data.

- [ ] **Step 7: Build a fresh overlay and validate all capability files through Lifecycle configure**

Run:

```bash
source /opt/ros/humble/setup.bash
bash scripts/build_external.sh
source /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/install/<run-id>/setup.bash
for platform in wheeled legged hopper; do
  ros2 run lunar_planner_ros lunar_planner_node --ros-args \
    --params-file /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression/install/<run-id>/share/lunar_isaac_validation/config/planner/${platform}.yaml &
  planner_pid=$!
  planner_ready=false
  for attempt in $(seq 1 30); do
    if timeout 2 ros2 lifecycle get /lunar_planner >/dev/null 2>&1; then
      planner_ready=true
      break
    fi
    sleep 1
  done
  test "$planner_ready" = true
  ros2 lifecycle set /lunar_planner configure
  ros2 lifecycle get /lunar_planner
  ros2 lifecycle set /lunar_planner cleanup
  kill "$planner_pid"
  wait "$planner_pid" || true
done
```

Expected per platform: configure succeeds, state is `inactive`, cleanup succeeds. Track each explicit PID; do not leave overlapping `/plan_motion` servers.

- [ ] **Step 8: Commit the reviewed real-scene lock and CLI entry points**

```bash
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression add scripts ros2_ws/src/lunar_isaac_validation/scenarios/scenario_lock.json ros2_ws/src/lunar_isaac_validation/test/test_cli_contract.py
git -C /home/kai/CodexDownloads/lunar_navigation/isaac_ros_action_regression commit -m "test: lock current lunar scene Action fixtures"
```

Do not commit snapshots, build/install/log directories, candidate reports, tokens, or process logs.

---

### Task 11: Align qualification and lock migration with production planner semantics

**Files:**
- Create: external `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/planner_semantics.py`
- Modify: external `scripts/preflight.py`
- Modify: external `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/scenario_qualifier.py`
- Test: external `test/test_cli_contract.py`
- Test: external `ros2_ws/src/lunar_isaac_validation/test/{test_planner_semantics.py,test_scenario_qualifier.py,test_bridge_node.py}`

**Interfaces:**
- Consumes: `SnapshotBundle`, proxy capabilities, continuous platform planning pose, and the approved design commit.
- Produces: `ProjectedStart`, `project_trajectory_start(platform_key, request_start_position_m, bundle, capabilities)`, `lunar-scenario-lock/v2`, planner-equivalent goal-clipped hopper evidence, and explicit legacy-v1 rebaseline migration.

- [ ] **Step 1: Write failing hand-derived grid-projection tests**

Create `ros2_ws/src/lunar_isaac_validation/test/test_planner_semantics.py` with a local fixture. It uses a `0.25 m` grid with origin `(-4.125,-4.125)`, continuous XY `(0.124,-0.124)`, and hand-set start-cell elevation `1.25`:

```python
def _bundle() -> SnapshotBundle:
    wheel = GridDescriptor(
        "wheel", "odom", (-4.125, -4.125), 0.25, 33, 33
    )
    legged = replace(wheel, key="legged")
    wheel_elevation = np.zeros((33, 33), dtype=np.float64)
    legged_elevation = np.zeros((33, 33), dtype=np.float64)
    wheel_elevation[16, 16] = 1.25
    legged_elevation[16, 16] = 1.25
    return SnapshotBundle(
        Path("/fixture/manifest.json"),
        Path("/fixture/arrays.npz"),
        {},
        {"wheel": wheel, "legged": legged},
        {},
        {
            "wheel__elevation": wheel_elevation,
            "legged__elevation": legged_elevation,
        },
    )


def test_projected_start_is_derived_from_grid_and_capability() -> None:
    bundle = _bundle()
    capabilities = {"legged": {"body_height_m": [0.50, 0.60]}}
    wheel = project_trajectory_start(
        "wheel", (0.124, -0.124, 9.0), bundle, capabilities
    )
    assert wheel.cell_xy == (16, 16)
    assert wheel.position_m == pytest.approx((0.0, 0.0, 1.25))
    assert wheel.maximum_xy_error_m == pytest.approx(0.125)

    legged = project_trajectory_start(
        "legged", (0.124, -0.124, 1.80), bundle, capabilities
    )
    assert legged.cell_xy == (16, 16)
    assert legged.position_m == pytest.approx((0.0, 0.0, 1.80))
```

The legged capability body-height interval is `[0.50,0.60]`, so the expected Z is `1.25+0.55=1.80`. The production mutation caught by this test is returning the continuous request pose instead of the grid-cell projection.

Run:

```bash
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_planner_semantics.py
```

Expected: FAIL importing the missing module or function.

- [ ] **Step 2: Implement the minimal projection contract**

Define:

```python
class PlannerSemanticsError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectedStart:
    cell_xy: tuple[int, int]
    position_m: tuple[float, float, float]
    maximum_xy_error_m: float


def project_trajectory_start(
    platform_key: str,
    request_start_position_m: Sequence[float],
    bundle: SnapshotBundle,
    capabilities: Mapping[str, object],
) -> ProjectedStart:
    if platform_key not in {"wheel", "legged"}:
        raise PlannerSemanticsError("PLANNER_SEMANTICS_PLATFORM_INVALID")
    if len(request_start_position_m) != 3:
        raise PlannerSemanticsError("PLANNER_SEMANTICS_START_INVALID")
    start = tuple(float(value) for value in request_start_position_m)
    if not all(math.isfinite(value) for value in start):
        raise PlannerSemanticsError("PLANNER_SEMANTICS_START_INVALID")
    descriptor = bundle.grids.get(platform_key)
    if descriptor is None:
        raise PlannerSemanticsError("PLANNER_SEMANTICS_GRID_MISSING")
    cell_x = math.floor(
        (start[0] - descriptor.origin_xy_m[0]) / descriptor.resolution_m
    )
    cell_y = math.floor(
        (start[1] - descriptor.origin_xy_m[1]) / descriptor.resolution_m
    )
    if not (0 <= cell_x < descriptor.width and 0 <= cell_y < descriptor.height):
        raise PlannerSemanticsError("PLANNER_SEMANTICS_START_OUTSIDE_GRID")
    center_x = descriptor.origin_xy_m[0] + (cell_x + 0.5) * descriptor.resolution_m
    center_y = descriptor.origin_xy_m[1] + (cell_y + 0.5) * descriptor.resolution_m
    elevation = float(bundle.arrays[f"{platform_key}__elevation"][cell_y, cell_x])
    center_z = elevation
    if platform_key == "legged":
        document = capabilities.get("legged")
        if not isinstance(document, Mapping):
            raise PlannerSemanticsError("PLANNER_SEMANTICS_CAPABILITY_INVALID")
        section = document.get("legged", document)
        interval = section.get("body_height_m") if isinstance(section, Mapping) else None
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise PlannerSemanticsError("PLANNER_SEMANTICS_CAPABILITY_INVALID")
        lower, upper = (float(value) for value in interval)
        if not all(math.isfinite(value) for value in (lower, upper)) or lower > upper:
            raise PlannerSemanticsError("PLANNER_SEMANTICS_CAPABILITY_INVALID")
        center_z += 0.5 * (lower + upper)
    maximum_error = 0.5 * descriptor.resolution_m
    if (
        abs(center_x - start[0]) > maximum_error + 1.0e-9
        or abs(center_y - start[1]) > maximum_error + 1.0e-9
    ):
        raise PlannerSemanticsError("PLANNER_SEMANTICS_PROJECTION_INVALID")
    return ProjectedStart(
        cell_xy=(cell_x, cell_y),
        position_m=(center_x, center_y, center_z),
        maximum_xy_error_m=maximum_error,
    )
```

Accept only `wheel` and `legged`. Use `floor((position-origin)/resolution)`, the selected local cell centre, and that cell's elevation. Wheel Z is elevation; legged Z is elevation plus the midpoint of `body_height_m`. Reject nonfinite/out-of-grid inputs and any per-axis XY displacement greater than `resolution/2+1e-9`; never inspect an Action result.

Run the Step 1 test and expect PASS.

- [ ] **Step 3: Write failing v2 lock, projected-evidence, and hopper-clipping tests**

Extend `test_scenario_qualifier.py` with behavior tests that assert:

```python
lock, _ = qualify_snapshot(synthetic_bundle(), capability_documents())
assert lock.schema_version == "lunar-scenario-lock/v2"

wheel = lock.case("wheel-positive")
assert wheel.evidence["start_cell_xy"] == [16, 16]
assert wheel.evidence["projected_start_position_m"] == [0.0, 0.0, 0.0]
assert wheel.evidence["start_projection_maximum_xy_error_m"] == 0.125

hopper = lock.case("hopper-positive")
assert hopper.goal_tolerance_m == 0.75
assert hopper.evidence["landing_goal_tolerance_m"] == 0.75
assert hopper.evidence["landing_candidate_cell_count"] == 29
assert hopper.evidence["landing_seed_cell_xy"] == [16, 12]
assert hopper.evidence["landing_rectangle_area_m2"] == 1.5625
assert hopper.evidence["landing_rectangle_cell_bounds_xyxy"] == [14, 10, 18, 14]
```

Add a second test that sets `minimum_landing_region_area_m2=1.562501` on the otherwise globally safe synthetic map and requires `SCENARIO_CASE_NOT_FOUND:hopper-positive`. This catches implementations that use safe cells outside the Action goal tolerance.

Add a migration test: serialize a valid lock with schema changed to `lunar-scenario-lock/v1`; normal `load_scenario_lock` must reject it with `SCENARIO_LOCK_INVALID`, while `write_scenario_lock(..., allow_rebaseline=True, reason="approved semantic migration")` must archive the v1 payload and replace it with v2. Run:

```bash
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_scenario_qualifier.py
```

Expected: FAIL on the v1 schema, missing evidence, `0.50` hopper-positive tolerance, or unbounded landing area.

- [ ] **Step 4: Implement v2 qualification and explicit legacy migration**

Set `SCENARIO_LOCK_SCHEMA="lunar-scenario-lock/v2"`, `LEGACY_SCENARIO_LOCK_SCHEMA="lunar-scenario-lock/v1"`, `QUALIFICATION_EVIDENCE_SCHEMA="lunar-scenario-qualification-evidence/v2"`, default goal tolerance `0.50`, and hopper-positive tolerance `0.75`.

For wheel and legged cases, call `project_trajectory_start` and store its three exact evidence fields. V2 parsing must reject missing, nonfinite, or malformed projection evidence. Default lock loading accepts only v2. The rebaseline writer alone may validate and archive a structurally valid v1 lock before replacing it with a freshly qualified v2 lock; no normal run path accepts v1. Update the bridge-node test's hand-authored v2 lock fixture with literal, hand-derived projection evidence; do not call the production projection helper to construct expected test data.

Replace the global `_rectangle_for_cell` decision with a goal-clipped mask. Select the safe seed by `(distance_to_goal,y,x)`, then repeatedly try `minimum_x`, `maximum_x`, `minimum_y`, and `maximum_y` expansions in that order. Store `landing_goal_tolerance_m`, `landing_candidate_cell_count`, `landing_seed_cell_xy`, the final bounds, and final area. A candidate qualifies only when the clipped area and all existing dynamics checks pass.

- [ ] **Step 5: Bind preflight to the approved semantic authority**

First add a failing `test_cli_contract.py` assertion that `APPROVED_MAIN_COMMITS` contains `a02601a8112f3417f57de9fb7df4d3cf2237b705`, then append that exact commit to `scripts/preflight.py`. Run:

```bash
python3 -m pytest -q test/test_cli_contract.py
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_planner_semantics.py \
  ros2_ws/src/lunar_isaac_validation/test/test_scenario_qualifier.py
```

Expected: both commands PASS.

- [ ] **Step 6: Run source verification and commit Task 11**

```bash
python3 -m pytest -q test ros2_ws/src/lunar_isaac_validation/test
python3 -m py_compile \
  scripts/preflight.py \
  ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/planner_semantics.py \
  ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/scenario_qualifier.py
git diff --check
git add scripts/preflight.py test/test_cli_contract.py ros2_ws/src/lunar_isaac_validation
git commit -m "fix: align locked fixtures with planner semantics"
```

Expected: all source tests PASS and only Task 11 source/test files enter the commit; the old live lock remains unchanged until Task 13.

---

### Task 12: Validate projected starts and complete six-case repeat semantics

**Files:**
- Modify: external `ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/{trajectory_checks.py,reports.py}`
- Test: external `ros2_ws/src/lunar_isaac_validation/test/{test_trajectory_checks.py,test_reports.py}`

**Interfaces:**
- Consumes: `ScenarioCase` v2 evidence and `project_trajectory_start`.
- Produces: trajectory-start validation independent of Action output and `compare_normalized_runs` that cannot pass missing or failed case summaries.

- [ ] **Step 1: Write and observe failing projected-start assertion tests**

Keep the existing valid fixture centred by changing its test-only grid origin to `(-5.0625,-5.0625)`. Add a separate wheel case on origin `(-5.0,-5.0)`, resolution `0.125`, continuous request start `(0.0,0.0,0.0)`, and locked evidence `cell=[40,40]`, projected position `[0.0625,0.0625,0.0]`, maximum per-axis error `0.0625`. A reference whose first point is that projected position must pass; changing either the first point back to `(0.0,0.0,0.0)` or the locked projection evidence must raise `TRAJECTORY_START_MISMATCH` or `TRAJECTORY_START_EVIDENCE_MISMATCH`, respectively.

Run:

```bash
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_trajectory_checks.py
```

Expected: the projected-start acceptance test fails because the current validator compares against the continuous request pose.

- [ ] **Step 2: Implement projected-start validation**

For wheel and legged references, recompute `ProjectedStart` from the case request start, bundle, and capabilities. Require the lock's three evidence fields to equal the recomputed cell, position within `1e-9 m`, and maximum error within `1e-12 m`; otherwise raise `TRAJECTORY_START_EVIDENCE_MISMATCH`. Compare the first reference point to the recomputed projected position within `1e-3 m`; retain all terminal, kinematic, timing, sweep, and map checks unchanged.

Run the Step 1 command and expect PASS.

- [ ] **Step 3: Write and observe a failing missing-summary comparison test**

Extend `test_reports.py` so a six-case pair with either `passed=False`, an absent `normalized_reference`, or `normalized_reference=None` returns the affected case id in `mismatches`. Existing comparisons must continue to ignore duration, ROS stamps, PIDs, logs, and assertion-detail text.

Run:

```bash
python3 -m pytest -q ros2_ws/src/lunar_isaac_validation/test/test_reports.py
```

Expected: FAIL because the current comparator treats two missing summaries as equal.

- [ ] **Step 4: Implement complete semantic comparison and commit Task 12**

Require equal six-case order, both cases passing, and a mapping-valued `normalized_reference` on both sides before comparing summaries. Then run:

```bash
python3 -m pytest -q \
  ros2_ws/src/lunar_isaac_validation/test/test_trajectory_checks.py \
  ros2_ws/src/lunar_isaac_validation/test/test_action_assertions.py \
  ros2_ws/src/lunar_isaac_validation/test/test_reports.py
python3 -m pytest -q test ros2_ws/src/lunar_isaac_validation/test
python3 -m py_compile \
  ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/trajectory_checks.py \
  ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/reports.py
git diff --check
git add ros2_ws/src/lunar_isaac_validation
git commit -m "fix: validate projected trajectory starts"
```

Expected: all commands PASS and the source Git worktree is clean.

---

### Task 13: Rebaseline, run the six cases twice, and document handoff

**Files:**
- Modify through approved CLI: external `ros2_ws/src/lunar_isaac_validation/scenarios/scenario_lock.json`
- Modify: external `README.md`
- Create at runtime: external `build/`, `install/`, `log/`, and `artifacts/` evidence only
- Preserve: active USD, locked snapshot arrays, production planner source, and unrelated main-repository state

**Interfaces:**
- Consumes: snapshot `20260804T025645914927Z`, v1 lock, Task 11–12 source, and approved rebaseline reason.
- Produces: reviewed v2 lock/report, fresh install overlay, two complete 6/6 runs, deterministic semantic comparison, final evidence paths, and operator instructions.

- [ ] **Step 1: Explicitly rebaseline the reviewed lock**

From the external root, run in one shell:

```bash
LUNAR_REBASELINE_RUN_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
bash scripts/qualify_fixtures.sh \
  --manifest snapshots/20260804T025645914927Z/snapshot_manifest.json \
  --run-id "$LUNAR_REBASELINE_RUN_ID" \
  --rebaseline \
  --reason "2026-08-04 approved planner-semantic alignment: grid-projected starts and goal-clipped hopper landing region"
printf '%s\n' "$LUNAR_REBASELINE_RUN_ID"
```

Expected: a v1→v2 `scenario_rebaseline_report.json`; unchanged stage and arrays hashes; wheel/legged projected-start evidence; hopper-positive tolerance `0.75` and goal-clipped area at least `1.327322 m²`. Commit only the lock:

```bash
git add ros2_ws/src/lunar_isaac_validation/scenarios/scenario_lock.json
git commit -m "test: rebaseline planner-aligned scene fixtures"
```

- [ ] **Step 2: Build one fresh external overlay and run its tests**

```bash
LUNAR_BUILD_RUN_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
bash scripts/build_external.sh --run-id "$LUNAR_BUILD_RUN_ID"
set +u
source /opt/ros/humble/setup.bash
source "install/$LUNAR_BUILD_RUN_ID/setup.bash"
set -u
colcon --log-base "log/${LUNAR_BUILD_RUN_ID}-test" test \
  --build-base "build/$LUNAR_BUILD_RUN_ID" \
  --install-base "install/$LUNAR_BUILD_RUN_ID" \
  --packages-select lunar_planner_core lunar_planner_ros lunar_isaac_validation
colcon test-result \
  --test-result-base "build/$LUNAR_BUILD_RUN_ID" --verbose
printf '%s\n' "$LUNAR_BUILD_RUN_ID"
```

Expected: build succeeds and test-result reports zero failed tests.

- [ ] **Step 3: Run two formal six-case regressions against the same lock and install**

In the same shell that retains `LUNAR_BUILD_RUN_ID`:

```bash
LUNAR_FORMAL_RUN_ONE_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
bash scripts/run_action_regression.sh \
  --manifest snapshots/20260804T025645914927Z/snapshot_manifest.json \
  --lock ros2_ws/src/lunar_isaac_validation/scenarios/scenario_lock.json \
  --install "install/$LUNAR_BUILD_RUN_ID" \
  --run-id "$LUNAR_FORMAL_RUN_ONE_ID"

LUNAR_FORMAL_RUN_TWO_ID="$(date -u +%Y%m%dT%H%M%S%6NZ)"
bash scripts/run_action_regression.sh \
  --manifest snapshots/20260804T025645914927Z/snapshot_manifest.json \
  --lock ros2_ws/src/lunar_isaac_validation/scenarios/scenario_lock.json \
  --install "install/$LUNAR_BUILD_RUN_ID" \
  --run-id "$LUNAR_FORMAL_RUN_TWO_ID"
printf '%s\n' "$LUNAR_FORMAL_RUN_ONE_ID" "$LUNAR_FORMAL_RUN_TWO_ID"
```

Expected for each run: exit 0; six isolated sessions; three exact new-reference and three exact goal-infeasible results; six cleanup assertions pass; no stale/TF/skew/layout failure.

- [ ] **Step 4: Compare all six normalized summaries**

Load both `scenario_results.json` files as `CaseResult` values and call `compare_normalized_runs`. Assert both `summary.json` objects equal `{"exit_code":0,"failed":0,"passed":6,"total":6}`, all twelve case results have mapping-valued `normalized_reference`, and comparison equals `{"passed":True,"mismatches":[]}`. Timing, ROS stamps, PIDs, and raw logs remain excluded.

In the same shell that retains both formal run ids, run:

```bash
PYTHONPATH=ros2_ws/src/lunar_isaac_validation python3 - \
  "$LUNAR_FORMAL_RUN_ONE_ID" "$LUNAR_FORMAL_RUN_TWO_ID" <<'PY'
from collections.abc import Mapping
import json
from pathlib import Path
import sys

from lunar_isaac_validation.reports import CaseResult, compare_normalized_runs

expected_summary = {"exit_code": 0, "failed": 0, "passed": 6, "total": 6}
runs = []
for run_id in sys.argv[1:]:
    directory = Path("artifacts") / run_id
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary != expected_summary:
        raise SystemExit(f"unexpected summary for {run_id}: {summary}")
    payload = json.loads(
        (directory / "scenario_results.json").read_text(encoding="utf-8")
    )
    results = [CaseResult(**item) for item in payload["results"]]
    if len(results) != 6 or not all(
        result.passed
        and isinstance(result.actual.get("normalized_reference"), Mapping)
        for result in results
    ):
        raise SystemExit(f"incomplete semantic results for {run_id}")
    runs.append(results)
comparison = compare_normalized_runs(runs[0], runs[1])
if comparison != {"passed": True, "mismatches": []}:
    raise SystemExit(f"normalized mismatch: {comparison}")
print(json.dumps(comparison, sort_keys=True))
PY
```

- [ ] **Step 5: Verify immutable scene, artifact boundary, and main repository**

```bash
test "$(sha256sum /home/kai/CodexDownloads/lunar_navigation/isaac_sim/exports/lunar_polar_terrain_5deg/lunar_polar_terrain_5deg_safe_wheeled_start.usda | cut -d' ' -f1)" = "0e5de317252740d691a852b69d5304a4150e26b4f2741a42149627a3016196b8"
test "$(sha256sum snapshots/20260804T025645914927Z/snapshot_arrays.npz | cut -d' ' -f1)" = "5853a6a8499c3663828549c155d278ddab4a31e767b93f15860e6d282c4c7ac4"
cd /mnt/data/WS/lunar-navigation
python3 tools/check_repository_boundaries.py .
python3 -m pytest -q tests/foundation/test_repository_boundaries.py
git status --short
```

Expected: both hashes match, boundary checks pass, and the only unrelated main entry remains `?? .vscode/`. All new runtime artifacts remain below the external root.

- [ ] **Step 6: Document the exact operator workflow**

README must explain: Python Server enable/check and optional `0600` token file; collect; qualify once; v1→v2 explicit rebaseline and required reason; v2 grid-projected wheel/legged start oracle; `0.75 m` goal-clipped hopper-positive rule; build; two formal runs; report locations; exit codes 10/20/30/40/50/60; six-session isolation; replay after post-snapshot Isaac disconnect; and the absence of platform execution/control.

- [ ] **Step 7: Run final external verification and commit the handoff**

```bash
python3 -m pytest -q test ros2_ws/src/lunar_isaac_validation/test
python3 -m py_compile \
  scripts/*.py \
  ros2_ws/src/lunar_isaac_validation/lunar_isaac_validation/*.py
git diff --check
git status --short
git add README.md
git commit -m "docs: document Isaac ROS regression workflow"
git status --short
```

Expected: tests and compilation PASS; external source Git worktree is clean; ignored runtime artifacts remain recoverable.

---

## Plan Self-Review

- **Spec coverage:** Tasks 1–4 create the isolated two-runtime snapshot path; Task 5 creates all capability and geometry inputs; Task 6 locks six real cases without data injection; Task 7 publishes the frozen topics; Task 8 validates platform references; Task 9 handles Lifecycle, Action, cleanup, and reports; Task 10 qualifies the live scene; Tasks 11–13 implement the approved planner-semantic alignment, explicit v2 rebaseline, and two complete deterministic regressions.
- **Boundary coverage:** Every implementation source/build/output path is external; the main repository contains only the approved design and plan. Production planner source remains read-only, all pre-existing unrelated worktree entries are preserved, and final repository boundary checks are explicit.
- **Failure coverage:** Preflight, Isaac, fixture, ROS, Action, and cleanup failures map to exit codes 10/20/30/40/50/60. Exact timeouts, continuation rules, token redaction, post-snapshot Isaac disconnect, and explicit-PID cleanup are assigned to Tasks 4, 9, and 10.
- **Type consistency:** `SnapshotBundle`, `ProjectedStart`, `ScenarioCase`, `ScenarioLock`, `CaseResult`, layer keys, platform keys, topic names, Action enums, reason codes, frames, and public function names are defined before downstream use and remain identical across tasks.
- **Placeholder scan:** All production values, paths, commands, expected results, algorithms, and test oracles are fixed; the plan contains no deferred implementation marker.
