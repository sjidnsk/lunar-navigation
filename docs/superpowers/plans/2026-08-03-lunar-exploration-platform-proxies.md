# Lunar Exploration Platform Proxies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified Isaac Sim USD derivative with polar illumination, independently collidable rocks, and three parameterized physical exploration-platform proxies.

**Architecture:** A single external Python authoring script opens the prepared terrain USD in Isaac Sim, validates its static terrain collider, then creates the lighting, rock collision APIs, platform compounds and custom capability data. It saves only a new derivative USD. A separate external verifier opens the derivative and checks every required API, count, parameter and transform without changing the stage.

**Tech Stack:** Isaac Sim 6.0.1 Python server, USD (`pxr.Usd`, `UsdGeom`, `UsdLux`, `UsdPhysics`), PhysX schema APIs, Python 3.10.

## Global Constraints

- All generated scripts, USD files and reports remain outside the repository under `~/CodexDownloads/lunar_navigation/isaac_sim/`.
- Input is the existing `lunar_polar_terrain_5deg_physics.usda`; output is a distinct `lunar_polar_terrain_5deg_platform_proxies.usda` derivative.
- Preserve the terrain's static triangle-mesh collision and `1.62 m/s²` PhysicsScene gravity.
- Add independent static convex-hull collisions to all 260 lunar rocks; do not apply `RigidBodyAPI` to rock roots.
- Add three proxy roots only under `/World/LunarExplorationPlatforms`; no URDFs, articulation, controller, sensor, ROS bridge, asset download or `.blend` save.
- Maintain 5° solar elevation and a single shadow-casting directional sun; do not add ambient fill light.
- Do not stage or commit generated external files. Preserve existing unrelated `AGENTS.md` and `.vscode/` changes.

---

## File Structure

- Create: `~/CodexDownloads/lunar_navigation/isaac_sim/create_lunar_platform_proxies.py` — idempotently authors the derivative USD.
- Create: `~/CodexDownloads/lunar_navigation/isaac_sim/verify_lunar_platform_proxies.py` — read-only USD verifier.
- Create: `~/CodexDownloads/lunar_navigation/isaac_sim/exports/lunar_polar_terrain_5deg/lunar_polar_terrain_5deg_platform_proxies.usda` — generated derivative only.
- Preserve: `~/CodexDownloads/lunar_navigation/isaac_sim/exports/lunar_polar_terrain_5deg/lunar_polar_terrain_5deg_physics.usda` — input baseline remains unchanged.
- Preserve: `docs/interfaces/external-input-baseline.md` — proxy values are not an external interface contract.

### Task 1: Write a failing verifier for the desired USD state

**Files:**
- Create: `~/CodexDownloads/lunar_navigation/isaac_sim/verify_lunar_platform_proxies.py`

**Interfaces:**
- Consumes: `verify(stage: Usd.Stage) -> list[str]` and the output USD path.
- Produces: exit code 0 only when all checks pass; otherwise prints each failed invariant and exits 1.

- [ ] **Step 1: Define exact expected platform capabilities and a validation entry point**

```python
EXPECTED = {
    "WheeledScout": {"platform_type": "WHEELED", "mass_kg": 50.0,
                      "clearance_m": 0.22, "max_slope_deg": 18.0},
    "LeggedScout": {"platform_type": "LEGGED", "mass_kg": 35.0,
                     "body_height_m": 0.55, "max_step_height_m": 0.22},
    "HopperScout": {"platform_type": "HOPPER", "mass_kg": 20.0,
                     "landing_region_radius_m": 0.65,
                     "max_launch_speed_mps": 2.20},
}

def verify(stage: Usd.Stage) -> list[str]:
    errors: list[str] = []
    # append one string per violated invariant
    return errors
```

- [ ] **Step 2: Run the verifier before authoring**

Run:

```bash
python3 /home/kai/CodexDownloads/lunar_navigation/isaac_sim/verify_lunar_platform_proxies.py \
  /home/kai/CodexDownloads/lunar_navigation/isaac_sim/exports/lunar_polar_terrain_5deg/lunar_polar_terrain_5deg_physics.usda
```

Expected: nonzero exit and missing platform / rock-collision failures.

- [ ] **Step 3: Implement rock, terrain, light and platform validation**

The verifier must assert exactly 260 `LunarPolarRock_*` mesh prims, each has `UsdPhysics.CollisionAPI` and `MeshCollisionAPI` with approximation `convexHull`, and no rock root has `RigidBodyAPI`. It must assert the terrain mesh has collision approximation `none`, PhysicsScene gravity magnitude is `1.62`, the one DistantLight has `shadow:enable=True` and a normalized direction whose absolute vertical component differs from `sin(5°)` by less than `0.001`. For each expected root, assert `RigidBodyAPI`, a `physics:mass` value, matching custom data, at least one child collider and a root translate with finite coordinates.

- [ ] **Step 4: Syntax-check the verifier**

Run:

```bash
python3 -m py_compile /home/kai/CodexDownloads/lunar_navigation/isaac_sim/verify_lunar_platform_proxies.py
```

Expected: exit code 0.

### Task 2: Implement idempotent USD authoring

**Files:**
- Create: `~/CodexDownloads/lunar_navigation/isaac_sim/create_lunar_platform_proxies.py`
- Modify: runtime stage only, then create output USD in the external exports directory.

**Interfaces:**
- Consumes: `INPUT_USD`, `OUTPUT_USD`, `make_platform(stage, spec) -> Usd.Prim`, and `ground_z(stage, x_m, y_m) -> float`.
- Produces: an authored stage containing `LunarExplorationPlatforms`, three named proxy roots and 260 independently collidable rocks.

- [ ] **Step 1: Add source and stage guards**

```python
INPUT_USD = Path("/home/kai/CodexDownloads/lunar_navigation/isaac_sim/exports/lunar_polar_terrain_5deg/lunar_polar_terrain_5deg_physics.usda")
OUTPUT_USD = INPUT_USD.with_name("lunar_polar_terrain_5deg_platform_proxies.usda")
ROCK_COUNT = 260

def require_prepared_terrain(stage: Usd.Stage) -> Usd.Prim:
    meshes = [p for p in Usd.PrimRange(stage.GetPseudoRoot())
              if p.IsA(UsdGeom.Mesh) and "LunarPolarTerrain" in str(p.GetPath())]
    if len(meshes) != 1 or not meshes[0].HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Prepared static lunar terrain collider is required")
    return meshes[0]
```

The script must fail before saving if the input USD is absent, is already the output path, lacks the terrain collider, lacks 260 rock roots or has a gravity magnitude outside `1.62 ± 0.001`.

- [ ] **Step 2: Add one collision API to each rock mesh**

```python
def configure_static_rock(mesh: Usd.Prim) -> None:
    UsdPhysics.CollisionAPI.Apply(mesh)
    collision = UsdPhysics.MeshCollisionAPI.Apply(mesh)
    collision.CreateApproximationAttr().Set("convexHull")
```

Enumerate by prim name, require one mesh under every rock root, call this function once per mesh, and reject a root that has `RigidBodyAPI`.

- [ ] **Step 3: Configure the 5° shadow-casting sun**

Create or update only `/World/LunarPolarSun/LunarPolarSun_Light_002`. Set intensity `2000.0`, angle `0.53`, `shadow:enable=True`, and rotate it so its beam direction is `(cos(5°)*cos(28°), cos(5°)*sin(28°), -sin(5°))`. Delete no other light prim and create no dome/ambient light.

- [ ] **Step 4: Create the three compound proxies from exact specs**

```python
PLATFORM_SPECS = (
    {"name": "WheeledScout", "platform_type": "WHEELED", "mass_kg": 50.0,
     "xy": (-38.0, -30.0), "clearance_m": 0.22, "max_slope_deg": 18.0,
     "max_obstacle_height_m": 0.18, "max_speed_mps": 0.70,
     "max_acceleration_mps2": 0.40, "min_turn_radius_m": 0.70,
     "motion_primitives": "arc_and_line"},
    {"name": "LeggedScout", "platform_type": "LEGGED", "mass_kg": 35.0,
     "xy": (-30.0, -30.0), "body_height_m": 0.55, "clearance_m": 0.25,
     "max_slope_deg": 28.0, "max_roughness_m": 0.12,
     "max_step_height_m": 0.22, "max_speed_mps": 0.45,
     "max_acceleration_mps2": 0.35, "motion_primitives": "body_lattice"},
    {"name": "HopperScout", "platform_type": "HOPPER", "mass_kg": 20.0,
     "xy": (-22.0, -30.0), "landing_clearance_m": 0.25,
     "max_landing_slope_deg": 15.0, "max_landing_roughness_m": 0.08,
     "landing_region_radius_m": 0.65, "max_launch_speed_mps": 2.20,
     "max_flight_time_s": 3.00, "minimum_settle_guard_s": 1.00,
     "actuator_or_impulse_profile": "proxy_impulse_v1"},
)
```

For every root, apply `UsdPhysics.RigidBodyAPI` and `UsdPhysics.MassAPI`, set `physics:mass`, record all capability values as custom data, and position it using triangle interpolation from the terrain mesh. Build only child box, cylinder or sphere meshes; apply `CollisionAPI` and `MeshCollisionAPI` with `convexHull` to every child. Create four visual/collider wheels for the wheeled proxy, one body plus six fixed leg/foot colliders for the legged proxy, and one body plus fixed leg/foot collider for the hopper proxy. Do not apply an articulation API or create joints.

- [ ] **Step 5: Create a new USD derivative and ensure idempotence**

Open the input into a new stage, remove only an existing `/World/LunarExplorationPlatforms` scope from that in-memory copy, create the scope and proxies, save as `OUTPUT_USD`, and reopen the output. Do not overwrite `INPUT_USD` and do not modify a currently open interactive stage until the generated USD passes verification.

- [ ] **Step 6: Syntax-check the authoring script**

Run:

```bash
python3 -m py_compile /home/kai/CodexDownloads/lunar_navigation/isaac_sim/create_lunar_platform_proxies.py
```

Expected: exit code 0.

### Task 3: Author, verify, then open only the verified derivative in Isaac Sim

**Files:**
- Create: `~/CodexDownloads/lunar_navigation/isaac_sim/exports/lunar_polar_terrain_5deg/lunar_polar_terrain_5deg_platform_proxies.usda`
- Modify: active Isaac Sim stage only after local verification passes.

**Interfaces:**
- Consumes: Task 1 verifier and Task 2 authoring script.
- Produces: a verified active Isaac Sim stage rooted at the output USD.

- [ ] **Step 1: Run the authoring script with Isaac Sim Python**

Run:

```bash
/home/kai/isaacsim-6.0.1/python.sh \
  /home/kai/CodexDownloads/lunar_navigation/isaac_sim/create_lunar_platform_proxies.py
```

Expected: a new output USD path is printed, with 260 rock colliders and three proxy roots.

- [ ] **Step 2: Run the verifier against the derivative**

Run:

```bash
/home/kai/isaacsim-6.0.1/python.sh \
  /home/kai/CodexDownloads/lunar_navigation/isaac_sim/verify_lunar_platform_proxies.py \
  /home/kai/CodexDownloads/lunar_navigation/isaac_sim/exports/lunar_polar_terrain_5deg/lunar_polar_terrain_5deg_platform_proxies.usda
```

Expected: exit code 0 and output including `rocks=260`, all three platform names, lunar gravity, and 5° sun evidence.

- [ ] **Step 3: Open the verified derivative through the authenticated Isaac Python server**

Use the local `127.0.0.1:8226` Python server with its token read locally (never print it). Execute:

```python
success, error = await omni.usd.get_context().open_stage_async(
    "/home/kai/CodexDownloads/lunar_navigation/isaac_sim/exports/lunar_polar_terrain_5deg/lunar_polar_terrain_5deg_platform_proxies.usda"
)
assert success, error
```

Expected: the interactive stage root equals the derivative path.

- [ ] **Step 4: Re-run read-only live-stage verification**

Run the same `verify()` invariants in the Python server against `omni.usd.get_context().get_stage()`. Expected: all checks pass and the server reports no unsaved session-only platform changes.

## Plan Self-Review

- Spec coverage: Task 1 states the checks that must initially fail; Task 2 authors all approved light, rock and platform requirements; Task 3 verifies the derivative before and after it becomes the active stage.
- Placeholder scan: all output paths, prim names, parameter values, APIs and commands are explicit; no deferred implementation or undefined interface remains.
- Type consistency: Task 1 and Task 2 share the same named roots, rock count, USD APIs, parameter values and output USD path; Task 3 runs the verifier against that exact artifact.
