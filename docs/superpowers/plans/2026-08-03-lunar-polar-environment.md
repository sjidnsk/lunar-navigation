# Lunar Polar Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 120×120 m lunar-polar Blender environment with a permanently shadowed crater floor, layered craters, irregular rocks, and a local CC0 PBR material.

**Architecture:** The asset layer is confined to `~/CodexDownloads/lunar_navigation/blender/assets/` and is recorded by a local manifest. A new approved Blender-MCP script validates that layer before rebuilding only `LunarEnvironment`; it creates terrain and shared rock meshes deterministically, then binds a PBR material and a low-angle sun. Verification uses Blender MCP read-only scene queries after execution.

**Tech Stack:** Blender 4.5 Python API (`bpy`, `bmesh`), Blender MCP approved-script mode, Python 3.10 syntax checking, Poly Haven `moon_meteor_01` 2K CC0 texture set, SHA-256.

## Global Constraints

- Keep the scene in metric metres: 1 Blender Unit = 1 metre.
- Keep all downloaded files and generated scripts outside the repository, under `~/CodexDownloads/lunar_navigation/blender/`.
- Download only the four 2K texture maps specified below; do not import an external `.blend` model.
- Rebuild only `LunarEnvironment`; do not alter the default Cube, Camera, or Light.
- Do not save or export a `.blend` file.
- Use deterministic seeds and named objects so reruns replace prior generated content predictably.
- Confirm every downloaded file's SHA-256 in the external manifest; source is CC0.

---

## File Structure

- Create: `~/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/` — local texture maps and immutable provenance manifest.
- Create: `~/CodexDownloads/lunar_navigation/blender/create_lunar_polar_terrain.py` — approved Blender MCP script that validates assets, rebuilds the environment, and never saves or exports.
- Preserve: `~/CodexDownloads/lunar_navigation/blender/create_lunar_terrain.py` — existing non-polar generator remains a fallback and is not edited.
- Preserve: `docs/superpowers/specs/2026-08-03-lunar-polar-environment-design.md` — approved design.

### Task 1: Acquire and attest the CC0 PBR texture set

**Files:**
- Create: `~/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/moon_meteor_01_diff_2k.jpg`
- Create: `~/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/moon_meteor_01_disp_2k.png`
- Create: `~/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/moon_meteor_01_nor_gl_2k.exr`
- Create: `~/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/moon_meteor_01_rough_2k.exr`
- Create: `~/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/ASSET_MANIFEST.md`

**Interfaces:**
- Produces: an asset directory consumed by `require_pbr_assets(asset_dir: pathlib.Path) -> dict[str, pathlib.Path]`.
- Produces: exact local keys `diffuse`, `displacement`, `normal`, and `roughness`.

- [ ] **Step 1: Create the dedicated external asset directory**

Run:

```bash
mkdir -p /home/kai/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k
```

- [ ] **Step 2: Download exactly the four reviewed 2K files into that directory**

Run each command with `--fail --location --remote-time --output` and the final filename:

```bash
curl --fail --location --remote-time \
  --output /home/kai/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/moon_meteor_01_diff_2k.jpg \
  https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/moon_meteor_01/moon_meteor_01_diff_2k.jpg
curl --fail --location --remote-time \
  --output /home/kai/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/moon_meteor_01_disp_2k.png \
  https://dl.polyhaven.org/file/ph-assets/Textures/png/2k/moon_meteor_01/moon_meteor_01_disp_2k.png
curl --fail --location --remote-time \
  --output /home/kai/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/moon_meteor_01_nor_gl_2k.exr \
  https://dl.polyhaven.org/file/ph-assets/Textures/exr/2k/moon_meteor_01/moon_meteor_01_nor_gl_2k.exr
curl --fail --location --remote-time \
  --output /home/kai/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/moon_meteor_01_rough_2k.exr \
  https://dl.polyhaven.org/file/ph-assets/Textures/exr/2k/moon_meteor_01/moon_meteor_01_rough_2k.exr
```

- [ ] **Step 3: Verify the supplier's MD5 values and calculate SHA-256 values**

Run:

```bash
cd /home/kai/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k
printf '%s  %s\n' \
  405b7ad08fb5ee17a753d5be03b3737a moon_meteor_01_diff_2k.jpg \
  139f923456f5078325c3cb4e60476751 moon_meteor_01_disp_2k.png \
  fc9dd681a25ab0138f444231b899c01d moon_meteor_01_nor_gl_2k.exr \
  f1a9a256bd29117df6405f2a7b780f30 moon_meteor_01_rough_2k.exr | md5sum --check
sha256sum moon_meteor_01_diff_2k.jpg moon_meteor_01_disp_2k.png moon_meteor_01_nor_gl_2k.exr moon_meteor_01_rough_2k.exr
```

Expected: four `OK` MD5 lines and four SHA-256 digests.

- [ ] **Step 4: Write the external provenance manifest with source, licence, sizes, and the computed SHA-256 values**

The manifest must identify `moon_meteor_01`, source `https://polyhaven.com/a/moon_meteor_01`, license `CC0 1.0`, the four source URLs, and the output of Step 3. It must state that no `.blend` asset was downloaded or imported.

- [ ] **Step 5: Verify directory scope and manifest encoding**

Run:

```bash
find /home/kai/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k -maxdepth 1 -type f -printf '%f\n' | sort
python3 -c "from pathlib import Path; Path('/home/kai/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k/ASSET_MANIFEST.md').read_text(encoding='utf-8'); print('UTF-8 OK')"
```

Expected: exactly four texture maps and `ASSET_MANIFEST.md`; no assets appear in the repository.

### Task 2: Implement the deterministic polar terrain generator

**Files:**
- Create: `~/CodexDownloads/lunar_navigation/blender/create_lunar_polar_terrain.py`
- Test: `python3 -m py_compile ~/CodexDownloads/lunar_navigation/blender/create_lunar_polar_terrain.py`

**Interfaces:**
- Consumes: `require_pbr_assets(asset_dir: pathlib.Path) -> dict[str, pathlib.Path]` from the external asset directory.
- Produces: `main() -> None`, `LunarPolarTerrain`, `LunarPolarSun`, and `LunarPolarRock_000` through `LunarPolarRock_259` in collection `LunarEnvironment`.
- Produces: material `LunarPolar_Regolith_PBR`.

- [ ] **Step 1: Write the executable acceptance checks before building geometry**

At the top of the script define these constants and validation function:

```python
ENVIRONMENT_COLLECTION = "LunarEnvironment"
TERRAIN_NAME = "LunarPolarTerrain"
SUN_NAME = "LunarPolarSun"
ROCK_COUNT = 260
RANDOM_SEED = 20260803
ASSET_DIR = Path("/home/kai/CodexDownloads/lunar_navigation/blender/assets/polyhaven_moon_meteor_01_2k")

def require_pbr_assets(asset_dir: Path) -> dict[str, Path]:
    required = {
        "diffuse": asset_dir / "moon_meteor_01_diff_2k.jpg",
        "displacement": asset_dir / "moon_meteor_01_disp_2k.png",
        "normal": asset_dir / "moon_meteor_01_nor_gl_2k.exr",
        "roughness": asset_dir / "moon_meteor_01_rough_2k.exr",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing reviewed PBR assets: " + ", ".join(missing))
    return required
```

Run the script with one required file temporarily renamed and expect it to fail before `clear_collection()` is reached; rename the file back immediately after this test.

- [ ] **Step 2: Implement the polar height field and crater profile**

Define `polar_terrain_height(x: float, y: float) -> float` using a 257×257 grid over 120 metres. Apply the main crater at `(-12.0, 8.0)` with 20 m radius, 6.0 m bowl depth, 1.8 m asymmetric rim, and an eastward ejecta falloff. Add 15 secondary craters with radii from 1.4 to 10.0 m, each with age-dependent rim sharpness. Define `psr_mask(x: float, y: float) -> float` for the central 9 m-radius floor of the main crater.

- [ ] **Step 3: Implement a PBR material with procedural polar correction**

Create `make_polar_regolith_material(asset_paths: dict[str, Path]) -> bpy.types.Material`. Load the four maps as non-packed external images, use object/generated coordinates, and wire diffuse, normal, roughness, and displacement/bump into a Principled BSDF. Multiply diffuse output toward cool gray; set metallic to `0.0` and base roughness to at least `0.86`. Use `psr_mask` or a separate floor material assignment to make the permanent-shadow floor visibly unlit without adding hidden lights.

- [ ] **Step 4: Implement four irregular shared rock mesh prototypes and deterministic scattering**

Create `make_rock_prototypes() -> list[bpy.types.Mesh]` by perturbing low-subdivision icospheres with seeded directional displacement and planar cuts. Create `scatter_rocks(collection, terrain_height, prototypes) -> None` that emits exactly 260 named objects; assign larger rocks near the main rim and eastward ejecta, then smaller rocks across the background. Give each object deterministic rotation, non-uniform scale, and `LunarPolar_Regolith_PBR` material.

- [ ] **Step 5: Implement isolated rebuild and polar illumination**

`main()` must call `require_pbr_assets()` before it calls `clear_collection()`, verify `METRIC` / `METERS`, remove only existing objects in `LunarEnvironment`, build the terrain and rocks, and make a `SUN` at 1.5° elevation. It must not call `bpy.ops.wm.save*`, exporters, subprocesses, network APIs, or file writes.

- [ ] **Step 6: Run static checks**

Run:

```bash
python3 -m py_compile /home/kai/CodexDownloads/lunar_navigation/blender/create_lunar_polar_terrain.py
rg -n "bpy\.ops\.wm\.(save|save_as_mainfile)|export_|urllib|requests|subprocess|open\(" /home/kai/CodexDownloads/lunar_navigation/blender/create_lunar_polar_terrain.py
```

Expected: compilation exits zero and the search returns no matches.

### Task 3: Execute through Blender MCP and verify the generated scene

**Files:**
- Modify at runtime only: objects in the active Blender scene's `LunarEnvironment` collection.
- Preserve: the active `.blend` file is not saved or exported.

**Interfaces:**
- Consumes: the Task 1 asset directory and Task 2 script.
- Produces: a live Blender scene with `LunarPolarTerrain`, one low-angle sun, 260 named rocks, and `LunarPolar_Regolith_PBR`.

- [ ] **Step 1: Execute only the reviewed script through the local MCP bridge**

Use `python.execute` with:

```json
{
  "script_path": "/home/kai/CodexDownloads/lunar_navigation/blender/create_lunar_polar_terrain.py",
  "timeout_seconds": 60
}
```

Expected stdout reports the terrain size, crater count, rock count, PSR crater identifier, and `LunarPolarSun`; `error` is null.

- [ ] **Step 2: Run read-only scene validation**

Query `scene.get_info`, `scene.list_objects`, and `material.list`. Validate this condition:

```python
names = [entry["name"] for entry in objects_result["objects"]]
rocks = sorted(name for name in names if name.startswith("LunarPolarRock_"))
assert "LunarPolarTerrain" in names
assert "LunarPolarSun" in names
assert rocks == [f"LunarPolarRock_{index:03d}" for index in range(260)]
assert "LunarPolar_Regolith_PBR" in material_names
```

Expected: all assertions pass and scene object count is 265 (three preserved default objects plus terrain, sun, and 260 rocks).

- [ ] **Step 3: Verify no save or export occurred**

Re-run the static prohibited-call scan from Task 2 and inspect the Blender MCP execution response. Expected: no save/export function references and no export result returned.

- [ ] **Step 4: Report the asset provenance and scene evidence**

Report the four SHA-256 values from `ASSET_MANIFEST.md`, the exact names/counts from Step 2, and that the `.blend` file was neither saved nor exported.

## Plan Self-Review

- Spec coverage: Task 1 covers CC0 asset acquisition and provenance; Task 2 covers extreme-polar crater geometry, irregular rocks, PBR material, deterministic scale, isolated scene ownership, and low-angle lighting; Task 3 covers execution, PSR scene evidence, and no-save verification.
- Placeholder scan: no unresolved placeholders, deferred work, or undefined function names remain.
- Type consistency: Task 1 produces the four `dict[str, Path]` keys that Task 2 consumes; Task 2 produces the exact object and material names that Task 3 validates.
