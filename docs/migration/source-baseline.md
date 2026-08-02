# Legacy migration source baseline

This baseline freezes the source commits, relative paths, and SHA-256 values
that authorize later migration copies. It does not authorize importing a
legacy workspace into the new runtime.

## Frozen repositories

| Repository key | Commit | Origin |
| --- | --- | --- |
| `legacy_root` | `7309e93fdb85c60ff3736efe1a7f3c7eb640ee78` | `git@github.com:sjidnsk/lunar-path-planning.git` |
| `path_planner` | `2f6378d3c47da027c0d4146d94cab881b8f2a594` | `git@github.com:sjidnsk/path-planner.git` |
| `dev_platform_constraints` | `61e9fa8afd09db83632456bdcf181c222ee13513` | `https://github.com/sjidnsk/dev-platform-constraints.git` |

The historical bundle set is stored under the backup storage relative location
`lunar-path-planning-backups/2026-08-01-6a4c2dd`. It contains the two listed
gitlink commits, but its legacy-root bundle predates the frozen
`7309e93fdb85c60ff3736efe1a7f3c7eb640ee78` commit. Recovering that root commit
therefore also requires the recorded origin to retain the object, or a newer
verified bundle before the legacy checkout is retired.

## Scope

`migration/source_inventory.yaml` selects only PPO observation, environment,
collector, rollout, trainer, checkpoint, resume, and evaluation core material;
v3 platform capability inputs; frozen C++ planner v3 headers, sources, CMake
inputs, and selected contract/integration tests; and platform-constraint
capability material. `migration/fixture_inventory.yaml` selects only three
small planner test fixtures. The legacy Python A* entries have the explicit
`tests/differential_reference` role: they are migration comparison inputs only,
not platform capability inputs or production runtime source.

The selection deliberately excludes legacy governance, workflows, runners,
artifact utilities, checkpoints as artifacts, job state, generated objects,
Python bindings, benchmarks, and other runtime artifacts.

## Legacy transition policy

The frozen commits above do not change. Until the AGX acceptance observation
period ends, the `legacy-maintenance` branch may receive only urgent defect or
security fixes; all new functionality belongs in this repository. Every such
legacy fix must explicitly record whether it is also migrated here. After the
AGX observation period ends, the legacy repositories become read-only archives.

## Rebuild and verify

From the new repository root, substitute a local legacy checkout for
`<legacy-root>` and run:

```powershell
python tools/create_source_inventory.py `
  --legacy-root "<legacy-root>" `
  --output migration/source_inventory.yaml `
  --fixture-output migration/fixture_inventory.yaml
git diff --exit-code -- migration/source_inventory.yaml migration/fixture_inventory.yaml
```

The generator rejects any selected dirty or assume-unchanged file, requires a
regular blob at the selected path in `HEAD`, and hashes that `HEAD` blob rather
than checkout-filtered working-tree bytes. The final `git diff` is the fixture
SHA-256 verification command as well as the source-inventory verification
command.

## Runtime boundary

New runtime code must not import from, execute from, or add either the legacy
root or a legacy gitlink to its module search path. Migration is a deliberate
copy-and-adapt operation from this frozen baseline, never a runtime dependency.
