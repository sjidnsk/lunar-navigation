# Legacy migration source baseline

This baseline is a read-only source declaration. It authorizes later volumes to
copy only files whose repository key, commit, relative path, and SHA-256 match
the checked-in inventories. It does not authorize importing a legacy workspace
into the new runtime.

## Frozen repositories

| Repository key | Commit | Origin |
| --- | --- | --- |
| `legacy_root` | `7309e93fdb85c60ff3736efe1a7f3c7eb640ee78` | `git@github.com:sjidnsk/lunar-path-planning.git` |
| `path_planner` | `2f6378d3c47da027c0d4146d94cab881b8f2a594` | `git@github.com:sjidnsk/path-planner.git` |
| `dev_platform_constraints` | `61e9fa8afd09db83632456bdcf181c222ee13513` | `https://github.com/sjidnsk/dev-platform-constraints.git` |

The legacy root and both listed gitlinks are read-only migration sources.
The historical bundle set is stored under the backup storage relative location
`lunar-path-planning-backups/2026-08-01-6a4c2dd`. It contains the two listed
gitlink commits, but its legacy-root bundle predates the frozen
`7309e93fdb85c60ff3736efe1a7f3c7eb640ee78` commit. Recovering that root commit
therefore also requires the recorded origin to retain the object, or a newer
verified bundle before the legacy checkout is retired.

## Scope

`migration/source_inventory.yaml` selects only PPO core material, v3 platform
capability inputs, C++ planner v3 material, and platform-constraint capability
material. `migration/fixture_inventory.yaml` selects only three small planner
test fixtures. It deliberately excludes historical runners, checkpoints, job
state, generated objects, and other runtime artifacts.

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

The generator rejects any selected dirty file before hashing its original
bytes. The final `git diff` is the fixture SHA-256 verification command as
well as the source-inventory verification command.

## Runtime boundary

New runtime code must not import from, execute from, or add either the legacy
root or a legacy gitlink to its module search path. Migration is a deliberate
copy-and-adapt operation from this frozen baseline, never a runtime dependency.
