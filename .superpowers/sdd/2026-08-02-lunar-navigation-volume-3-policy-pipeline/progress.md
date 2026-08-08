# SDD ledger — plan: docs/superpowers/plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md

Branch: `volume-3-policy-pipeline`
Branch base: `430961a88f5340bd7bec5163a13bb2b92cdd792c`
Baseline: `repository boundaries: OK`; `tests/foundation: 113 passed`
Task 1: review 1 — spec incomplete; actual frozen PPO core import remains open; importer dependency boundary needs fixes
Task 1: minor (deferred): add HOPPER to shared-policy forward parametrization
Task 1: minor (deferred): CUDA mask assertion should reuse the batch passed to the policy
Task 1: fix round 1/5 (3 addressed, 4 open — dynamic import bypass; duplicate six-input network; forbidden Stage/durable artifact graph; type-only replacement causes runtime NameError; commits 34e6ab5..dd49163)
Task 1: fix round 2/5 (3 addressed, 4 open — function-local dynamic import alias; omit_imports bypasses forbidden dependencies; checkpoint failure partially mutates train state; frozen-source test hardcodes host path; commits dd49163..ff08644)
Task 1: fix round 3/5 (3 addressed, 1 open — ContentRef module/import-name omission bypass; commits ff08644..8505ede)
Task 1: fix round 4/5 (1 addressed, 0 open; commits 8505ede..e896aa0)
Task 1: complete (commits 430961a..e896aa0, review clean)
Task 2: review 1 — 1 Critical, 2 Important open (authoritative hopper landing feedback; output consistency matrix; production env-to-C++ bridge composition)
Task 2: minor (deferred): add a behavioral test proving GIL release permits concurrent Python progress
Task 2: minor (deferred): validate finite positive normalization scales and platform string/one-hot alignment
Task 2: fix round 1/5 (3 addressed, 0 open; commits 12587d1..a0cc8df)
Task 2: cross-task review note resolved — later single-network/artifact/deployment/DDS/durable gates remain assigned to their planned tasks; Task 2 added no conflicting path
Task 2: complete (commits e896aa0..a0cc8df, review clean)
Task 3: review 1 — scoped to 2 Critical, 1 Important open (real C++ v3 on-policy hot path; real calibration/shared 24h budget; complete resume state/cadence)
Task 3: deferred by user speed preference — external venv hard gate; candidate TOCTOU; deep OMP/MKL runtime proof; CLI split and extra test polish
Task 3: fix round 1/5 submitted for scoped re-review (3 addressed; commits 09c256a..799c838)
Task 3: scoped re-review 1 — 2 addressed, 1 Critical open (final bounded GPU unit may exceed the 86400-second physical cap)
Task 3: fix round 2/5 submitted for scoped re-review (final bounded-unit reserve and terminal budget persistence; commits 799c838..e8ce6b8)
Task 3: scoped re-review 2 — final finding addressed; no direct Critical/Important regression
Task 3: complete (commits a0cc8df..e8ce6b8, review clean; independent 298 passed/1 skipped, boundary OK)
Task 4: review 1 — 3 Important open (proxy reference execution; full frozen scenario evaluation; measured safety/commitment metrics)
Task 4: fix round 1/5 (3 addressed, 1 new Important open — evaluation swallows InvalidTransition; commits 07e88c8..dc50e93)
Task 4: fix round 2/5 (1 addressed, 0 open — invalid evaluation transitions now fail closed; commits dc50e93..7318ff3)
Task 4: complete (commits e8ce6b8..7318ff3, review clean; independent 343 passed/1 skipped, boundary OK)
Task 5: complete (commits 2d36778..c56a16e; C++ v3 traversability/plan, 64 yaw bins, decision identity and no-action boundaries)
Task 6: complete (commits b1cca75..ddd7b48; success-led Reward V2, frozen PPO, 30-minute latest checkpoint and explicit budget extension)
Task 7: complete through approved parity fix (commits 536e371..b5bd229; formal capability closure and checkpoint v3 identity fail closed)
Task 8: complete (commits 4734ef9..a6c4648; official NASA/JAXA source lock and seed-4080 split recorded outside Git data roots)
Task 9: pretraining-ready / blocked-on-capability; implementation, bounded smoke and official data handoff verified
Task 9 evidence: watcher 13 passed; CPU bounded source/split/hazard/V2/C++ projection+plan/update/checkpoint/resume smoke passed; formal missing/test-only refusal covered for train/resume/evaluate
Task 9 regression: focused 191 passed/1 deselected; non-CUDA 566 passed/1 skipped/3 deselected; RTX CUDA 2 passed/12 deselected; ROS core/bridge 80 tests with zero failures; boundaries OK + foundation 13 passed
Task 9 hard stop: no formal seed 4080 rollout, no formal 24-hour GPU use, no ONNX/TensorRT/model candidate, and no AGX tag
