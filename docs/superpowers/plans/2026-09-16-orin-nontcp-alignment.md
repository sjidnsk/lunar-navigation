# Orin Non-TCP Alignment Implementation Plan

> Execute inline with superpowers:executing-plans, retaining per-task red/green evidence. No parallel agent work is required.

**Goal:** Integrate the 104 static TF adapter and 2 m start margin, then align transport-neutral P4 behavior/configuration with the audited 216 deployment.

**Architecture:** P3 owns localization/elevation; an Orin adapter relays static map-to-odom without restamping. Navigation owns certified planning and derived maps; exploration owns task progression; the single controller owns vehicle commands. TCP, UE, fabricated feedback and simulation calibration remain excluded.

**Tech Stack:** C++20, Python, ROS 2 Humble/Orin deployment; separate Jazzy development tests.

**Spec:** User-approved design in the conversation: all non-TCP differences, including completion, error-state and longitudinal feedback semantics; retain real 104 input interfaces. Explicitly include static TF adapter and start_blind_zone_margin_m=2.0.

## Global Constraints

- Develop from integration in feat/orin-nontcp-alignment; no automatic merge, push or branch-pointer changes.
- Preserve current 104 configuration and 216 source snapshots before edits.
- No controller startup or vehicle commands during deployment verification.
- Do not clear BLOCKED cells, relax physical limits or report arrival from planning alone.
- Runtime logs, snapshots and build artifacts remain outside Git.
- A 2.0 m start margin adds to the footprint circumscribed radius; it is not a 2.0 m total radius.

## Task 1 — Baseline and audit

- [x] Archive 104 source/config/scripts/docs and active margin override; archive 216 source/config.
- [x] Record archive location: /tmp/p4-alignment-audit-20260916-2126.
- [x] Run `python3 -m pytest tests/test_incremental_navigation_interface_contract.py tests/test_incremental_navigation_message_contract.py -q`: original baseline 13 passed, one stale single-publisher assertion failed.
- [x] Remove obsolete source-count and fixed-QoS spelling assumptions; retain endpoint/value/default checks and native publisher behavior tests.
- [x] Re-run the same baseline (14 passed) and `git diff --check`.

## Task 2 — Formal static TF adapter and hardware profile

Files: scripts/orin/static_tf_input.py, scripts/orin/start_static_tf_input.sh, tests/test_static_tf_input.py, config/orin_hardware.yaml, config/wheel_orin.yaml, docs/STATIC_TF_INPUT.md.
Consumes /tf_static (Reliable/TransientLocal). Produces only /P4/input/map_to_odom at 5 Hz with original transform timestamp; never synthesize identity.

- [x] Import the archived DDS regression test first; run in isolated domain 181 and confirm missing implementation fails.
- [x] Integrate the audited adapter; run its three real-DDS cases for no invented identity, late-join repeat without restamping, replacement and absence of shared outputs.
- [x] Add hardware-only profile rather than changing demo/global defaults; set margin 2.0 and explicit private TF input.
- [x] Test configuration loading and relative platform path resolution; inspect navigation-only launch; document startup and rollback. Live parameter checks remain in Task 5.

## Task 3 — Navigation alignment

Files: lunar_incremental_navigation_core map/local/global/session source and tests; lunar_incremental_navigation_ros adapters, publisher, parameter loading, launch and tests.

- [x] Compare current integration, orin-humble, archived 104 and 216 by file contents; enumerate missing functional changes, excluding package stripping and TCP.
- [x] Existing integration already covers frame checks, local goals, route cache and certification. Add red/green native regressions for remaining goal-tolerance and fine/debug publication differences.
- [x] Preserve existing passing integration capabilities; do not overwrite later generic fixes with a release export.
- [x] Build affected packages and run local Jazzy native tests serially, including the 750 m performance case.

## Task 4 — Exploration and feedback alignment

Files: lunar_pure_exploration_ros incremental node/tests, lunar_pure_exploration_core candidate/state helpers/tests, lunar_pure_wheeled_controller node/execution/tests, lunar_planning_msgs/msg/TrackingStatus.msg.

- [x] Add coverage-target continuation regression; run existing empty-candidate, invalid-input and wait/cancel/pause package tests.
- [x] Preserve existing signed longitudinal feedback contract; add lateral-only stop regression and document that longitudinal speed alone does not prove planar rest.
- [x] Keep execution logic and feedback consumers contract-consistent; import no unused second orchestration stack.
- [x] Set hardware capability speed to 0.2 m/s and goal tolerances to 0.1 m / 0.05 rad; retain physical geometry/slope/relief limits.
- [x] Run exploration/controller regression suites and interface builds (Jazzy: 999 package test results, plus focused root tests).

## Task 5 — Release verification

- [x] Update README, operator instructions and difference matrix with configuration scope and known limitations.
- [ ] Full relevant local tests; independent Humble/Orin build/test; retain failures and NOT_RUN boundaries.
- Local regression is green; 104 nine-package native build passed, 20 focused and 153 controller tests passed. Native C++ failures remain recorded in ORIN_NON_TCP_ALIGNMENT.md; release acceptance is not complete.
- [x] Export to a new P4 directory, verify source archive hashes and preserve previous install.
- [x] Restart only navigation/TF after explicit user request; confirm no stale action, no controller and zero command publishers on domains 19/59. Runtime parameters verified on candidate.
- [ ] Test PLAN_FOUND plus revision-consistent ACTIVE reference, then cancel. START_BLOCKED remains a failure, not a reason to weaken certification.
- Deferred: P3 stopped; user requested candidate switch without waiting for joint testing. No goal sent.
- [ ] Hand off exact source SHA or dirty-tree manifest, startup commands, test evidence and rollback. Do not merge/push without explicit instruction.
