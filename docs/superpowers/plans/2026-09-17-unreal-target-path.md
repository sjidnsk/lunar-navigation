# Unreal target and local path implementation plan

Approved scope: extend a P4-owned copy of lunar_car_ctrl; original Env_X/P3 files untouched. One vehicle TCP connection. No requested goal yaw.

- [x] Snapshot original interface and active P4 adapter; record hashes and rollback runtime environment.
- [x] RED: protocol payload, nonfinite inputs, coordinate roundtrip, request/session/revision tests.
- [x] Add command 7/8 dispatch in copied interface, bounded payloads and complete serialized writes. Retain vehicle command/telemetry format.
- [x] Extend existing P4 feedback process with async goal dispatch, exploration pause, current-session PathReference return and inverse coordinate transform. Use same alignment and explicit map-to-odom input. No second TCP connection or new running node.
- [x] Verify standalone helpers and Humble native interface build; run isolated TCP/ROS loopback including fragmented requests, response and telemetry multiplexing, latest-goal filtering.
- [x] Stop P4, snapshot and replace only old vehicle process, verify sole connection and fresh feedback. Start P4 navigation ready for UE clicks; no synthetic live driving goal.
- [x] Document commands, response semantics, deployment/rollback and unverified UE renderer framing limitations. No commit or push requested.
