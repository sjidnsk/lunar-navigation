# Luna commands

Use `./luna init --profile ubuntu22-humble-amd64` (or `jetson-orin-r36`) first.

`luna prepare --dry-run` is read-only. `luna prepare --apply --yes` is the only command that can invoke sudo; it never changes NVIDIA, CUDA, TensorRT or Jetson firmware.

Before a ROS policy adapter is delivered, a non-fallback start is refused with `POLICY_RUNTIME_UNBOUND`.
