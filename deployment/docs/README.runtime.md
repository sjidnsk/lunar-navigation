# Lunar Navigation Runtime

`luna` installs and starts the current safe planner on Ubuntu 22.04 + ROS 2 Humble amd64 or Jetson AGX Orin R36.

The initial runtime uses `policy.mode: fallback`. A policy artifact can be staged and verified, but no model is allowed to drive a ROS planner until a separately delivered policy adapter exists.

The runtime consumes the v5 global/local maps, odometry, localization state, TF, task and motion feedback; it serves `/plan_motion` and publishes diagnostics and route markers. Safety projection, footprint, clearance, endpoint and route certification remain mandatory.
