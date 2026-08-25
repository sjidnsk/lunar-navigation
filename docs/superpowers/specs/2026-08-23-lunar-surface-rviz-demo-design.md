# Lunar Surface RViz Demo Design

## Purpose

Provide a repeatable, test-only ROS 2 Jazzy demonstration of the isolated
wheel planner. RViz users can inspect a synthetic 1 km by 1 km lunar
surface and place a `2D Goal Pose` to request a route.

## Safety and scope boundary

The demo is self-contained.  It uses `/lunar_demo/*` inputs and the Action
`/lunar_demo/plan_motion`; it does not publish to or subscribe from the
production `/Car/T4/plan_motion` interface.  The production planner's map,
obstacle, slope, and reachability checks remain unchanged.

## Components and data flow

1. A `lunar_surface_demo` publisher generates a deterministic scene using a
   configurable seed (default fixed). Its 1000 by 1000 cells are 1 m square
   (a physical 1 km by 1 km area), with map origin `(-500, -500)`. It publishes:
   - `/lunar_demo/global_overview` (`nav_msgs/OccupancyGrid`),
   - `/lunar_demo/grid_map` (`grid_map_msgs/GridMap`) with `occupancy` and
     `elevation` layers,
   - `/lunar_demo/odometry`, and static identity `map -> odom` TF.
2. The existing pure planner runs with only its endpoint parameters remapped
   to those demo inputs and `/lunar_demo/plan_motion`.
3. The existing RViz goal bridge receives `/lunar_demo/rviz_goal` and sends
   the demo Action request.
4. A test-only visualizer converts successful planner paths to
   `/lunar_demo/path` (`nav_msgs/Path`) and publishes a sampled,
   height-coloured terrain `MarkerArray`.
5. An RViz configuration shows the full occupancy map, wheel-traversability
   layer, terrain markers, robot, start/goal, and planned path. Fixed frame is
   `map`; the initial top-down view spans the complete kilometre-scale map.

## Scene generation

Elevation is the sum of a shallow global gradient, deterministic sinusoidal
ripples, and seeded crater depressions.  Obstacles are seeded rocks and
crater rims.  The publisher reserves a safety disk around the rover start,
then chooses a default goal only after a grid connectivity check verifies a
free route.  This makes the initial demo deterministic and plan-capable;
goals selected on blocked or unreachable cells still exercise normal planner
rejection behavior.

## Launch and interaction

`lunar_surface_rviz_demo.launch.py` starts the scene publisher, pure planner,
goal bridge, path visualizer, and RViz.  The user selects RViz `2D Goal Pose`;
the display publishes it to `/lunar_demo/rviz_goal`.  The demo publishes a
visible default goal so the initial successful request can be reproduced
without manual interaction.

## Acceptance criteria

1. Generator tests verify exactly 1 km by 1 km geometry, finite elevation,
   nonzero bounded obstacle density, safe start, and a connected default goal.
2. A ROS launch test observes all demo inputs and `/lunar_demo/plan_motion`,
   sends the default goal, and requires a successful result with a nonempty
   path/reference.
3. In RViz, setting a reachable 2D goal updates `/lunar_demo/path`; a blocked
   or unreachable goal must not bypass feasibility checks.

## Non-goals

- This is not a dynamics simulator, localization validation, or vehicle
  controller test.
- It does not certify Jazzy deployment on Jetson/Orin or replace the Humble
  production validation boundary.
- It does not alter pure-planner algorithms or production ROS interfaces.
