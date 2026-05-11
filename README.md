# Drone Autonomy Lab

Simulation-first autonomous drone mission stack.

This project implements core concepts from autonomous robotics in a small,
testable Python codebase. It is intentionally organized as a software stack,
not a single demo script, so each autonomy concern can evolve independently.

```text
Sensors -> State Estimation -> Planning -> Control -> Mission/Safety Logic
```

The goal is not to build a full flight controller, but to demonstrate a modular autonomy architecture with noisy sensor simulation, Kalman-based state estimation, A* path planning, PID waypoint control, mission state handling and safety monitoring.

## Architecture

The package mirrors the autonomy loop:

| Layer | Package | Responsibility |
| --- | --- | --- |
| Sensors | `drone_autonomy_lab.sensors` | Simulate and normalize IMU, GPS, barometer, and future sensor readings. |
| State Estimation | `drone_autonomy_lab.estimation` | Fuse noisy sensor readings into a consistent vehicle state. |
| Planning | `drone_autonomy_lab.planning` | Convert mission goals into waypoints or paths that respect the environment. |
| Control | `drone_autonomy_lab.control` | Convert planned targets into bounded vehicle commands. |
| Mission/Safety Logic | `drone_autonomy_lab.mission`, `drone_autonomy_lab.safety` | Coordinate mission progress, geofence checks, battery limits, and failsafe behavior. |

Configuration lives in `configs/`:

```text
configs/
  environment.yaml  # simulation timing, sensor rates/noise, disturbances
  mission.yaml      # mission waypoints, controller gains, safety limits
```

## Simulation-First Approach

The project treats simulation as the first integration target. That means new
features should be validated against deterministic configs and tests before
they are wired to any real vehicle or hardware SDK. The early workflow is:

1. Define mission and environment assumptions in YAML.
2. Simulate sensor readings and disturbances.
3. Estimate state from noisy observations.
4. Plan and control against the estimated state.
5. Run mission and safety checks continuously.

This keeps the codebase small enough to reason about while still preserving the
interfaces expected in a real autonomous drone stack.

## Relation to Principles of Autonomous Drones

Autonomous drones typically depend on a sense-think-act loop with explicit
safety supervision. This repository maps those principles directly into code:

- Perception is represented by the sensor layer and environment configuration.
- Belief/state is represented by estimation models that smooth noisy inputs.
- Decision-making is represented by planning and mission orchestration.
- Action is represented by control commands with bounded outputs.
- Safety is represented as a first-class layer instead of an afterthought.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The package should also install with:

```bash
pip install -e .
```
