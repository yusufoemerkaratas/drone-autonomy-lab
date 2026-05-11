# Drone Autonomy Lab

Simulation-first autonomous drone mission stack.

This project implements core concepts from autonomous robotics in a small, testable Python codebase:

Sensors → State Estimation → Planning → Control → Mission/Safety Logic

The goal is not to build a full flight controller, but to demonstrate a modular autonomy architecture with noisy sensor simulation, Kalman-based state estimation, A* path planning, PID waypoint control, mission state handling and safety monitoring.
