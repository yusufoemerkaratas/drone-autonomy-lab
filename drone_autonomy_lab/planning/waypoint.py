"""Mission waypoint representation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Waypoint:
    """A target point for the planner/controller handoff."""

    x_m: float
    y_m: float
    z_m: float
    hold_s: float = 0.0
