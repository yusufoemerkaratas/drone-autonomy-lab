"""Mission profile data structures."""

from __future__ import annotations

from dataclasses import dataclass

from drone_autonomy_lab.planning import Waypoint


@dataclass(frozen=True)
class MissionProfile:
    """Named mission with ordered waypoints and safety constraints."""

    name: str
    waypoints: tuple[Waypoint, ...]
    max_duration_s: float
