"""Safety limits used by mission and control logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyLimits:
    """Operational envelope for simulation-first validation."""

    max_altitude_m: float
    max_speed_mps: float
    geofence_radius_m: float
    low_battery_percent: float
