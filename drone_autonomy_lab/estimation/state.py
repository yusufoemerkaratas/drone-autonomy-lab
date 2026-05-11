"""Vehicle state representation used across planning and control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleState:
    """Estimated drone state in a local simulation frame."""

    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    yaw_rad: float
