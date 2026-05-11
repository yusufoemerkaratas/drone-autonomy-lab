"""Control command data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlCommand:
    """Normalized command emitted by the controller."""

    thrust: float
    roll_rad: float
    pitch_rad: float
    yaw_rate_rad_s: float
