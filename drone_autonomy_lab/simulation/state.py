"""Ground-truth drone state for the simulation layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DroneState:
    """Ground-truth state in a local frame.

    The state intentionally stays separate from noisy sensor measurements and
    future estimated state, which keeps the simulation honest for estimator
    tests.
    """

    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    yaw_rad: float
    yaw_rate_rad_s: float

    def to_dict(self) -> dict[str, float]:
        """Serialize the state for logs and tests."""
        return asdict(self)


DEFAULT_INITIAL_STATE = DroneState(
    x_m=0.0,
    y_m=0.0,
    z_m=0.0,
    vx_mps=0.6,
    vy_mps=0.15,
    vz_mps=0.08,
    yaw_rad=0.0,
    yaw_rate_rad_s=0.03,
)
