"""Fixed-step drone simulator for ground truth and measurements."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_autonomy_lab.simulation.perception import PerceptionMock
from drone_autonomy_lab.simulation.sensors import SensorSuite
from drone_autonomy_lab.simulation.state import DEFAULT_INITIAL_STATE, DroneState


@dataclass(frozen=True)
class SimulationFrame:
    """One fixed-dt simulation output frame."""

    timestamp_s: float
    true_state: DroneState
    measurements: dict[str, dict[str, float] | None] = field(default_factory=dict)
    detections: list[dict[str, object]] = field(default_factory=list)

    def to_log_record(self) -> dict[str, object]:
        """Return a JSON-serializable record for demo logs."""
        return {
            "timestamp_s": round(self.timestamp_s, 6),
            "true_state": self.true_state.to_dict(),
            "sensor_measurements": self.measurements,
            "detections": self.detections,
        }


class DroneSimulator:
    """Advance a lightweight drone state with deterministic fixed-dt dynamics."""

    def __init__(
        self,
        *,
        dt_s: float,
        seed: int = 7,
        initial_state: DroneState = DEFAULT_INITIAL_STATE,
        wind_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gust_std_mps: float = 0.0,
        sensor_suite: SensorSuite | None = None,
        perception: PerceptionMock | None = None,
    ) -> None:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")

        self.dt_s = dt_s
        self.timestamp_s = 0.0
        self.state = initial_state
        self.wind_mps = wind_mps
        self.gust_std_mps = gust_std_mps
        self.rng = np.random.default_rng(seed)
        self.sensor_suite = sensor_suite
        self.perception = perception

    def step(self) -> SimulationFrame:
        """Advance the true state by exactly one fixed timestep."""
        self.timestamp_s = round(self.timestamp_s + self.dt_s, 10)
        self.state = self._advance_state(self.state, self.timestamp_s)
        measurements = {}
        detections: list[dict[str, object]] = []

        if self.sensor_suite is not None:
            measurements = self.sensor_suite.measure(
                timestamp_s=self.timestamp_s,
                state=self.state,
                dt_s=self.dt_s,
                rng=self.rng,
            )
        if self.perception is not None:
            detections = self.perception.detect(
                timestamp_s=self.timestamp_s,
                state=self.state,
                rng=self.rng,
            )

        return SimulationFrame(
            timestamp_s=self.timestamp_s,
            true_state=self.state,
            measurements=measurements,
            detections=detections,
        )

    def run(self, duration_s: float) -> list[SimulationFrame]:
        """Run the simulator for a fixed duration and return all frames."""
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")

        steps = int(round(duration_s / self.dt_s))
        return [self.step() for _ in range(steps)]

    def _advance_state(self, state: DroneState, timestamp_s: float) -> DroneState:
        gust = self.rng.normal(0.0, self.gust_std_mps, size=3)
        wind_x, wind_y, wind_z = (
            self.wind_mps[0] + float(gust[0]),
            self.wind_mps[1] + float(gust[1]),
            self.wind_mps[2] + float(gust[2]),
        )

        ax_mps2 = 0.08 * np.sin(0.4 * timestamp_s)
        ay_mps2 = 0.05 * np.cos(0.25 * timestamp_s)
        az_mps2 = 0.02 * np.sin(0.3 * timestamp_s)

        vx_mps = state.vx_mps + float(ax_mps2) * self.dt_s
        vy_mps = state.vy_mps + float(ay_mps2) * self.dt_s
        vz_mps = state.vz_mps + float(az_mps2) * self.dt_s

        return DroneState(
            x_m=state.x_m + (vx_mps + wind_x) * self.dt_s,
            y_m=state.y_m + (vy_mps + wind_y) * self.dt_s,
            z_m=max(0.0, state.z_m + (vz_mps + wind_z) * self.dt_s),
            vx_mps=vx_mps,
            vy_mps=vy_mps,
            vz_mps=vz_mps,
            yaw_rad=state.yaw_rad + state.yaw_rate_rad_s * self.dt_s,
            yaw_rate_rad_s=state.yaw_rate_rad_s,
        )
