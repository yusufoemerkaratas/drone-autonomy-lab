"""Kalman-based state estimation for local drone motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from drone_autonomy_lab.estimation.state import VehicleState


ArrayLike = list[float] | tuple[float, ...] | np.ndarray


@dataclass(frozen=True)
class EstimatorSnapshot:
    """JSON-friendly estimator state at one timestamp."""

    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    covariance_trace: float

    def to_dict(self) -> dict[str, object]:
        return {
            "position_m": self.position_m,
            "velocity_mps": self.velocity_mps,
            "covariance_trace": self.covariance_trace,
        }


class KalmanFilter:
    """Linear Kalman filter with state [px, py, pz, vx, vy, vz]."""

    def __init__(
        self,
        *,
        initial_state: ArrayLike | None = None,
        initial_covariance: ArrayLike | None = None,
        process_noise: float | ArrayLike = 0.05,
        measurement_noise: Mapping[str, float | ArrayLike] | None = None,
    ) -> None:
        state = np.zeros(6, dtype=float) if initial_state is None else np.asarray(initial_state, dtype=float)
        if state.shape != (6,):
            raise ValueError("initial_state must contain 6 values: px, py, pz, vx, vy, vz")

        self.x = state.reshape(6, 1)
        self.P = self._matrix(initial_covariance, size=6, default=1.0)
        self.Q = self._matrix(process_noise, size=6, default=0.05)
        noise = measurement_noise or {}
        self.R: dict[str, np.ndarray] = {
            "gps": self._matrix(noise.get("gps", 0.16), size=2, default=0.16),
            "barometer": self._matrix(noise.get("barometer", 0.01), size=1, default=0.01),
        }

    def predict(self, dt: float, control_input: ArrayLike | None = None) -> np.ndarray:
        """Advance the estimate using IMU acceleration as control input."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        acceleration = (
            np.zeros((3, 1), dtype=float)
            if control_input is None
            else np.asarray(control_input, dtype=float).reshape(3, 1)
        )

        f = np.eye(6, dtype=float)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt

        b = np.zeros((6, 3), dtype=float)
        b[0, 0] = 0.5 * dt * dt
        b[1, 1] = 0.5 * dt * dt
        b[2, 2] = 0.5 * dt * dt
        b[3, 0] = dt
        b[4, 1] = dt
        b[5, 2] = dt

        self.x = f @ self.x + b @ acceleration
        self.P = f @ self.P @ f.T + self.Q
        return self.state_vector.copy()

    def update(self, measurement: Mapping[str, float] | ArrayLike, sensor_type: str) -> np.ndarray:
        """Correct the estimate with a supported sensor measurement."""
        if sensor_type == "gps":
            h = np.zeros((2, 6), dtype=float)
            h[0, 0] = 1.0
            h[1, 1] = 1.0
            z = self._gps_measurement(measurement)
        elif sensor_type == "barometer":
            h = np.zeros((1, 6), dtype=float)
            h[0, 2] = 1.0
            z = self._barometer_measurement(measurement)
        else:
            raise ValueError(f"Unsupported sensor_type: {sensor_type}")

        r = self.R[sensor_type]
        innovation = z - h @ self.x
        innovation_covariance = h @ self.P @ h.T + r
        gain = self.P @ h.T @ np.linalg.inv(innovation_covariance)
        identity = np.eye(6, dtype=float)

        self.x = self.x + gain @ innovation
        self.P = (identity - gain @ h) @ self.P
        return self.state_vector.copy()

    @property
    def state_vector(self) -> np.ndarray:
        return self.x.reshape(6)

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    def vehicle_state(self, *, yaw_rad: float = 0.0) -> VehicleState:
        state = self.state_vector
        return VehicleState(
            position_m=(float(state[0]), float(state[1]), float(state[2])),
            velocity_mps=(float(state[3]), float(state[4]), float(state[5])),
            yaw_rad=yaw_rad,
        )

    def snapshot(self) -> EstimatorSnapshot:
        state = self.state_vector
        return EstimatorSnapshot(
            position_m=(float(state[0]), float(state[1]), float(state[2])),
            velocity_mps=(float(state[3]), float(state[4]), float(state[5])),
            covariance_trace=float(np.trace(self.P)),
        )

    @staticmethod
    def _matrix(value: float | ArrayLike | None, *, size: int, default: float) -> np.ndarray:
        if value is None:
            return np.eye(size, dtype=float) * default
        array = np.asarray(value, dtype=float)
        if array.shape == ():
            return np.eye(size, dtype=float) * float(array)
        if array.shape == (size,):
            return np.diag(array)
        if array.shape == (size, size):
            return array
        raise ValueError(f"Expected scalar, length-{size} vector, or {size}x{size} matrix")

    @staticmethod
    def _gps_measurement(measurement: Mapping[str, float] | ArrayLike) -> np.ndarray:
        if isinstance(measurement, Mapping):
            return np.array([[float(measurement["x_m"])], [float(measurement["y_m"])]])
        return np.asarray(measurement, dtype=float).reshape(2, 1)

    @staticmethod
    def _barometer_measurement(measurement: Mapping[str, float] | ArrayLike) -> np.ndarray:
        if isinstance(measurement, Mapping):
            key = "altitude_m" if "altitude_m" in measurement else "z_m"
            return np.array([[float(measurement[key])]])
        return np.asarray(measurement, dtype=float).reshape(1, 1)
