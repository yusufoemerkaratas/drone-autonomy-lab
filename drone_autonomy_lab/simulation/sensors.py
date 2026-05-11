"""Noisy sensor models for the drone simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_autonomy_lab.simulation.state import DroneState


@dataclass(frozen=True)
class SensorMeasurement:
    """A JSON-friendly sensor measurement."""

    sensor_name: str
    timestamp_s: float
    values: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Serialize the measurement for logs."""
        return {
            "sensor_name": self.sensor_name,
            "timestamp_s": round(self.timestamp_s, 6),
            "values": self.values,
        }


class ScheduledSensor:
    """Base class for fixed-rate simulated sensors."""

    def __init__(self, *, sensor_name: str, update_rate_hz: float) -> None:
        if update_rate_hz <= 0:
            raise ValueError("update_rate_hz must be positive")

        self.sensor_name = sensor_name
        self.period_s = 1.0 / update_rate_hz
        self._next_sample_s = self.period_s

    def due(self, timestamp_s: float) -> bool:
        """Return whether the sensor should emit at this timestamp."""
        return timestamp_s + 1e-9 >= self._next_sample_s

    def mark_sampled(self) -> None:
        """Move the schedule to the next sample timestamp."""
        self._next_sample_s += self.period_s


class GPSSensor(ScheduledSensor):
    """Noisy GPS position sensor with probabilistic dropout."""

    def __init__(
        self,
        *,
        update_rate_hz: float,
        position_noise_std_m: float,
        dropout_probability: float,
    ) -> None:
        super().__init__(sensor_name="gps", update_rate_hz=update_rate_hz)
        self.position_noise_std_m = position_noise_std_m
        self.dropout_probability = dropout_probability

    def measure(
        self,
        *,
        timestamp_s: float,
        state: DroneState,
        rng: np.random.Generator,
    ) -> SensorMeasurement | None:
        if not self.due(timestamp_s):
            return None

        self.mark_sampled()
        if rng.random() < self.dropout_probability:
            return None

        noise = rng.normal(0.0, self.position_noise_std_m, size=3)
        return SensorMeasurement(
            sensor_name=self.sensor_name,
            timestamp_s=timestamp_s,
            values={
                "x_m": state.x_m + float(noise[0]),
                "y_m": state.y_m + float(noise[1]),
                "z_m": state.z_m + float(noise[2]),
            },
        )


class BarometerSensor(ScheduledSensor):
    """Noisy altitude sensor with linear drift."""

    def __init__(
        self,
        *,
        update_rate_hz: float,
        altitude_noise_std_m: float,
        drift_mps: float,
    ) -> None:
        super().__init__(sensor_name="barometer", update_rate_hz=update_rate_hz)
        self.altitude_noise_std_m = altitude_noise_std_m
        self.drift_mps = drift_mps

    def measure(
        self,
        *,
        timestamp_s: float,
        state: DroneState,
        rng: np.random.Generator,
    ) -> SensorMeasurement | None:
        if not self.due(timestamp_s):
            return None

        self.mark_sampled()
        noise = float(rng.normal(0.0, self.altitude_noise_std_m))
        return SensorMeasurement(
            sensor_name=self.sensor_name,
            timestamp_s=timestamp_s,
            values={"altitude_m": state.z_m + self.drift_mps * timestamp_s + noise},
        )


class IMUSensor(ScheduledSensor):
    """Noisy IMU with acceleration and gyro bias."""

    def __init__(
        self,
        *,
        update_rate_hz: float,
        accel_noise_std: float,
        gyro_noise_std: float,
        accel_bias_mps2: tuple[float, float, float],
        gyro_bias_rad_s: float,
    ) -> None:
        super().__init__(sensor_name="imu", update_rate_hz=update_rate_hz)
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_mps2 = accel_bias_mps2
        self.gyro_bias_rad_s = gyro_bias_rad_s
        self._previous_state: DroneState | None = None

    def measure(
        self,
        *,
        timestamp_s: float,
        state: DroneState,
        dt_s: float,
        rng: np.random.Generator,
    ) -> SensorMeasurement | None:
        if not self.due(timestamp_s):
            self._previous_state = state
            return None

        previous_state = self._previous_state or state
        self._previous_state = state
        self.mark_sampled()

        acceleration = (
            (state.vx_mps - previous_state.vx_mps) / dt_s,
            (state.vy_mps - previous_state.vy_mps) / dt_s,
            (state.vz_mps - previous_state.vz_mps) / dt_s,
        )
        accel_noise = rng.normal(0.0, self.accel_noise_std, size=3)
        gyro_noise = float(rng.normal(0.0, self.gyro_noise_std))

        return SensorMeasurement(
            sensor_name=self.sensor_name,
            timestamp_s=timestamp_s,
            values={
                "ax_mps2": acceleration[0] + self.accel_bias_mps2[0] + float(accel_noise[0]),
                "ay_mps2": acceleration[1] + self.accel_bias_mps2[1] + float(accel_noise[1]),
                "az_mps2": acceleration[2] + self.accel_bias_mps2[2] + float(accel_noise[2]),
                "yaw_rate_rad_s": state.yaw_rate_rad_s + self.gyro_bias_rad_s + gyro_noise,
            },
        )


class SensorSuite:
    """Collection of simulator sensors sampled from the same RNG stream."""

    def __init__(
        self,
        *,
        gps: GPSSensor,
        barometer: BarometerSensor,
        imu: IMUSensor,
    ) -> None:
        self.gps = gps
        self.barometer = barometer
        self.imu = imu

    def measure(
        self,
        *,
        timestamp_s: float,
        state: DroneState,
        dt_s: float,
        rng: np.random.Generator,
    ) -> dict[str, dict[str, object] | None]:
        """Measure all sensors, using None when a sensor is silent or dropped."""
        measurements = {
            "gps": self.gps.measure(timestamp_s=timestamp_s, state=state, rng=rng),
            "barometer": self.barometer.measure(timestamp_s=timestamp_s, state=state, rng=rng),
            "imu": self.imu.measure(timestamp_s=timestamp_s, state=state, dt_s=dt_s, rng=rng),
        }
        return {
            sensor_name: measurement.to_dict() if measurement is not None else None
            for sensor_name, measurement in measurements.items()
        }
