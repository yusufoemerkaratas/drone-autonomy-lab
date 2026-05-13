"""Run the reproducible simulator demo and write a JSONL log."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_autonomy_lab.config import PROJECT_ROOT, load_environment_config
from drone_autonomy_lab.estimation import KalmanFilter
from drone_autonomy_lab.simulation.perception import DetectionObject, PerceptionMock
from drone_autonomy_lab.simulation.sensors import BarometerSensor, GPSSensor, IMUSensor, SensorSuite
from drone_autonomy_lab.simulation.simulator import DroneSimulator, SimulationFrame


@dataclass(frozen=True)
class EstimationDemoResult:
    """Trajectory comparison produced by the Kalman estimator demo."""

    frames: list[dict[str, object]] = field(default_factory=list)
    rmse: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"frames": self.frames, "rmse": self.rmse}


def build_simulator_from_config(
    config: dict[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> DroneSimulator:
    """Build a simulator from `configs/environment.yaml`-style data."""
    environment_config = config or load_environment_config()
    environment = environment_config["environment"]
    sensors = environment_config["sensors"]
    wind = environment_config["disturbances"]["wind"]
    perception = environment_config["perception"]
    demo = environment_config.get("demo", {})

    sensor_suite = SensorSuite(
        gps=GPSSensor(
            update_rate_hz=sensors["gps"]["update_rate_hz"],
            position_noise_std_m=sensors["gps"]["position_noise_std_m"],
            dropout_probability=sensors["gps"]["dropout_probability"],
        ),
        barometer=BarometerSensor(
            update_rate_hz=sensors["barometer"]["update_rate_hz"],
            altitude_noise_std_m=sensors["barometer"]["altitude_noise_std_m"],
            drift_mps=sensors["barometer"]["drift_mps"],
        ),
        imu=IMUSensor(
            update_rate_hz=sensors["imu"]["update_rate_hz"],
            accel_noise_std=sensors["imu"]["accel_noise_std"],
            gyro_noise_std=sensors["imu"]["gyro_noise_std"],
            accel_bias_mps2=tuple(sensors["imu"]["accel_bias_mps2"]),
            gyro_bias_rad_s=sensors["imu"]["gyro_bias_rad_s"],
        ),
    )
    perception_mock = PerceptionMock(
        update_rate_hz=perception["update_rate_hz"],
        detection_range_m=perception["detection_range_m"],
        position_noise_std_m=perception["position_noise_std_m"],
        objects=tuple(
            DetectionObject(
                object_id=entry["id"],
                object_type=entry["type"],
                position_m=tuple(entry["position_m"]),
            )
            for entry in perception["objects"]
        ),
    )

    return DroneSimulator(
        dt_s=environment["time_step_s"],
        seed=seed if seed is not None else demo.get("seed", 7),
        wind_mps=tuple(wind["mean_mps"]),
        gust_std_mps=wind["gust_std_mps"],
        sensor_suite=sensor_suite,
        perception=perception_mock,
    )


def write_jsonl_log(frames: list[SimulationFrame], output_path: str | Path) -> Path:
    """Write simulation frames as one JSON object per line."""
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log_file:
        for frame in frames:
            log_file.write(json.dumps(frame.to_log_record(), sort_keys=True))
            log_file.write("\n")

    return path


def write_estimation_jsonl_log(result: EstimationDemoResult, output_path: str | Path) -> Path:
    """Write estimator demo frames and a final metrics row as JSONL."""
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log_file:
        for frame in result.frames:
            log_file.write(json.dumps(frame, sort_keys=True))
            log_file.write("\n")
        log_file.write(json.dumps({"type": "rmse", "rmse": result.rmse}, sort_keys=True))
        log_file.write("\n")

    return path


def run_estimation_demo(
    *,
    duration_s: float | None = None,
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> EstimationDemoResult:
    """Compare ground truth, noisy GPS, and Kalman estimated trajectory."""
    config = load_environment_config()
    sensors = config["sensors"]
    demo_config = config.get("demo", {})
    simulator = build_simulator_from_config(config, seed=seed)
    estimator = KalmanFilter(
        process_noise=[0.002, 0.002, 0.002, 0.03, 0.03, 0.03],
        measurement_noise={
            "gps": sensors["gps"]["position_noise_std_m"] ** 2,
            "barometer": sensors["barometer"]["altitude_noise_std_m"] ** 2,
        },
    )

    latest_acceleration = np.zeros(3, dtype=float)
    records: list[dict[str, object]] = []
    gps_errors: list[float] = []
    estimate_errors_at_gps: list[float] = []
    total_duration_s = duration_s or demo_config.get("duration_s", 30)

    for frame in simulator.run(total_duration_s):
        measurements = frame.measurements
        imu = _measurement_values(measurements.get("imu"))
        if imu is not None:
            latest_acceleration = np.array(
                [imu["ax_mps2"], imu["ay_mps2"], imu["az_mps2"]],
                dtype=float,
            )

        estimator.predict(simulator.dt_s, latest_acceleration)

        gps = _measurement_values(measurements.get("gps"))
        if gps is not None:
            estimator.update(gps, "gps")

        barometer = _measurement_values(measurements.get("barometer"))
        if barometer is not None:
            estimator.update(barometer, "barometer")

        truth = np.array(
            [frame.true_state.x_m, frame.true_state.y_m, frame.true_state.z_m],
            dtype=float,
        )
        estimated = np.array(estimator.snapshot().position_m, dtype=float)

        if gps is not None:
            gps_position = np.array([gps["x_m"], gps["y_m"], gps["z_m"]], dtype=float)
            gps_errors.append(float(np.linalg.norm(gps_position - truth)))
            estimate_errors_at_gps.append(float(np.linalg.norm(estimated - truth)))

        records.append(
            {
                "type": "estimation_frame",
                "timestamp_s": round(frame.timestamp_s, 6),
                "true_position_m": tuple(float(value) for value in truth),
                "gps_position_m": (
                    None
                    if gps is None
                    else (float(gps["x_m"]), float(gps["y_m"]), float(gps["z_m"]))
                ),
                "barometer_altitude_m": None if barometer is None else float(barometer["altitude_m"]),
                "estimated_state": estimator.snapshot().to_dict(),
            }
        )

    result = EstimationDemoResult(
        frames=records,
        rmse={
            "raw_gps_m": _root_mean_square(gps_errors),
            "kalman_estimate_m": _root_mean_square(estimate_errors_at_gps),
        },
    )
    if output_path is not None:
        write_estimation_jsonl_log(result, output_path)
    return result


def run_demo(
    *,
    duration_s: float | None = None,
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> Path:
    """Run the default 30 second simulation demo."""
    config = load_environment_config()
    demo_config = config.get("demo", {})
    simulator = build_simulator_from_config(config, seed=seed)
    frames = simulator.run(duration_s or demo_config.get("duration_s", 30))
    return write_jsonl_log(frames, output_path or demo_config["output_path"])


def _measurement_values(measurement: dict[str, object] | None) -> dict[str, float] | None:
    if measurement is None:
        return None
    values = measurement.get("values")
    if not isinstance(values, dict):
        return None
    return {str(key): float(value) for key, value in values.items()}


def _root_mean_square(errors: list[float]) -> float:
    if not errors:
        return 0.0
    return float(np.sqrt(np.mean(np.square(errors))))


def main() -> None:
    """CLI entry point for the simulator demo."""
    parser = argparse.ArgumentParser(description="Run the drone autonomy simulator demo.")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    output_path = run_demo(duration_s=args.duration_s, output_path=args.output, seed=args.seed)
    print(f"Wrote simulation log to {output_path}")


if __name__ == "__main__":
    main()
