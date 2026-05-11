"""Run the reproducible simulator demo and write a JSONL log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drone_autonomy_lab.config import PROJECT_ROOT, load_environment_config
from drone_autonomy_lab.simulation.perception import DetectionObject, PerceptionMock
from drone_autonomy_lab.simulation.sensors import BarometerSensor, GPSSensor, IMUSensor, SensorSuite
from drone_autonomy_lab.simulation.simulator import DroneSimulator, SimulationFrame


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
