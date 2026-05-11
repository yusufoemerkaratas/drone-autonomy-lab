import json

import numpy as np

from drone_autonomy_lab.config import load_environment_config
from drone_autonomy_lab.simulation import DroneSimulator, GPSSensor
from drone_autonomy_lab.simulation.demo import build_simulator_from_config, run_demo
from drone_autonomy_lab.simulation.state import DEFAULT_INITIAL_STATE


def test_drone_simulator_advances_state_with_fixed_dt():
    simulator = DroneSimulator(dt_s=0.1, seed=1)

    frame = simulator.step()

    assert frame.timestamp_s == 0.1
    assert frame.true_state.x_m > DEFAULT_INITIAL_STATE.x_m
    assert frame.true_state.y_m > DEFAULT_INITIAL_STATE.y_m
    assert frame.true_state.yaw_rad == DEFAULT_INITIAL_STATE.yaw_rate_rad_s * 0.1


def test_seed_reproducibility_for_truth_measurements_and_detections():
    config = load_environment_config()

    first_run = [
        frame.to_log_record()
        for frame in build_simulator_from_config(config, seed=99).run(duration_s=1.0)
    ]
    second_run = [
        frame.to_log_record()
        for frame in build_simulator_from_config(config, seed=99).run(duration_s=1.0)
    ]

    assert first_run == second_run


def test_gps_noise_moves_measurement_away_from_true_state():
    gps = GPSSensor(update_rate_hz=10, position_noise_std_m=0.4, dropout_probability=0.0)
    rng = np.random.default_rng(123)

    measurement = gps.measure(timestamp_s=0.1, state=DEFAULT_INITIAL_STATE, rng=rng)

    assert measurement is not None
    assert measurement.values["x_m"] != DEFAULT_INITIAL_STATE.x_m
    assert measurement.values["y_m"] != DEFAULT_INITIAL_STATE.y_m
    assert measurement.values["z_m"] != DEFAULT_INITIAL_STATE.z_m


def test_demo_writes_jsonl_log_with_truth_measurements_and_detections(tmp_path):
    output_path = tmp_path / "simulation.jsonl"

    run_demo(duration_s=0.4, output_path=output_path, seed=42)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records
    assert {"timestamp_s", "true_state", "sensor_measurements", "detections"} <= set(records[0])
    assert {"x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "yaw_rad", "yaw_rate_rad_s"} <= set(
        records[0]["true_state"]
    )
    assert any(record["sensor_measurements"]["gps"] is not None for record in records)
    assert any(record["detections"] for record in records)
