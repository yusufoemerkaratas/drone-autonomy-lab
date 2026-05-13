import numpy as np

from drone_autonomy_lab.estimation import KalmanFilter
from drone_autonomy_lab.simulation.demo import run_estimation_demo


def test_kalman_predict_keeps_six_state_shape():
    estimator = KalmanFilter()

    state = estimator.predict(0.1, [1.0, 0.0, -0.5])

    assert state.shape == (6,)
    assert estimator.covariance.shape == (6, 6)
    assert state[0] > 0.0
    assert state[3] > 0.0


def test_gps_update_reduces_position_covariance():
    estimator = KalmanFilter(initial_covariance=10.0, measurement_noise={"gps": 0.25})
    before = np.trace(estimator.covariance[:2, :2])

    estimator.update({"x_m": 1.0, "y_m": 2.0}, "gps")

    after = np.trace(estimator.covariance[:2, :2])
    assert after < before


def test_gps_update_corrects_xy_without_overwriting_z():
    estimator = KalmanFilter(initial_state=[10.0, -10.0, 7.0, 0.0, 0.0, 0.0])

    estimator.update({"x_m": 2.0, "y_m": 3.0}, "gps")

    state = estimator.state_vector
    assert abs(state[0] - 2.0) < abs(10.0 - 2.0)
    assert abs(state[1] - 3.0) < abs(-10.0 - 3.0)
    assert state[2] == 7.0


def test_barometer_update_corrects_z():
    estimator = KalmanFilter(initial_state=[0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

    estimator.update({"altitude_m": 2.0}, "barometer")

    assert abs(estimator.state_vector[2] - 2.0) < abs(10.0 - 2.0)


def test_estimation_demo_improves_over_raw_gps_rmse(tmp_path):
    result = run_estimation_demo(duration_s=20.0, output_path=tmp_path / "estimation.jsonl", seed=42)

    assert result.rmse["raw_gps_m"] > 0.0
    assert result.rmse["kalman_estimate_m"] < result.rmse["raw_gps_m"]
    assert result.frames
    assert {"true_position_m", "gps_position_m", "estimated_state"} <= set(result.frames[0])
