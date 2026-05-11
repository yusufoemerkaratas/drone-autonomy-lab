from drone_autonomy_lab.config import load_environment_config, load_mission_config


def test_mission_config_defines_waypoints_and_safety_limits():
    config = load_mission_config()

    assert config["mission"]["name"] == "courtyard-survey"
    assert len(config["mission"]["waypoints"]) >= 2
    assert config["safety"]["max_altitude_m"] > 0
    assert config["safety"]["geofence_radius_m"] > 0


def test_environment_config_defines_sensor_noise_models():
    config = load_environment_config()

    assert config["environment"]["time_step_s"] > 0
    assert {"imu", "gps", "barometer"} <= set(config["sensors"])
    assert config["sensors"]["imu"]["update_rate_hz"] > config["sensors"]["gps"]["update_rate_hz"]
