from drone_autonomy_lab import AUTONOMY_LOOP, describe_autonomy_loop


def test_autonomy_loop_matches_architecture_order():
    assert AUTONOMY_LOOP == (
        "Sensors",
        "State Estimation",
        "Planning",
        "Control",
        "Mission/Safety Logic",
    )


def test_autonomy_loop_description_is_readable():
    assert describe_autonomy_loop() == (
        "Sensors -> State Estimation -> Planning -> Control -> Mission/Safety Logic"
    )
