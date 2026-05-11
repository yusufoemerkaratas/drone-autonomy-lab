"""Autonomy-loop vocabulary shared by the stack and documentation."""

AUTONOMY_LOOP = (
    "Sensors",
    "State Estimation",
    "Planning",
    "Control",
    "Mission/Safety Logic",
)


def describe_autonomy_loop() -> str:
    """Return the canonical high-level autonomy pipeline."""
    return " -> ".join(AUTONOMY_LOOP)
