"""Shared sensor data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorReading:
    """A single timestamped measurement from a simulated sensor."""

    sensor_name: str
    timestamp_s: float
    values: tuple[float, ...]
