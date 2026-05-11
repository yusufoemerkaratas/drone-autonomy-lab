"""Mock perception detections for simulation-first integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_autonomy_lab.simulation.sensors import ScheduledSensor
from drone_autonomy_lab.simulation.state import DroneState


@dataclass(frozen=True)
class DetectionObject:
    """Static object that can be detected by the perception mock."""

    object_id: str
    object_type: str
    position_m: tuple[float, float, float]


class PerceptionMock(ScheduledSensor):
    """Emit noisy obstacle and target detections near the drone."""

    def __init__(
        self,
        *,
        update_rate_hz: float,
        detection_range_m: float,
        position_noise_std_m: float,
        objects: tuple[DetectionObject, ...],
    ) -> None:
        super().__init__(sensor_name="perception", update_rate_hz=update_rate_hz)
        self.detection_range_m = detection_range_m
        self.position_noise_std_m = position_noise_std_m
        self.objects = objects

    def detect(
        self,
        *,
        timestamp_s: float,
        state: DroneState,
        rng: np.random.Generator,
    ) -> list[dict[str, object]]:
        """Return visible detections at the configured update rate."""
        if not self.due(timestamp_s):
            return []

        self.mark_sampled()
        detections: list[dict[str, object]] = []
        drone_position = np.array([state.x_m, state.y_m, state.z_m])

        for candidate in self.objects:
            object_position = np.array(candidate.position_m)
            distance_m = float(np.linalg.norm(object_position - drone_position))
            if distance_m > self.detection_range_m:
                continue

            noise = rng.normal(0.0, self.position_noise_std_m, size=3)
            noisy_position = object_position + noise
            detections.append(
                {
                    "id": candidate.object_id,
                    "type": candidate.object_type,
                    "timestamp_s": round(timestamp_s, 6),
                    "position_m": [float(value) for value in noisy_position],
                    "distance_m": distance_m,
                    "confidence": max(0.1, 1.0 - distance_m / self.detection_range_m),
                }
            )

        return detections
