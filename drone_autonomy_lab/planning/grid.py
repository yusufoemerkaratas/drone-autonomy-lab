"""Occupancy-grid mapping for waypoint planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable, Mapping

import numpy as np

from drone_autonomy_lab.config import load_environment_config


GridCell = tuple[int, int]
WorldPoint2D = tuple[float, float]


@dataclass(frozen=True)
class GridObstacle:
    """Axis-aligned rectangular obstacle in world coordinates."""

    obstacle_id: str
    center_m: WorldPoint2D
    size_m: WorldPoint2D


class OccupancyGrid:
    """2D occupancy map for local waypoint planning."""

    def __init__(
        self,
        *,
        width_m: float,
        height_m: float,
        resolution_m: float,
        origin_m: WorldPoint2D = (0.0, 0.0),
        obstacles: Iterable[GridObstacle | Mapping[str, object]] = (),
        inflation_radius_m: float = 0.0,
    ) -> None:
        if width_m <= 0 or height_m <= 0:
            raise ValueError("width_m and height_m must be positive")
        if resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if inflation_radius_m < 0:
            raise ValueError("inflation_radius_m cannot be negative")

        self.width_m = width_m
        self.height_m = height_m
        self.resolution_m = resolution_m
        self.origin_m = origin_m
        self.width_cells = int(ceil(width_m / resolution_m))
        self.height_cells = int(ceil(height_m / resolution_m))
        self.inflation_radius_m = inflation_radius_m
        self.occupied = np.zeros((self.height_cells, self.width_cells), dtype=bool)

        for obstacle in obstacles:
            self.add_obstacle(_coerce_obstacle(obstacle), inflation_radius_m=inflation_radius_m)

    @classmethod
    def from_environment_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        inflate: bool = True,
    ) -> "OccupancyGrid":
        """Build a grid from `configs/environment.yaml` planning settings."""
        environment_config = config or load_environment_config()
        planning = environment_config.get("planning", {})
        if not isinstance(planning, Mapping):
            raise ValueError("environment planning config must be a mapping")

        grid = planning.get("grid", {})
        if not isinstance(grid, Mapping):
            raise ValueError("planning.grid config must be a mapping")

        inflation_radius_m = float(planning.get("inflation_radius_m", 0.0)) if inflate else 0.0
        return cls(
            width_m=float(grid["width_m"]),
            height_m=float(grid["height_m"]),
            resolution_m=float(grid["resolution_m"]),
            origin_m=tuple(grid.get("origin_m", (0.0, 0.0))),  # type: ignore[arg-type]
            obstacles=planning.get("obstacles", ()),
            inflation_radius_m=inflation_radius_m,
        )

    def add_obstacle(self, obstacle: GridObstacle, *, inflation_radius_m: float | None = None) -> None:
        """Mark all cells covered by a rectangular obstacle."""
        inflation = self.inflation_radius_m if inflation_radius_m is None else inflation_radius_m
        half_x = obstacle.size_m[0] / 2.0 + inflation
        half_y = obstacle.size_m[1] / 2.0 + inflation
        min_x = obstacle.center_m[0] - half_x
        max_x = obstacle.center_m[0] + half_x
        min_y = obstacle.center_m[1] - half_y
        max_y = obstacle.center_m[1] + half_y

        min_cell = self.world_to_grid((min_x, min_y), clamp=True)
        max_cell = self.world_to_grid((max_x, max_y), clamp=True)
        for y in range(min_cell[1], max_cell[1] + 1):
            for x in range(min_cell[0], max_cell[0] + 1):
                if self.in_bounds((x, y)):
                    self.occupied[y, x] = True

    def world_to_grid(self, point_m: WorldPoint2D, *, clamp: bool = False) -> GridCell:
        """Convert world coordinates to integer grid coordinates."""
        x = int(floor((point_m[0] - self.origin_m[0]) / self.resolution_m))
        y = int(floor((point_m[1] - self.origin_m[1]) / self.resolution_m))
        if clamp:
            x = min(max(x, 0), self.width_cells - 1)
            y = min(max(y, 0), self.height_cells - 1)
        return (x, y)

    def grid_to_world(self, cell: GridCell) -> WorldPoint2D:
        """Convert a grid cell to the world coordinate at its center."""
        return (
            self.origin_m[0] + (cell[0] + 0.5) * self.resolution_m,
            self.origin_m[1] + (cell[1] + 0.5) * self.resolution_m,
        )

    def in_bounds(self, cell: GridCell) -> bool:
        return 0 <= cell[0] < self.width_cells and 0 <= cell[1] < self.height_cells

    def is_occupied(self, cell: GridCell) -> bool:
        if not self.in_bounds(cell):
            return True
        return bool(self.occupied[cell[1], cell[0]])

    def clone(self) -> "OccupancyGrid":
        copy = OccupancyGrid(
            width_m=self.width_m,
            height_m=self.height_m,
            resolution_m=self.resolution_m,
            origin_m=self.origin_m,
            inflation_radius_m=self.inflation_radius_m,
        )
        copy.occupied = self.occupied.copy()
        return copy


def _coerce_obstacle(obstacle: GridObstacle | Mapping[str, object]) -> GridObstacle:
    if isinstance(obstacle, GridObstacle):
        return obstacle
    obstacle_id = str(obstacle.get("id", obstacle.get("obstacle_id", "obstacle")))
    center = tuple(obstacle["center_m"])  # type: ignore[arg-type]
    size = tuple(obstacle["size_m"])  # type: ignore[arg-type]
    return GridObstacle(
        obstacle_id=obstacle_id,
        center_m=(float(center[0]), float(center[1])),
        size_m=(float(size[0]), float(size[1])),
    )
