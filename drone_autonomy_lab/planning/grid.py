"""Occupancy-grid mapping for waypoint planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import ceil, floor, hypot
from typing import Iterable, Mapping, Sequence

import numpy as np

from drone_autonomy_lab.config import load_environment_config
from drone_autonomy_lab.estimation import VehicleState
from drone_autonomy_lab.planning.waypoint import Waypoint


GridCell = tuple[int, int]
WorldPoint2D = tuple[float, float]


@dataclass(frozen=True)
class GridObstacle:
    """Axis-aligned rectangular obstacle in world coordinates."""

    obstacle_id: str
    center_m: WorldPoint2D
    size_m: WorldPoint2D


@dataclass(frozen=True)
class ReplanResult:
    """Result returned when checking whether a path must be replanned."""

    path: list[WorldPoint2D] | None
    replanned: bool
    event_log: list[str] = field(default_factory=list)


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


class AStarPlanner:
    """A* planner over an `OccupancyGrid`."""

    def __init__(self, grid: OccupancyGrid, *, allow_diagonal: bool = False) -> None:
        self.grid = grid
        self.allow_diagonal = allow_diagonal

    def find_path(self, start_m: WorldPoint2D, goal_m: WorldPoint2D) -> list[WorldPoint2D] | None:
        """Find a world-coordinate path or return None when unreachable."""
        start = self.grid.world_to_grid(start_m)
        goal = self.grid.world_to_grid(goal_m)
        if self.grid.is_occupied(start) or self.grid.is_occupied(goal):
            return None

        open_set: list[tuple[float, GridCell]] = []
        heappush(open_set, (0.0, start))
        came_from: dict[GridCell, GridCell] = {}
        cost_so_far: dict[GridCell, float] = {start: 0.0}

        while open_set:
            _, current = heappop(open_set)
            if current == goal:
                return [self.grid.grid_to_world(cell) for cell in _reconstruct_path(came_from, current)]

            for neighbor in self._neighbors(current):
                step_cost = hypot(neighbor[0] - current[0], neighbor[1] - current[1])
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + _heuristic(neighbor, goal)
                    heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current

        return None

    def plan_to_waypoint(self, state: VehicleState | WorldPoint2D, waypoint: Waypoint) -> list[WorldPoint2D] | None:
        """Plan from the current vehicle position to a mission waypoint."""
        if isinstance(state, VehicleState):
            start_m = (state.position_m[0], state.position_m[1])
        else:
            start_m = state
        return self.find_path(start_m, (waypoint.x_m, waypoint.y_m))

    def replan_if_blocked(
        self,
        *,
        current_path: Sequence[WorldPoint2D] | None,
        start_m: WorldPoint2D,
        goal_m: WorldPoint2D,
        new_obstacles: Iterable[GridObstacle | Mapping[str, object]],
    ) -> ReplanResult:
        """Add new obstacles and replan if they block the current path."""
        event_log: list[str] = []
        for obstacle in new_obstacles:
            grid_obstacle = _coerce_obstacle(obstacle)
            self.grid.add_obstacle(grid_obstacle)
            event_log.append(f"obstacle_added:{grid_obstacle.obstacle_id}")

        blocked = current_path is None or any(
            self.grid.is_occupied(self.grid.world_to_grid(point))
            for point in current_path
        )
        if not blocked:
            return ReplanResult(path=list(current_path), replanned=False, event_log=event_log)

        event_log.append("replan_triggered:path_blocked")
        return ReplanResult(
            path=self.find_path(start_m, goal_m),
            replanned=True,
            event_log=event_log,
        )

    def _neighbors(self, cell: GridCell) -> list[GridCell]:
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.allow_diagonal:
            offsets.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

        neighbors = [(cell[0] + dx, cell[1] + dy) for dx, dy in offsets]
        return [
            neighbor
            for neighbor in neighbors
            if self.grid.in_bounds(neighbor) and not self.grid.is_occupied(neighbor)
        ]


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


def _heuristic(a: GridCell, b: GridCell) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct_path(came_from: dict[GridCell, GridCell], current: GridCell) -> list[GridCell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
