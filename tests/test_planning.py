from drone_autonomy_lab.config import load_environment_config
from drone_autonomy_lab.estimation import VehicleState
from drone_autonomy_lab.planning import AStarPlanner, GridObstacle, OccupancyGrid, Waypoint


def test_occupancy_grid_loads_obstacles_from_environment_config():
    grid = OccupancyGrid.from_environment_config(load_environment_config())

    obstacle_cell = grid.world_to_grid((5.0, 2.5))
    assert grid.is_occupied(obstacle_cell)


def test_astar_finds_path_in_empty_map():
    planner = AStarPlanner(OccupancyGrid(width_m=5.0, height_m=5.0, resolution_m=1.0))

    path = planner.find_path((0.5, 0.5), (4.5, 4.5))

    assert path is not None
    assert path[0] == (0.5, 0.5)
    assert path[-1] == (4.5, 4.5)


def test_astar_avoids_static_obstacles():
    grid = OccupancyGrid(
        width_m=6.0,
        height_m=6.0,
        resolution_m=1.0,
        obstacles=[GridObstacle("box", center_m=(2.5, 0.5), size_m=(1.0, 1.0))],
    )
    planner = AStarPlanner(grid)

    path = planner.find_path((0.5, 0.5), (5.5, 0.5))

    assert path is not None
    assert all(not grid.is_occupied(grid.world_to_grid(point)) for point in path)
    assert any(point[1] > 0.5 for point in path)


def test_astar_returns_none_for_unreachable_goal():
    grid = OccupancyGrid(
        width_m=5.0,
        height_m=5.0,
        resolution_m=1.0,
        obstacles=[GridObstacle("wall", center_m=(2.5, 2.5), size_m=(1.0, 5.0))],
    )
    planner = AStarPlanner(grid)

    assert planner.find_path((0.5, 2.5), (4.5, 2.5)) is None


def test_world_and_grid_coordinate_conversion():
    grid = OccupancyGrid(width_m=10.0, height_m=10.0, resolution_m=0.5, origin_m=(-1.0, -1.0))

    cell = grid.world_to_grid((0.25, 1.25))

    assert cell == (2, 4)
    assert grid.grid_to_world(cell) == (0.25, 1.25)


def test_planner_requests_path_to_next_waypoint_from_vehicle_state():
    planner = AStarPlanner(OccupancyGrid(width_m=6.0, height_m=6.0, resolution_m=1.0))
    state = VehicleState(position_m=(0.5, 0.5, 2.0), velocity_mps=(0.0, 0.0, 0.0), yaw_rad=0.0)
    waypoint = Waypoint(x_m=5.5, y_m=5.5, z_m=2.0)

    path = planner.plan_to_waypoint(state, waypoint)

    assert path is not None
    assert path[0] == (0.5, 0.5)
    assert path[-1] == (5.5, 5.5)


def test_replanning_triggers_when_new_obstacle_blocks_path():
    grid = OccupancyGrid(width_m=6.0, height_m=6.0, resolution_m=1.0)
    planner = AStarPlanner(grid)
    current_path = planner.find_path((0.5, 0.5), (5.5, 0.5))

    result = planner.replan_if_blocked(
        current_path=current_path,
        start_m=(0.5, 0.5),
        goal_m=(5.5, 0.5),
        new_obstacles=[GridObstacle("new-box", center_m=(2.5, 0.5), size_m=(1.0, 1.0))],
    )

    assert result.replanned is True
    assert "replan_triggered:path_blocked" in result.event_log
    assert result.path is not None
    assert all(not grid.is_occupied(grid.world_to_grid(point)) for point in result.path)
