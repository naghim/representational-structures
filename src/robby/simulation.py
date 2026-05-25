from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum

import numpy as np

from robby.environment import Environment2D, Environment3D
from robby.perception import PerceptionMapper, compute_state_2d, compute_state_3d


class Action(IntEnum):
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    STAY_PUT = 4
    PICK_UP = 5
    RANDOM_MOVE = 6
    ROTATE_LEFT = 7
    ROTATE_RIGHT = 8
    MOVE_UP = 9
    MOVE_DOWN = 10
    ROTATE_FORWARD = 11
    ROTATE_BACKWARD = 12


WALL_PENALTY = -5

FITNESS_POINTS_PER_QUANTITY: dict[int, int] = {
    0: 0, 1: 1, 2: 3, 3: 6, 4: 9, 5: 10,
    6: 9, 7: 6, 8: 3, 9: 1, 10: 0,
}


def _is_in_bounds_2d(row: int, col: int, num_rows: int, num_cols: int) -> bool:
    return (0 < row < num_rows + 1) and (0 < col < num_cols + 1)


def _is_in_bounds_3d(sl: int, row: int, col: int, size: int) -> bool:
    return (0 < sl < size + 1) and (0 < row < size + 1) and (0 < col < size + 1)


def simulate_2d(
    genome: Sequence[int],
    env: Environment2D,
    num_moves: int,
    bases: dict[str, int],
    mapper: PerceptionMapper,
    num_rows: int,
    num_cols: int,
    rng: np.random.Generator | None = None,
    sight_genes: dict[str, int] | None = None,
    sight_penalty: float = 0.0,
) -> float:
    rng = rng or np.random.default_rng()
    row = col = 5
    total_reward = 0.0

    for _ in range(num_moves):
        state = compute_state_2d(
            env.grid, row, col, mapper, bases, num_rows, num_cols, sight_genes
        )
        action = genome[state - 1]
        reward = 0.0
        directions_used = 0

        if sight_genes:
            directions_used = sum(
                1 for g in sight_genes.values() if int(genome[g]) % 2 == 1
            )

        if action == Action.MOVE_NORTH:
            nr, nc = row - 1, col
            if _is_in_bounds_2d(nr, nc, num_rows, num_cols):
                row = nr
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_SOUTH:
            nr, nc = row + 1, col
            if _is_in_bounds_2d(nr, nc, num_rows, num_cols):
                row = nr
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_EAST:
            nr, nc = row, col + 1
            if _is_in_bounds_2d(nr, nc, num_rows, num_cols):
                col = nc
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_WEST:
            nr, nc = row, col - 1
            if _is_in_bounds_2d(nr, nc, num_rows, num_cols):
                col = nc
            else:
                reward = WALL_PENALTY

        elif action == Action.PICK_UP:
            water_quantity = env.get_value(row, col)
            reward = FITNESS_POINTS_PER_QUANTITY.get(water_quantity, 0)
            env.set_value(row, col, 0)

        elif action == Action.RANDOM_MOVE:
            random_step = int(rng.integers(0, 4))
            if random_step == 0:
                nr, nc = row - 1, col
                if _is_in_bounds_2d(nr, nc, num_rows, num_cols):
                    row = nr
                else:
                    reward = WALL_PENALTY
            elif random_step == 1:
                nr, nc = row + 1, col
                if _is_in_bounds_2d(nr, nc, num_rows, num_cols):
                    row = nr
                else:
                    reward = WALL_PENALTY
            elif random_step == 2:
                nr, nc = row, col + 1
                if _is_in_bounds_2d(nr, nc, num_rows, num_cols):
                    col = nc
                else:
                    reward = WALL_PENALTY
            else:
                nr, nc = row, col - 1
                if _is_in_bounds_2d(nr, nc, num_rows, num_cols):
                    col = nc
                else:
                    reward = WALL_PENALTY

        elif action == Action.ROTATE_LEFT:
            row, col = env.rotate_left(row, col)

        elif action == Action.ROTATE_RIGHT:
            row, col = env.rotate_right(row, col)

        total_reward += reward + (sight_penalty * directions_used)

    return total_reward





def simulate_3d(
    genome: Sequence[int],
    env: Environment3D,
    num_moves: int,
    bases: dict[str, int],
    mapper: PerceptionMapper,
    num_rows: int,
    num_cols: int,
    rng: np.random.Generator | None = None,
    sight_genes: dict[str, int] | None = None,
    sight_penalty: float = 0.0,
) -> float:
    rng = rng or np.random.default_rng()
    sl = row = col = 5
    total_reward = 0.0

    for _ in range(num_moves):
        state = compute_state_3d(
            env.cube, sl, row, col, mapper, bases, num_rows, num_cols, sight_genes
        )
        action = genome[state - 1]
        reward = 0.0
        directions_used = 0

        if sight_genes:
            directions_used = sum(
                1 for g in sight_genes.values() if int(genome[g]) % 2 == 1
            )

        if action == Action.MOVE_NORTH:
            nr, nc = row - 1, col
            if _is_in_bounds_3d(sl, nr, nc, num_rows):
                row = nr
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_SOUTH:
            nr, nc = row + 1, col
            if _is_in_bounds_3d(sl, nr, nc, num_rows):
                row = nr
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_EAST:
            nr, nc = row, col + 1
            if _is_in_bounds_3d(sl, nr, nc, num_rows):
                col = nc
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_WEST:
            nr, nc = row, col - 1
            if _is_in_bounds_3d(sl, nr, nc, num_rows):
                col = nc
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_UP:
            ns = sl + 1
            if _is_in_bounds_3d(ns, row, col, num_rows):
                sl = ns
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_DOWN:
            ns = sl - 1
            if _is_in_bounds_3d(ns, row, col, num_rows):
                sl = ns
            else:
                reward = WALL_PENALTY

        elif action == Action.PICK_UP:
            water_quantity = env.get_value(sl, row, col)
            reward = FITNESS_POINTS_PER_QUANTITY.get(water_quantity, 0)
            env.set_value(sl, row, col, 0)

        elif action == Action.RANDOM_MOVE:
            random_step = int(rng.integers(0, 6))
            if random_step == 0:
                nr, nc = row - 1, col
                if _is_in_bounds_3d(sl, nr, nc, num_rows):
                    row = nr
                else:
                    reward = WALL_PENALTY
            elif random_step == 1:
                nr, nc = row + 1, col
                if _is_in_bounds_3d(sl, nr, nc, num_rows):
                    row = nr
                else:
                    reward = WALL_PENALTY
            elif random_step == 2:
                nr, nc = row, col + 1
                if _is_in_bounds_3d(sl, nr, nc, num_rows):
                    col = nc
                else:
                    reward = WALL_PENALTY
            elif random_step == 3:
                nr, nc = row, col - 1
                if _is_in_bounds_3d(sl, nr, nc, num_rows):
                    col = nc
                else:
                    reward = WALL_PENALTY
            elif random_step == 4:
                ns = sl + 1
                if _is_in_bounds_3d(ns, row, col, num_rows):
                    sl = ns
                else:
                    reward = WALL_PENALTY
            else:
                ns = sl - 1
                if _is_in_bounds_3d(ns, row, col, num_rows):
                    sl = ns
                else:
                    reward = WALL_PENALTY
            row, col = env.rotate_left(row, col)

        elif action == Action.ROTATE_RIGHT:
            row, col = env.rotate_right(row, col)

        total_reward += reward + (sight_penalty * directions_used)

    return total_reward


def simulate_nn_3d(
    genome: Sequence[float],
    env: Environment3D,
    num_moves: int,
    bases: dict[str, int],
    mapper: PerceptionMapper,
    num_rows: int,
    num_cols: int,
    network: Callable[[np.ndarray], tuple[int, np.ndarray]],
    rng: np.random.Generator | None = None,
    sight_genes: dict[str, int] | None = None,
    sight_penalty: float = 0.0,
) -> float:
    rng = rng or np.random.default_rng()
    sl = row = col = 5
    total_reward = 0.0
    state = np.zeros(19)
    state[0] = 1

    for move_idx in range(1, num_moves + 1):
        directions_used = 0

        if sight_genes:
            _fill_state_sighted(state, env.cube, sl, row, col, mapper, bases, num_rows, num_cols, genome, sight_genes)
            directions_used = sum(1 for g in sight_genes.values() if genome[g] >= 0)
        else:
            _fill_state_full(state, env.cube, sl, row, col, mapper, bases, num_rows, num_cols)

        state[18] = move_idx

        action, state = network(state)
        reward = 0.0

        if action == Action.MOVE_NORTH:
            nr, nc = row - 1, col
            if _is_in_bounds_3d(sl, nr, nc, num_rows):
                row = nr
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_SOUTH:
            nr, nc = row + 1, col
            if _is_in_bounds_3d(sl, nr, nc, num_rows):
                row = nr
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_EAST:
            nr, nc = row, col + 1
            if _is_in_bounds_3d(sl, nr, nc, num_rows):
                col = nc
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_WEST:
            nr, nc = row, col - 1
            if _is_in_bounds_3d(sl, nr, nc, num_rows):
                col = nc
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_UP:
            ns = sl + 1
            if _is_in_bounds_3d(ns, row, col, num_rows):
                sl = ns
            else:
                reward = WALL_PENALTY

        elif action == Action.MOVE_DOWN:
            ns = sl - 1
            if _is_in_bounds_3d(ns, row, col, num_rows):
                sl = ns
            else:
                reward = WALL_PENALTY

        elif action == Action.PICK_UP:
            water_quantity = env.get_value(sl, row, col)
            reward = FITNESS_POINTS_PER_QUANTITY.get(water_quantity, 0)
            env.set_value(sl, row, col, 0)

        elif action == Action.RANDOM_MOVE:
            random_step = rng.integers(0, 6)
            if random_step == 0:
                nr, nc = row - 1, col
                if _is_in_bounds_3d(sl, nr, nc, num_rows):
                    row = nr
                else:
                    reward = WALL_PENALTY
            elif random_step == 1:
                nr, nc = row + 1, col
                if _is_in_bounds_3d(sl, nr, nc, num_rows):
                    row = nr
                else:
                    reward = WALL_PENALTY
            elif random_step == 2:
                nr, nc = row, col + 1
                if _is_in_bounds_3d(sl, nr, nc, num_rows):
                    col = nc
                else:
                    reward = WALL_PENALTY
            elif random_step == 3:
                nr, nc = row, col - 1
                if _is_in_bounds_3d(sl, nr, nc, num_rows):
                    col = nc
                else:
                    reward = WALL_PENALTY
            elif random_step == 4:
                ns = sl + 1
                if _is_in_bounds_3d(ns, row, col, num_rows):
                    sl = ns
                else:
                    reward = WALL_PENALTY
            elif random_step == 5:
                ns = sl - 1
                if _is_in_bounds_3d(ns, row, col, num_rows):
                    sl = ns
                else:
                    reward = WALL_PENALTY

        elif action == Action.ROTATE_LEFT:
            row, col = env.rotate_left(row, col)

        elif action == Action.ROTATE_RIGHT:
            row, col = env.rotate_right(row, col)

        elif action == 10:
            sl, row, col = env.rotate_up(sl, row, col)

        elif action == 11:
            sl, row, col = env.rotate_down(sl, row, col)

        total_reward += reward + (sight_penalty * directions_used)

    return total_reward


def _fill_state_full(
    state: np.ndarray,
    cube,
    sl: int,
    row: int,
    col: int,
    mapper: PerceptionMapper,
    bases: dict[str, int],
    num_rows: int,
    num_cols: int,
) -> None:
    directions = ["CENTER", "NORTH", "EAST", "SOUTH", "WEST", "UP", "DOWN",
                  "NORTH2", "EAST2", "SOUTH2", "WEST2", "UP2", "DOWN2"]
    offsets = {
        "CENTER": (0, 0, 0),
        "NORTH": (0, -1, 0), "NORTH2": (0, -2, 0),
        "EAST": (0, 0, 1), "EAST2": (0, 0, 2),
        "SOUTH": (0, 1, 0), "SOUTH2": (0, 2, 0),
        "WEST": (0, 0, -1), "WEST2": (0, 0, -2),
        "UP": (1, 0, 0), "UP2": (2, 0, 0),
        "DOWN": (-1, 0, 0), "DOWN2": (-2, 0, 0),
    }
    idx_map = {
        "CENTER": 1, "NORTH": 2, "NORTH2": 3, "EAST": 4, "EAST2": 5,
        "SOUTH": 6, "SOUTH2": 7, "WEST": 8, "WEST2": 9,
        "UP": 10, "UP2": 11, "DOWN": 12, "DOWN2": 13,
    }
    for direction in directions:
        ds, dr, dc = offsets[direction]
        ns, nr, nc = sl + ds, row + dr, col + dc
        state[idx_map[direction]] = _sense_cell_nn(
            cube, ns, nr, nc, mapper, bases[direction], num_rows
        )


def _fill_state_sighted(
    state: np.ndarray,
    cube,
    sl: int,
    row: int,
    col: int,
    mapper: PerceptionMapper,
    bases: dict[str, int],
    num_rows: int,
    num_cols: int,
    genome: Sequence[float],
    sight_genes: dict[str, int],
) -> None:
    directions_idx = [
        ("CENTER", "SIGHT_CURRENT", 1),
        ("NORTH", "SIGHT_NORTH", 2), ("NORTH2", "SIGHT_NORTH_NORTH", 3),
        ("EAST", "SIGHT_EAST", 4), ("EAST2", "SIGHT_EAST_EAST", 5),
        ("SOUTH", "SIGHT_SOUTH", 6), ("SOUTH2", "SIGHT_SOUTH_SOUTH", 7),
        ("WEST", "SIGHT_WEST", 8), ("WEST2", "SIGHT_WEST_WEST", 9),
        ("UP", "SIGHT_UP", 10), ("UP2", "SIGHT_UP_UP", 11),
        ("DOWN", "SIGHT_DOWN", 12), ("DOWN2", "SIGHT_DOWN_DOWN", 13),
    ]
    offsets = {
        "CENTER": (0, 0, 0), "NORTH": (0, -1, 0), "NORTH2": (0, -2, 0),
        "EAST": (0, 0, 1), "EAST2": (0, 0, 2),
        "SOUTH": (0, 1, 0), "SOUTH2": (0, 2, 0),
        "WEST": (0, 0, -1), "WEST2": (0, 0, -2),
        "UP": (1, 0, 0), "UP2": (2, 0, 0),
        "DOWN": (-1, 0, 0), "DOWN2": (-2, 0, 0),
    }
    for direction, sight_key, state_idx in directions_idx:
        gene_idx = sight_genes.get(sight_key)
        if gene_idx is not None and genome[gene_idx] < 0:
            continue
        ds, dr, dc = offsets[direction]
        ns, nr, nc = sl + ds, row + dr, col + dc
        state[state_idx] = _sense_cell_nn(
            cube, ns, nr, nc, mapper, 1, num_rows
        )


def _sense_cell_nn(
    cube,
    sl: int,
    row: int,
    col: int,
    mapper: PerceptionMapper,
    base: int,
    size: int,
) -> int:
    if (row <= 0 or row >= size + 1) or (col <= 0 or col >= size + 1) or (sl <= 0 or sl >= size + 1):
        return base * 2
    water_quantity = cube[sl][row][col]
    return base * mapper.map_water_to_colour(water_quantity)
