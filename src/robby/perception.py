from __future__ import annotations

from collections.abc import Sequence


WALL: int = 2
RED: int = 0
GREEN: int = 1


class PerceptionMapper:
    def __init__(self, genome: Sequence[float | int], colour_base: int) -> None:
        self._genome = genome
        self._colour_base = colour_base

    def map_water_to_colour(self, water_quantity: int) -> int:
        gene = self._genome[self._colour_base + water_quantity]
        if isinstance(gene, float):
            return GREEN if gene > 0 else RED
        return GREEN if int(gene) % 2 == 1 else RED


def sense_cell(
    env_grid: Sequence,
    row: int,
    col: int,
    mapper: PerceptionMapper,
    base: int,
    num_rows: int,
    num_cols: int,
    sl: int | None = None,
) -> int:
    if sl is not None:
        if (row <= 0 or row >= num_rows + 1) or (col <= 0 or col >= num_cols + 1) or (sl <= 0 or sl >= num_rows):
            return base * WALL
        water_quantity = env_grid[sl][row][col]
    else:
        if (row <= 0 or row >= num_rows + 1) or (col <= 0 or col >= num_cols + 1):
            return base * WALL
        water_quantity = env_grid[row][col]
    return base * mapper.map_water_to_colour(water_quantity)


def compute_state_2d(
    env_grid: Sequence,
    row: int,
    col: int,
    mapper: PerceptionMapper,
    bases: dict[str, int],
    num_rows: int,
    num_cols: int,
    sight_genes: dict[str, int] | None = None,
) -> int:
    state = 0
    directions = ["CENTER", "NORTH", "EAST", "SOUTH", "WEST"]
    offsets = {
        "CENTER": (0, 0),
        "NORTH": (-1, 0),
        "EAST": (0, 1),
        "SOUTH": (1, 0),
        "WEST": (0, -1),
    }

    for direction in directions:
        if sight_genes is not None:
            gene_idx = sight_genes.get(direction)
            if gene_idx is not None and int(mapper._genome[gene_idx]) % 2 == 0:
                continue
        dr, dc = offsets[direction]
        state += sense_cell(
            env_grid,
            row + dr,
            col + dc,
            mapper,
            bases[direction],
            num_rows,
            num_cols,
        )
    return state


def compute_state_3d(
    env_grid: Sequence,
    sl: int,
    row: int,
    col: int,
    mapper: PerceptionMapper,
    bases: dict[str, int],
    num_rows: int,
    num_cols: int,
    sight_genes: dict[str, int] | None = None,
) -> int:
    state = 0
    directions = ["CENTER", "NORTH", "EAST", "SOUTH", "WEST", "UP", "DOWN"]
    offsets = {
        "CENTER": (0, 0, 0),
        "NORTH": (0, -1, 0),
        "EAST": (0, 0, 1),
        "SOUTH": (0, 1, 0),
        "WEST": (0, 0, -1),
        "UP": (1, 0, 0),
        "DOWN": (-1, 0, 0),
    }

    for direction in directions:
        if sight_genes is not None:
            gene_idx = sight_genes.get(direction)
            if gene_idx is not None and int(mapper._genome[gene_idx]) % 2 == 0:
                continue
        ds, dr, dc = offsets[direction]
        state += sense_cell(
            env_grid,
            row + dr,
            col + dc,
            mapper,
            bases[direction],
            num_rows,
            num_cols,
            sl=sl + ds,
        )
    return state
