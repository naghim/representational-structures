from __future__ import annotations

import numpy as np


class Environment2D:
    rows: int
    cols: int
    grid: np.ndarray

    def __init__(self, num_rows: int = 10, num_cols: int = 10, rng: np.random.Generator | None = None) -> None:
        self._n = num_rows + 2
        self._m = num_cols + 2
        self.grid = np.zeros((self._n, self._m), dtype=int)
        self._init_grid(rng or np.random.default_rng())

    def _init_grid(self, rng: np.random.Generator) -> None:
        for i in range(self._n):
            for j in range(self._m):
                if i == 0 or i == self._n - 1 or j == 0 or j == self._m - 1:
                    self.grid[i, j] = -2
                else:
                    self.grid[i, j] = rng.integers(0, 11)

    def get_value(self, row: int, col: int) -> int:
        return int(self.grid[row, col])

    def set_value(self, row: int, col: int, value: int) -> None:
        self.grid[row, col] = value

    def rotate_right(self, row: int, col: int) -> tuple[int, int]:
        N = self._n
        moved = False
        for x in range(N // 2):
            for y in range(x, N - x - 1):
                if row == x and col == y and not moved:
                    row = y
                    col = N - 1 - x
                    moved = True
                temp = self.grid[x, y]
                self.grid[x, y] = self.grid[N - 1 - y, x]
                self.grid[N - 1 - y, x] = self.grid[N - 1 - x, N - 1 - y]
                self.grid[N - 1 - x, N - 1 - y] = self.grid[y, N - 1 - x]
                self.grid[y, N - 1 - x] = temp
        return row, col

    def rotate_left(self, row: int, col: int) -> tuple[int, int]:
        N = self._n
        moved = False
        for x in range(N // 2):
            for y in range(x, N - x - 1):
                if row == x and col == y and not moved:
                    row = N - 1 - y
                    col = x
                    moved = True
                temp = self.grid[x, y]
                self.grid[x, y] = self.grid[y, N - 1 - x]
                self.grid[y, N - 1 - x] = self.grid[N - 1 - x, N - 1 - y]
                self.grid[N - 1 - x, N - 1 - y] = self.grid[N - 1 - y, x]
                self.grid[N - 1 - y, x] = temp
        return row, col


class Environment3D:
    depth: int
    rows: int
    cols: int
    cube: np.ndarray

    def __init__(self, size: int = 10, rng: np.random.Generator | None = None) -> None:
        self._s = size + 2
        self._n = size + 2
        self._m = size + 2
        self.cube = np.zeros((self._s, self._n, self._m), dtype=int)
        self._init_cube(rng or np.random.default_rng())

    def _init_cube(self, rng: np.random.Generator) -> None:
        for s in range(self._s):
            if s == 0 or s == self._s - 1:
                self.cube[s] = np.full((self._n, self._m), -2, dtype=int)
                continue
            for i in range(self._n):
                for j in range(self._m):
                    if i == 0 or i == self._n - 1 or j == 0 or j == self._m - 1:
                        self.cube[s, i, j] = -2
                    else:
                        self.cube[s, i, j] = rng.integers(0, 11)

    def get_value(self, sl: int, row: int, col: int) -> int:
        return int(self.cube[sl, row, col])

    def set_value(self, sl: int, row: int, col: int, value: int) -> None:
        self.cube[sl, row, col] = value

    def rotate_right(self, row: int, col: int) -> tuple[int, int]:
        N = self._n
        moved = False
        for z in range(1, self._s):
            mat = self.cube[z]
            for x in range(N // 2):
                for y in range(x, N - x - 1):
                    if row == x and col == y and not moved:
                        row = y
                        col = N - 1 - x
                        moved = True
                    temp = mat[x, y]
                    mat[x, y] = mat[N - 1 - y, x]
                    mat[N - 1 - y, x] = mat[N - 1 - x, N - 1 - y]
                    mat[N - 1 - x, N - 1 - y] = mat[y, N - 1 - x]
                    mat[y, N - 1 - x] = temp
        return row, col

    def rotate_left(self, row: int, col: int) -> tuple[int, int]:
        N = self._n
        moved = False
        for z in range(1, self._s):
            mat = self.cube[z]
            for x in range(N // 2):
                for y in range(x, N - x - 1):
                    if row == x and col == y and not moved:
                        row = N - 1 - y
                        col = x
                        moved = True
                    temp = mat[x, y]
                    mat[x, y] = mat[y, N - 1 - x]
                    mat[y, N - 1 - x] = mat[N - 1 - x, N - 1 - y]
                    mat[N - 1 - x, N - 1 - y] = mat[N - 1 - y, x]
                    mat[N - 1 - y, x] = temp
        return row, col

    def rotate_up(self, sl: int, row: int, col: int) -> tuple[int, int, int]:
        new_cube = np.zeros_like(self.cube)
        for z in range(self._s):
            for x in range(self._s):
                new_cube[z, x] = self.cube[x, self._n - z - 1]
        self.cube = new_cube
        return self._n - row - 1, sl, col

    def rotate_down(self, sl: int, row: int, col: int) -> tuple[int, int, int]:
        new_cube = np.zeros_like(self.cube)
        for z in range(self._s):
            for x in range(self._s):
                new_cube[z, x] = self.cube[self._s - x - 1, z]
        self.cube = new_cube
        return sl, self._s - row - 1, col
