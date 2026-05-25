"""Visualize a pre-evolved GA-II agent foraging on a fixed map, producing an animated GIF."""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import PillowWriter

from robby.environment import Environment2D
from robby.perception import PerceptionMapper, WALL
from robby.simulation import Action

NUM_MOVES = 200
NUM_ENVIRONMENT_ROWS = NUM_ENVIRONMENT_COLUMNS = 10

NORTH_BASE = 81
SOUTH_BASE = 27
EAST_BASE = 9
WEST_BASE = 3
CENTER_BASE = 1

GREEN = 1
RED = 0

WALL_PENALTY = -5
WATER_COLOUR_BASE = 243
STARTING_COORDINATES = 4
ROBBY_POSITION_NUMBER = 12
WATER_QUANTITIES = 11

fitness_points_per_quantity = {
    0: 0, 1: 2, 2: 3, 3: 7, 4: 9, 5: 10,
    6: 8, 7: 6, 8: 5, 9: 4, 10: 1,
}

individual = [
    6, 6, 5, 1, 0, 1, 4, 1, 5, 3, 3, 5, 1, 2, 5, 1, 2, 6, 3, 3, 5, 0, 0, 1,
    2, 0, 0, 3, 4, 5, 0, 0, 5, 2, 5, 0, 2, 6, 0, 0, 4, 5, 2, 4, 5, 3, 6, 5,
    0, 1, 1, 2, 6, 5, 0, 1, 0, 5, 0, 3, 0, 0, 5, 0, 2, 4, 3, 5, 4, 3, 2, 2,
    6, 1, 3, 2, 5, 5, 3, 1, 5, 3, 4, 5, 1, 1, 5, 1, 4, 5, 3, 5, 5, 1, 2, 5,
    1, 0, 5, 3, 2, 5, 0, 1, 1, 6, 2, 5, 3, 1, 5, 2, 6, 5, 2, 4, 5, 3, 0, 5,
    2, 6, 5, 2, 4, 0, 6, 3, 5, 0, 3, 5, 4, 1, 5, 0, 2, 5, 0, 5, 2, 4, 1, 5,
    0, 4, 5, 0, 6, 2, 2, 1, 0, 2, 2, 5, 6, 1, 6, 1, 1, 5, 1, 5, 2, 5, 0, 3,
    4, 5, 5, 1, 4, 5, 1, 0, 1, 1, 2, 5, 3, 5, 3, 5, 6, 5, 4, 1, 5, 1, 4, 2,
    5, 2, 5, 0, 3, 5, 3, 3, 5, 3, 5, 4, 5, 3, 5, 3, 4, 5, 3, 2, 3, 5, 3, 6,
    5, 1, 6, 0, 6, 5, 2, 0, 5, 0, 1, 4, 4, 0, 4, 5, 5, 5, 3, 0, 2, 1, 3, 0,
    4, 1, 2, 1, 2, 0, 6, 0, 2, 6, 4, 6, 2, 0,
]

fixed_map = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, 8, 0, 2, 2, 9, 6, 0, 3, 5, 4, -1],
    [-1, 9, 9, 10, 8, 2, 7, 8, 1, 5, 0, -1],
    [-1, 5, 6, 6, 9, 5, 6, 6, 3, 5, 5, -1],
    [-1, 4, 9, 10, 9, 2, 10, 7, 8, 4, 2, -1],
    [-1, 5, 5, 2, 10, 0, 5, 2, 5, 10, 9, -1],
    [-1, 8, 1, 3, 6, 7, 0, 7, 1, 4, 9, -1],
    [-1, 2, 10, 1, 5, 10, 3, 10, 9, 8, 10, -1],
    [-1, 1, 7, 0, 10, 9, 4, 9, 4, 9, 3, -1],
    [-1, 3, 8, 1, 3, 3, 4, 10, 3, 5, 10, -1],
    [-1, 4, 2, 4, 5, 5, 0, 5, 1, 8, 8, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
]


class VisualEnvironment(Environment2D):
    def __init__(self):
        super().__init__(init=False)
        self.grid = np.array(fixed_map, dtype=int)

    def draw_perceived_frame(self, row, col, water_configuration):
        value = int(self.grid[row, col])
        self.grid[row, col] = ROBBY_POSITION_NUMBER
        matrix = np.array(self.grid, dtype=int)
        self.grid[row, col] = value

        colour_list = ["lightyellow"]
        colour_list.append("lightgrey")
        for wc in water_configuration:
            colour_list.append("red" if wc % 2 == 1 else "lightgreen")
        colour_list.append("darkblue")
        colour_list.append("darkblue")

        return sns.heatmap(
            matrix, annot=True, square=True, cbar=False,
            vmin=-3, vmax=12, center=5, cmap=colour_list,
            xticklabels="", yticklabels="",
        )


env = VisualEnvironment()
mapper = PerceptionMapper(individual, WATER_COLOUR_BASE)
robby_row = 3
robby_column = 1
pick_up_rate = 0
collected_water = 0

fig = plt.figure()
ax = sns.heatmap(
    np.array(env.grid, dtype=int), annot=True, square=True, cbar=False,
    vmin=-2, vmax=12, center=5, cmap="Reds",
    xticklabels="", yticklabels="",
)


def perform_action(action):
    global robby_row, robby_column, pick_up_rate, collected_water
    reward = 0

    nr, nc = robby_row - 1, robby_column
    if action == Action.MOVE_NORTH:
        if (0 < nr < NUM_ENVIRONMENT_ROWS + 1) and (0 < nc < NUM_ENVIRONMENT_COLUMNS + 1):
            robby_row = nr
        else:
            reward = WALL_PENALTY

    nr, nc = robby_row + 1, robby_column
    if action == Action.MOVE_SOUTH:
        if (0 < nr < NUM_ENVIRONMENT_ROWS + 1) and (0 < nc < NUM_ENVIRONMENT_COLUMNS + 1):
            robby_row = nr
        else:
            reward = WALL_PENALTY

    nr, nc = robby_row, robby_column + 1
    if action == Action.MOVE_EAST:
        if (0 < nr < NUM_ENVIRONMENT_ROWS + 1) and (0 < nc < NUM_ENVIRONMENT_COLUMNS + 1):
            robby_column = nc
        else:
            reward = WALL_PENALTY

    nr, nc = robby_row, robby_column - 1
    if action == Action.MOVE_WEST:
        if (0 < nr < NUM_ENVIRONMENT_ROWS + 1) and (0 < nc < NUM_ENVIRONMENT_COLUMNS + 1):
            robby_column = nc
        else:
            reward = WALL_PENALTY

    if action == Action.PICK_UP:
        pick_up_rate += 1
        print(f"Pick up --> {robby_row}  {robby_column}")
        water_quantity = int(env.grid[robby_row, robby_column])
        collected_water += fitness_points_per_quantity[water_quantity]
        env.grid[robby_row, robby_column] = 0

    if action == Action.RANDOM_MOVE:
        random_step = np.random.randint(0, 4)
        perform_action(random_step)

    return reward


def foraging(i):
    global robby_row, robby_column, ax
    state = 0

    state += CENTER_BASE * mapper.map_water_to_colour(int(env.grid[robby_row, robby_column]))
    state += NORTH_BASE * (
        WALL if robby_row - 1 <= 0 or robby_row - 1 >= NUM_ENVIRONMENT_ROWS + 1
        else mapper.map_water_to_colour(int(env.grid[robby_row - 1, robby_column]))
    )
    state += EAST_BASE * (
        WALL if robby_column + 1 <= 0 or robby_column + 1 >= NUM_ENVIRONMENT_COLUMNS + 1
        else mapper.map_water_to_colour(int(env.grid[robby_row, robby_column + 1]))
    )
    state += SOUTH_BASE * (
        WALL if robby_row + 1 <= 0 or robby_row + 1 >= NUM_ENVIRONMENT_ROWS + 1
        else mapper.map_water_to_colour(int(env.grid[robby_row + 1, robby_column]))
    )
    state += WEST_BASE * (
        WALL if robby_column - 1 <= 0 or robby_column - 1 >= NUM_ENVIRONMENT_COLUMNS + 1
        else mapper.map_water_to_colour(int(env.grid[robby_row, robby_column - 1]))
    )

    action = individual[state - 1]
    perform_action(action)

    plt.clf()
    ax = env.draw_perceived_frame(robby_row, robby_column, individual[-WATER_QUANTITIES:])
    return ax


ani = animation.FuncAnimation(fig, foraging, frames=NUM_MOVES, interval=1000)
writer = PillowWriter(fps=5)
ani.save("fix_map_ga_ii_visualized.gif", writer=writer)
print(f"Pick ups: {pick_up_rate}, collected: {collected_water}")
