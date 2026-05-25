from __future__ import annotations

import array
import multiprocessing
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from deap import algorithms, base, creator, tools

from robby.agent import NeuralNetwork, vector_to_matrix
from robby.environment import Environment2D, Environment3D
from robby.perception import PerceptionMapper


@dataclass
class GAConfig:
    num_moves: int = 200
    num_foraging: int = 100
    grid_size: int = 10
    pop_size: int = 200
    num_generations: int = 10
    cx_prob: float = 0.7
    mut_prob: float = 0.2
    tournament_size: int = 2
    seed: int = 170
    n_processes: int = 32
    genome_size: int = 243 + 11
    max_action: int = 6
    sight_penalty: float = 0.0
    extra_genes: int = 0


@dataclass
class NNConfig:
    num_moves: int = 200
    num_foraging: int = 100
    grid_size: int = 10
    pop_size: int = 200
    num_generations: int = 10
    cx_prob: float = 0.7
    mut_prob: float = 0.2
    tournament_size: int = 2
    seed: int = 170
    n_processes: int = 10
    genome_size: int = 1128 + 11
    sight_penalty: float = 0.0
    extra_genes: int = 0


BASES_2D: dict[str, int] = {
    "CENTER": 1, "WEST": 3, "EAST": 9, "SOUTH": 27, "NORTH": 81,
}

BASES_3D: dict[str, int] = {
    "CENTER": 1, "WEST": 3, "EAST": 9, "SOUTH": 27, "NORTH": 81,
    "UP": 243, "DOWN": 729,
}

BASES_NN: dict[str, int] = {
    "CENTER": 1, "WEST": 3, "EAST": 9, "SOUTH": 27, "NORTH": 81,
    "UP": 243, "DOWN": 729,
    "WEST2": 2187, "EAST2": 6561, "SOUTH2": 19683, "NORTH2": 59049,
    "UP2": 177147, "DOWN2": 531441,
}

SIGHT_GENES_2D: dict[str, int] = {
    "CENTER": 253, "WEST": 254, "EAST": 255, "SOUTH": 256, "NORTH": 257,
    "SIGHT_CURRENT": 254,
    "SIGHT_EAST": 255,
    "SIGHT_SOUTH": 256,
    "SIGHT_WEST": 253,
    "SIGHT_NORTH": 257,
}

SIGHT_GENES_NN: dict[str, int] = {
    "SIGHT_CURRENT": 1139,
    "SIGHT_EAST": 1140,
    "SIGHT_SOUTH": 1141,
    "SIGHT_WEST": 1138,
    "SIGHT_NORTH": 1142,
    "SIGHT_UP": 1143,
    "SIGHT_DOWN": 1144,
    "SIGHT_EAST_EAST": 1145,
    "SIGHT_SOUTH_SOUTH": 1146,
    "SIGHT_WEST_WEST": 1137,
    "SIGHT_NORTH_NORTH": 1147,
    "SIGHT_UP_UP": 1148,
    "SIGHT_DOWN_DOWN": 1149,
}


def _ensure_creator_types(typecode: str = "i") -> None:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    name = "Individual_GA" if typecode == "i" else "Individual_NN"
    if not hasattr(creator, name):
        creator.create(name, array.array, typecode=typecode, fitness=creator.FitnessMax)


def _setup_ga_toolbox(
    config: GAConfig,
    evaluate,
) -> base.Toolbox:
    _ensure_creator_types("i")

    toolbox = base.Toolbox()
    toolbox.register("attributes", random.randint, 0, config.max_action)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual_GA,
        toolbox.attributes,
        config.genome_size,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=config.max_action, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    toolbox.register("evaluate", evaluate)
    pool = multiprocessing.Pool(processes=config.n_processes)
    toolbox.register("map", pool.map)
    return toolbox


def _setup_nn_toolbox(
    config: NNConfig,
    evaluate,
) -> base.Toolbox:
    _ensure_creator_types("d")

    toolbox = base.Toolbox()
    toolbox.register("attributes", lambda: random.uniform(-1, 1))
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual_NN,
        toolbox.attributes,
        config.genome_size,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    toolbox.register("evaluate", evaluate)
    pool = multiprocessing.Pool(processes=config.n_processes)
    toolbox.register("map", pool.map)
    return toolbox


def _evaluate_ga_2d(
    individual,
    config: GAConfig,
    bases: dict[str, int],
    sight_genes: dict[str, int] | None = None,
) -> tuple[float]:
    mapper = PerceptionMapper(individual, len(individual) - 11)
    total = 0.0
    rng = np.random.default_rng()

    from robby.simulation import simulate_2d

    for _ in range(config.num_foraging):
        env = Environment2D(config.grid_size, config.grid_size, rng)
        total += simulate_2d(
            individual, env, config.num_moves, bases, mapper,
            config.grid_size, config.grid_size,
            rng, sight_genes, config.sight_penalty,
        )
    return (total / config.num_foraging,)


def _evaluate_ga_3d(
    individual,
    config: GAConfig,
    bases: dict[str, int],
) -> tuple[float]:
    mapper = PerceptionMapper(individual, config.genome_size - 11)
    total = 0.0
    rng = np.random.default_rng()

    from robby.simulation import simulate_3d

    for _ in range(config.num_foraging):
        env = Environment3D(config.grid_size, rng)
        total += simulate_3d(
            individual, env, config.num_moves, bases, mapper,
            config.grid_size, config.grid_size,
            rng,
        )
    return (total / config.num_foraging,)


def _evaluate_nn(
    individual,
    config: NNConfig,
    bases: dict[str, int],
    sight_genes: dict[str, int] | None = None,
) -> tuple[float]:
    mapper = PerceptionMapper(individual, 1128)
    weight1 = vector_to_matrix(
        np.array(individual, dtype=float),
        [19, 30], 0,
    )
    weight2 = vector_to_matrix(
        np.array(individual, dtype=float),
        [31, 18], 570,
    )
    network = NeuralNetwork(weight1, weight2)
    total = 0.0
    rng = np.random.default_rng()

    from robby.simulation import simulate_nn_3d

    for _ in range(config.num_foraging):
        env = Environment3D(config.grid_size, rng)
        total += simulate_nn_3d(
            individual, env, config.num_moves, bases, mapper,
            config.grid_size, config.grid_size,
            network.forward, rng,
            sight_genes, config.sight_penalty,
        )
    return (total / config.num_foraging,)


def run_ga_experiment(
    config: GAConfig,
    bases: dict[str, int] | None = None,
    sight_genes: dict[str, int] | None = None,
    is_3d: bool = False,
) -> tuple[list, tools.HallOfFame, tools.Statistics]:
    if bases is None:
        bases = BASES_3D if is_3d else BASES_2D

    if is_3d:

        def evaluate(ind):
            return _evaluate_ga_3d(ind, config, bases)
    else:

        def evaluate(ind):
            return _evaluate_ga_2d(ind, config, bases, sight_genes)

    toolbox = _setup_ga_toolbox(config, evaluate)
    random.seed(config.seed)

    pop = toolbox.population(n=config.pop_size)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(
        pop, toolbox, config.cx_prob, config.mut_prob,
        config.num_generations, stats=stats, halloffame=hof, verbose=True,
    )
    return pop, hof, stats


def run_nn_experiment(
    config: NNConfig,
    bases: dict[str, int] | None = None,
    sight_genes: dict[str, int] | None = None,
) -> tuple[list, tools.HallOfFame, tools.Statistics]:
    if bases is None:
        bases = BASES_NN

    def evaluate(ind):
        return _evaluate_nn(ind, config, bases, sight_genes)

    toolbox = _setup_nn_toolbox(config, evaluate)
    random.seed(config.seed)

    pop = toolbox.population(n=config.pop_size)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(
        pop, toolbox, config.cx_prob, config.mut_prob,
        config.num_generations, stats=stats, halloffame=hof, verbose=True,
    )
    return pop, hof, stats
