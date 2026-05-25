"""GA-III: 2D grid, agents evolve which directions to look at, per-direction sight penalty."""
from robby.experiments import GAConfig, BASES_2D, SIGHT_GENES_2D, run_ga_experiment

config = GAConfig(
    num_moves=200,
    num_foraging=100,
    grid_size=10,
    pop_size=200,
    num_generations=10,
    genome_size=243 + 11 + 5,
    max_action=8,
    sight_penalty=-0.01,
    n_processes=1,
    seed=170,
)

if __name__ == "__main__":
    pop, hof, stats = run_ga_experiment(config, BASES_2D, SIGHT_GENES_2D)
    for h in hof:
        print(h)
