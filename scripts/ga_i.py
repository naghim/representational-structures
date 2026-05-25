"""GA-I: 2D grid, 7 actions, basic foraging."""
from robby.experiments import GAConfig, BASES_2D, run_ga_experiment

config = GAConfig(
    num_moves=200,
    num_foraging=100,
    grid_size=10,
    pop_size=200,
    num_generations=1000,
    genome_size=243 + 11,
    max_action=6,
    n_processes=32,
    seed=170,
)

if __name__ == "__main__":
    pop, hof, stats = run_ga_experiment(config, BASES_2D)
    for h in hof:
        print(h)
