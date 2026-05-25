"""GA-II 3D: 10x10x10 cube, 9 actions (adds up/down movement), 7-direction perception."""
from robby.experiments import GAConfig, BASES_3D, run_ga_experiment

config = GAConfig(
    num_moves=200,
    num_foraging=100,
    grid_size=10,
    pop_size=200,
    num_generations=10,
    genome_size=2187 + 11,
    max_action=8,
    n_processes=32,
    seed=170,
)

if __name__ == "__main__":
    pop, hof, stats = run_ga_experiment(config, BASES_3D, is_3d=True)
    for h in hof:
        print(h)
