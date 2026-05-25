"""Neuroevolution 3D: same NN architecture but WITHOUT memory feedback neurons."""
from robby.experiments import NNConfig, BASES_NN, run_nn_experiment

config = NNConfig(
    num_moves=200,
    num_foraging=100,
    grid_size=10,
    pop_size=200,
    num_generations=10,
    genome_size=1128 + 11,
    n_processes=10,
    seed=170,
)

if __name__ == "__main__":
    pop, hof, stats = run_nn_experiment(config, BASES_NN)
    for h in hof:
        print(h)
