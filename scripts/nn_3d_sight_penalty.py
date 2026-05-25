"""Neuroevolution 3D with sight penalty: agents evolve which directions to perceive."""
from robby.experiments import NNConfig, BASES_NN, SIGHT_GENES_NN, run_nn_experiment

config = NNConfig(
    num_moves=200,
    num_foraging=100,
    grid_size=10,
    pop_size=200,
    num_generations=500,
    genome_size=1128 + 11 + 13,
    sight_penalty=-0.005,
    n_processes=10,
    seed=170,
)

if __name__ == "__main__":
    pop, hof, stats = run_nn_experiment(config, BASES_NN, SIGHT_GENES_NN)
    for h in hof:
        print(h)
