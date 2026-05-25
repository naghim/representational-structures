# Representational Structures

Evolution of perception and representational structures in artificial agents.

This project studies whether evolution favors agents tuned to **fitness** rather than **objective reality** - a question central to Donald Hoffman's interface theory of perception. It extends the classic "Robby the Robot" problem (Melanie Mitchell) using genetic algorithms and neuroevolution in both 2D and 3D environments.

> This is a modernized refactor of the [original thesis repository](https://github.com/naghim/Evolution-of-Representational-Structures), which remains archived at its original location.

## Overview

An agent ("Robby") navigates a grid collecting water resources with varying quantities. Its perception of water as "red" (low-value) or "green" (high-value) is encoded in its genome - evolution determines whether agents perceive reality veridically or develop interface strategies tuned to fitness.

### Experiments

| Experiment         | Space         | Controller                  | Actions                                | Key feature                    |
| ------------------ | ------------- | --------------------------- | -------------------------------------- | ------------------------------ |
| GA-I               | 2D (10×10)    | Genetic algorithm           | 7 (move N/S/E/W, stay, pickup, random) | Baseline foraging              |
| GA-II              | 2D (10×10)    | Genetic algorithm           | 9 (+ rotate left/right)                | Rotation + sight penalty       |
| GA-II 3D           | 3D (10×10×10) | Genetic algorithm           | 9 (+ move up/down)                     | Scalability to 3D              |
| GA-III             | 2D (10×10)    | Genetic algorithm           | 9                                      | Evolve which directions to see |
| NN (memory)        | 3D            | Neural network (19→30+1→18) | 14                                     | 4 memory feedback neurons      |
| NN (no memory)     | 3D            | Neural network (19→30+1→18) | 14                                     | No memory                      |
| NN (sight penalty) | 3D            | Neural network              | 14                                     | Evolve perception directions   |

## Getting started

```bash
uv sync                 # create venv and install all deps
uv run scripts/ga_i.py  # run GA-I experiment
```

## Repository layout

```
├── src/robby/            # Python package
│   ├── environment.py    # 2D/3D grid environments
│   ├── perception.py     # Perception encoding & state computation
│   ├── simulation.py     # Agent actions, rewards, simulation loop
│   ├── agent.py          # GA individual & neural network
│   └── experiments.py    # DEAP-based experiment runners
├── scripts/              # Entry points for experiments + visualization
├── UIs/                  # WPF apps for human-comparison studies
└── pyproject.toml
```

## Agent visualization

The `scripts/visualize_agent.py` script loads a pre-evolved GA-II genome and replays its behavior on a fixed 12×12 map, generating an animated GIF. Each frame shows the grid colored by Robby's perceived view: cells appear **red** or **green** depending on how the agent's water-color genes map the underlying quantity, and Robby's position is shown in dark blue. This way we can actually see how the evolved perception strategy drives foraging decisions.

## Companion UIs

The `UIs/` directory contains three WPF companion applications designed for **human-comparison studies**. They let human participants control Robby's behavior and test whether they can come up with a winning strategy, mirroring the evolutionary agent's task.

Each version progressively removes human-interpretable information:

| Version      | What the user sees                                                                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **UI ver 1** | Full grid layout with water quantities and Robby's position - maximum information                                                                  |
| **UI ver 2** | Grid structure preserved, but Robby has limited field of vision                                                                                    |
| **UI ver 3** | Everything stripped away - the user sees only **5 abstract symbols in a row** corresponding to a word, mimicking the agent's perception bottleneck |

## References

- [Mitchell, M _Robby the Robot_](https://melaniemitchell.me/ExplorationsContent/RobbyTheRobot/)
- Mitchell, M. _Complexity: A Guided Tour_. Oxford University Press, 2009.
- Hoffman, D. D. _The Case Against Reality_. W. W. Norton, 2019.
