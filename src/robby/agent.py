from __future__ import annotations

import array

import numpy as np


class GAIndividual:
    typecode: str = "i"

    @staticmethod
    def create(attributes, n: int):
        return array.array("i", (attributes() for _ in range(n)))


class NeuralNetwork:
    def __init__(self, w1: np.ndarray, w2: np.ndarray) -> None:
        self.w1 = w1
        self.w2 = w2
        self._input_size = w1.shape[1]
        self._output_size = w2.shape[1]

    def forward(self, state: np.ndarray) -> tuple[int, np.ndarray]:
        hidden = self._relu(np.dot(state, self.w1))
        hidden = np.append(hidden, 1)
        output = np.dot(hidden, self.w2)
        action = int(np.argmax(output[:8]))
        state[14] = output[14]
        state[15] = output[15]
        state[16] = output[16]
        state[17] = output[17]
        return action, state

    @staticmethod
    def _relu(v: np.ndarray) -> np.ndarray:
        return np.maximum(v, 0)


def vector_to_matrix(
    vector: np.ndarray | array.array,
    dimensions: tuple[int, int],
    offset: int = 0,
) -> np.ndarray:
    matrix = np.zeros(dimensions)
    for i in range(dimensions[0]):
        start = i * dimensions[1] + offset
        end = start + dimensions[1]
        matrix[i] = vector[start:end]
    return matrix
