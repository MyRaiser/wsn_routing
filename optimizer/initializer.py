import numpy as np
from numpy.random import rand

from . import Target


def logistic_map(target: Target, n_pop: int, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """use logistic chaos map to generate initial population"""
    func, _, lb, ub = target
    # generate logistic chaos values
    while (x := rand()) in {0.0, 0.25, 0.5, 0.75, 1.0}:
        continue
    chaos = []
    for i in range(n_pop):
        chaos.append(x)
        x = eta * x * (1 - x)
    chaos = np.array(chaos)

    pop = np.array(
        [chaos[i] * (ub - lb) + lb for i in range(n_pop)]
    )
    cost = np.array(
        [func(idt) for idt in pop]
    )
    return pop, cost
