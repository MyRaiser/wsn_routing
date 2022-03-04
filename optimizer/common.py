from typing import Callable, Iterable

import numpy as np


Target = tuple[Callable, int, np.ndarray, np.ndarray]


def optimize(optimizer: Iterable, *, verbose: bool = False) -> tuple[np.array, float]:
    opt = None
    val = None
    for i, (x, y) in enumerate(optimizer):
        opt = x
        val = y
        if verbose:
            print(f"Iteration = {i}, optimum = {opt}, value = {val}")
    return opt, val
