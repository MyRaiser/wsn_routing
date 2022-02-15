from __future__ import annotations

from typing import Callable, Iterable, Any

import numpy as np


def distance(p1: np.array, p2: np.array) -> float:
    """Euclidean distance of two vectors"""
    return np.linalg.norm(p1 - p2, 2)


def argmin_(f: Callable, parameters: Iterable[Any]) -> Any:
    min_y = None
    min_x = None
    for x in parameters:
        y = f(x)
        if min_y is None or y < min_y:
            min_y = y
            min_x = x
    return min_x