import numpy as np


def distance(p1: np.array, p2: np.array) -> float:
    """Euclidean distance of two vectors"""
    return np.linalg.norm(p1 - p2, 2)
