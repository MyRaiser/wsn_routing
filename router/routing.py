"""library of routing algorithms"""

from typing import Iterable, Callable
from itertools import combinations_with_replacement

from .node import Node


def prim(
        dist: Callable[[Node, Node], float],
        candidates: Iterable[Node],
        sink: Node
) -> list[tuple[Node, Node]]:
    """use Prim algorithm to form a tree"""
    visited = set()
    visited.add(sink)
    candidates = set(candidates)

    routes = []
    while candidates:
        min_src, min_dst = min(
            [
                min(
                    [(src, dst) for src in candidates],
                    key=lambda x: dist(x[0], x[1]),
                ) for dst in visited
            ],
            key=lambda x: dist(x[0], x[1]),
        )
        candidates.remove(min_src)
        visited.add(min_src)
        routes.append((min_src, min_dst))
    return routes


def greedy(
        cost: Callable[[Node, Node], float],
        candidates: Iterable[Node],
        sink: Node
) -> list[tuple[Node, Node]]:
    """using greedy method"""
    candidates = set(candidates)

    routes = []
    while candidates:
        min_src, min_dst = min(
            [(src, dst) for src, dst in combinations_with_replacement(candidates, 2)],
            key=lambda route: cost(*route)
        )
        candidates.remove(min_src)
        if min_src == min_dst:
            min_dst = sink
        routes.append((min_src, min_dst))
    return routes
