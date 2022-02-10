import numpy as np
from route import Node, NodeCategory, Router


def create_nodes_on_power_transmission_lines() -> list[Node]:
    dist_tower: float = 100  # distance between towers
    n_relay: int
    n_normal_each: int
    nodes = []
    sink = Node(np.array([0, 0]), NodeCategory.sink)
    nodes.append(sink)

    return nodes

