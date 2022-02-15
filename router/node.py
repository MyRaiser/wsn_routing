from __future__ import annotations

from enum import Enum

import numpy as np
from numpy import sqrt

from router.common import distance


class NodeCategory(Enum):
    sink = "sink"
    relay = "relay"
    sensor = "sensor"


class Node:
    # constants
    e0_tx = 50e-9  # J
    e0_rx = 50e-9
    energy_max = 0.5

    epsilon_fs = 10e-12  # amplifier coefficient in free space model
    epsilon_mp = 0.0013e-12  # amplifier coefficient in multi-path model
    dist_threshold = sqrt(epsilon_fs / epsilon_mp)

    def __init__(self, position: np.array, category: NodeCategory):
        self.position = position
        assert isinstance(category, NodeCategory)
        self.category = category
        if category == NodeCategory.sink:
            self.energy = float("inf")
        else:
            self.energy = Node.energy_max

    def broadcast(self, size: int, dist: float) -> bool:
        if self.is_alive():
            self.energy -= self.energy_tx(size, dist)
            return True
        return False

    def singlecast(self, size: int, target: Node) -> bool:
        """transmit several bits"""
        if self.is_alive():
            dist = distance(self.position, target.position)
            self.energy -= self.energy_tx(size, dist)
            if target.is_alive():
                target.energy -= target.energy_rx(size)
                return True
            return False
        return False

    def recv_broadcast(self, size: int) -> bool:
        """receive several bits from broadcast"""
        if self.is_alive():
            self.energy -= self.energy_rx(size)
            return True
        return False

    def energy_tx(self, size: int, dist: float) -> float:
        if dist <= self.dist_threshold:
            # free space
            return size * self.e0_tx + size * self.epsilon_fs * (dist ** 2)
        else:
            # multi-path
            return size * self.e0_tx + size * self.epsilon_mp * (dist ** 4)

    def energy_rx(self, size: int) -> float:
        return size * self.e0_rx

    def is_alive(self) -> bool:
        return self.energy > 0.001
