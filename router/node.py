from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from numpy import sqrt

from router.common import distance


class NodeCategory(Enum):
    sink = "sink"
    relay = "relay"
    sensor = "sensor"


# physical parameters
epsilon_fs = 10e-12  # amplifier coefficient in free space model
epsilon_mp = 0.0013e-12  # amplifier coefficient in multi-path model
dist_threshold = sqrt(epsilon_fs / epsilon_mp)


class Node:
    # constants
    default_e0_tx = 50e-9  # J
    default_e0_rx = 50e-9
    default_energy_max = 0.5
    default_alive_threshold = 0.001

    def __init__(
            self,
            position: np.array,
            category: NodeCategory,
            *,
            energy: Optional[float] = None
    ):
        self.position = position
        assert isinstance(category, NodeCategory)
        self.category = category

        self.alive_threshold = Node.default_alive_threshold
        if not energy:
            self.energy = Node.default_energy_max
        else:
            self.energy = energy

        self.e0_tx = Node.default_e0_tx
        self.e0_rx = Node.default_e0_rx

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
        if dist <= dist_threshold:
            # free space
            return size * self.e0_tx + size * epsilon_fs * (dist ** 2)
        else:
            # multi-path
            return size * self.e0_tx + size * epsilon_mp * (dist ** 4)

    def energy_rx(self, size: int) -> float:
        return size * self.e0_rx

    def is_alive(self) -> bool:
        return self.energy > self.alive_threshold
