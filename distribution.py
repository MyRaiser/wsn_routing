from typing import Callable

import numpy as np
from numpy import pi, cos, sin
from numpy.random import rand

from router import Node, NodeCategory


Position = tuple[float, float]
Distribution = list[Position]


def simple_loader(
        position_sink: Position,
        distribution: Distribution
) -> tuple[Node, list[Node]]:
    sink = Node(np.array(position_sink), NodeCategory.sink)
    sensors = [
        Node(np.array(pos), NodeCategory.sensor) for pos in distribution
    ]
    return sink, sensors


def power_line_naive(
        n_relay: int,
        d_relay: float,
        d_jitter_max: float,
        phi_max: float,
        n_sensors_per_relay: int,
        r_max: float,
        sink: Position,
) -> Distribution:
    # sink_position = [0, 0]
    # interval = 100
    # d_max = 10
    # phi_max = pi / 10
    # n_relay = 5
    # n_sensor_per_relay = 10
    # r_max = 80

    relays = []
    x, y = sink
    for _ in range(n_relay):
        d = d_relay + d_jitter_max * rand()
        phi = phi_max * (2 * rand() - 1)
        x += d * cos(phi)
        y += d * sin(phi)
        relays.append((x, y))

    sensors = []
    for relay in relays:
        for _ in range(n_sensors_per_relay):
            r = r_max * rand()
            theta = 2 * pi * rand()
            rx, ry = relay
            x = rx + r * cos(theta)
            y = ry + r * sin(theta)
            sensors.append((x, y))
    return relays + sensors


def uniform_in_square(side_len: float, n_sensor: int, sink: Position) -> Distribution:
    sx, sy = sink
    sensors = [
        (sx + rand() * side_len, sy + rand() * side_len) for _ in range(n_sensor)
    ]
    return sensors
