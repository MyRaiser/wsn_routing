from typing import Literal

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


RelativePosition = Literal[
    "mid", "left-bottem", "left-top", "right-bottem", "right-top",
    "left-mid"
]


def uniform_in_square(
        side_len: float,
        n_sensor: int,
        sink: Position,
        sink_relative_position: RelativePosition = "mid"
) -> Distribution:
    match sink_relative_position:
        case "left-bottem":
            dx, dy = 0, 0
        case "left-top":
            dx, dy = 0, -side_len
        case "right-bottem":
            dx, dy = -side_len, 0
        case "right-top":
            dx, dy = -side_len, -side_len
        case "mid":
            dx, dy = -side_len / 2, -side_len / 2
        case "left-mid":
            dx, dy = 0, -side_len / 2
        case _:
            raise Exception("Invalid relative position.")
    sx, sy = sink
    sx += dx
    sy += dy
    sensors = [
        (sx + rand() * side_len, sy + rand() * side_len) for _ in range(n_sensor)
    ]
    return sensors


def uniform_in_circle(
        radius: float,
        n_sensor: int,
        sink: Position,
) -> Distribution:
    sx, sy = sink
    sensors = []
    for _ in range(n_sensor):
        r = radius * rand()
        theta = 2 * pi * rand()
        x = sx + r * cos(theta)
        y = sy + r * sin(theta)
        sensors.append((x, y))
    return sensors
