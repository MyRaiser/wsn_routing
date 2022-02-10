from __future__ import annotations  # to allow forward references in type hint
import numpy as np
from numpy import sqrt, pi, cos, sin
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from enum import Enum
from typing import Optional, Callable, Any
from collections.abc import Iterable


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


class NodeCategory(Enum):
    sink = "sink"
    relay = "relay"
    sensor = "sensor"


class Node:
    # constants
    e0_tx = 1
    e0_rx = 1

    epsilon_fs = 1  # amplifier coefficient in free space model
    epsilon_mp = 1  # amplifier coefficient in multi-path model
    dist_threshold = sqrt(epsilon_fs / epsilon_mp)

    energy_max = 100

    SIZE_CONTROL = 32
    SIZE_DATA = 1000

    def __init__(self, position: np.array, category: NodeCategory):
        self.position = position
        assert isinstance(category, NodeCategory)
        self.category = category
        self.energy = Node.energy_max

    def transmit(self, size: int, target: Node):
        """transmission happens once"""
        dist = distance(self.position, target.position)
        tx = self.energy_tx(size, dist)
        rx = target.energy_rx(size)
        self.energy -= tx
        target.energy -= rx

    def energy_tx(self, size: int, dist: float) -> float:
        if dist <= self.dist_threshold:
            # free space
            return size * self.e0_tx + size * self.epsilon_fs * (dist ** 2)
        else:
            # multi-path
            return size * self.e0_tx + size * self.epsilon_mp * (dist ** 4)

    def energy_rx(self, size: int) -> float:
        return size * self.e0_rx


class Router:
    """Simple, round-based simulation of routing"""

    def __init__(self, nodes: Iterable[Node]):
        # not supporting dynamic change of nodes
        self.sensor_nodes = []
        self.relay_nodes = []
        self.sink_node = None
        self.position_max = np.max([node.position for node in nodes], axis=0)
        self.position_min = np.min([node.position for node in nodes], axis=0)

        # node classification
        for node in nodes:
            match node.category:
                case NodeCategory.sensor:
                    self.sensor_nodes.append(node)
                case NodeCategory.relay:
                    self.relay_nodes.append(node)
                case NodeCategory.sink:
                    if not self.sink_node:
                        self.sink_node = node
                    else:
                        raise Exception("Duplicated sink node.")
                case _:
                    raise Exception("Invalid node category.")
        if not self.sink_node:
            raise Exception("Sink node does not exist.")

        self.nodes = tuple([self.sink_node] + self.relay_nodes + self.sensor_nodes)
        self.indices = {node: i for i, node in enumerate(self.nodes)}

        # use a vector to represent route
        self.route = np.array([0] * len(self.nodes))

        # only support 2-d plot
        self.plotter = Plotter(
            self.position_min[0],
            self.position_min[1],
            self.position_max[0],
            self.position_max[1],
            max(
                self.position_max[axis] - self.position_min[axis] for axis in (0, 1)
            ) / (len(self.relay_nodes) + 2)
        )

    def initialize_topology(self):
        # sensor nodes link to nearst relay
        for node in self.sensor_nodes:
            nearest_relay = argmin_(
                lambda x: distance(node.position, x.position),
                self.relay_nodes
            )
            self.route[self.indices[node]] = self.indices[nearest_relay]

        # relay nodes form a line
        node = self.sink_node
        free_relay = set(self.relay_nodes)
        while len(free_relay) > 0:
            nearest = argmin_(
                lambda x: distance(node.position, x.position),
                free_relay
            )
            self.route[self.indices[nearest]] = self.indices[node]
            node = nearest
            free_relay.remove(nearest)

    def unpowered_nodes(self):
        for node in self.sensor_nodes:
            yield node
        for node in self.relay_nodes:
            yield node

    def plot(self):
        # plot nodes
        fmt = {
            NodeCategory.sink: "rs",
            NodeCategory.relay: "gs",
            NodeCategory.sensor: "bo"
        }
        for node in self.nodes:
            self.plotter.plot_point(node.position, fmt[node.category])

        # plot routes
        for i, src in enumerate(self.nodes):
            dst = self.nodes[self.route[i]]
            self.plotter.plot_line(src.position, dst.position)
        self.plotter.show()


class Plotter:
    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float, margin: float):
        self.fig, self.ax = plt.subplots()
        self.plt = plt
        # self.set_font()
        self.set_bound(min_x - margin, min_y - margin, max_x + margin, max_y + margin)

    def plot_line(self, src, dst, color: Optional[str] = None):
        self.ax.add_line(
            Line2D(
                (src[0], dst[0]), (src[1], dst[1]), linewidth=1, color=color
            )
        )

    def plot_point(self, position, fmt: Optional[str] = None):
        self.ax.plot(position[0], position[1], fmt)

    def set_bound(self, min_x: float, min_y: float, max_x: float, max_y: float, ):
        self.ax.axis([min_x, max_x, min_y, max_y])

    def plot_text(self, position, text, size=8):
        self.ax.text(position[0], position[1], text, fontsize=size)

    def set_font(self, ft_style='SimHei'):
        self.plt.rcParams['font.sans-serif'] = [ft_style]  # 用来正常显示中文标签

    @staticmethod
    def show():
        plt.show()


def main():
    sink = Node(np.array([0, 0]), NodeCategory.sink)
    interval = 100
    d_max = 10
    phi_max = pi / 10
    n_relay = 5
    relays = []
    x = 0
    y = 0
    for _ in range(n_relay):
        d = d_max * rand()
        phi = phi_max * rand() - phi_max / 2
        r = interval + d
        x += r * cos(phi)
        y += r * sin(phi)
        relays.append(
            Node(np.array([x, y]), NodeCategory.relay)
        )

    sensors = []
    n_sensor_per_relay = 10
    r_max = 80
    for relay in relays:
        for _ in range(n_sensor_per_relay):
            r = r_max * rand()
            theta = 2 * pi * rand()
            x = relay.position[0] + r * cos(theta)
            y = relay.position[1] + r * sin(theta)
            sensors.append(
                Node(np.array([x, y]), NodeCategory.sensor)
            )

    router = Router([sink] + relays + sensors)
    router.initialize_topology()
    print(router.route)
    router.plot()
    print("")


if __name__ == "__main__":
    main()
