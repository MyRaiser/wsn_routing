from __future__ import annotations  # to allow forward references in type hint
import numpy as np
from numpy import sqrt, pi, cos, sin
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from enum import Enum
from typing import Optional, Callable, Any
from collections.abc import Iterable
from abc import ABCMeta, abstractmethod


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
        e_tx = self.energy_tx(size, dist)
        e_rx = target.energy_rx(size)
        self.energy -= e_tx
        target.energy -= e_rx

    def energy_tx(self, size: int, dist: float) -> float:
        if dist <= self.dist_threshold:
            # free space
            return size * self.e0_tx + size * self.epsilon_fs * (dist ** 2)
        else:
            # multi-path
            return size * self.e0_tx + size * self.epsilon_mp * (dist ** 4)

    def energy_rx(self, size: int) -> float:
        return size * self.e0_rx


class Router(metaclass=ABCMeta):
    """Simple, round-based simulation of routing"""
    default_node_styles = ["rs", "gs", "bo"]
    default_route_styles = [("-", "r"), ("-", "k")]

    def __init__(
            self,
            sink: Node,
            non_sinks: Iterable[Node]
    ):
        # not supporting dynamic change of nodes
        self.sink = sink
        self.non_sinks = tuple(non_sinks)
        self.nodes = tuple([sink, *non_sinks])

        # assign index to each node
        self.__nodes_to_indices = {node: i for i, node in enumerate(self.nodes)}

        # use a vector to represent route
        self.route = np.array(
            [0 for _ in range(len(self.nodes))]
        )

        # plotting
        # only support 2-d plot

        # cache for styles
        self.__node_styles: dict[Any, str] = dict()
        self.__route_styles: dict[Any, tuple[str, str]] = dict()
        # function to get feature of a node, which distinguish it from other nodes.
        self.node_feature_op = lambda n: n.category
        self.route_feature_op = lambda n: n.category

        positions = [node.position for node in self.nodes]
        self.position_max = np.max(positions, axis=0)
        self.position_min = np.min(positions, axis=0)

        self.plotter = Plotter2D(
            self.position_min[0],
            self.position_min[1],
            self.position_max[0],
            self.position_max[1],
            max(
                self.position_max[axis] - self.position_min[axis] for axis in (0, 1)
            ) / 10
        )

    def node(self, i: int) -> Node:
        return self.nodes[i]

    def index(self, node: Node) -> int:
        return self.__nodes_to_indices[node]

    def set_route(self, src: Node, dst: Node):
        self.route[self.index(src)] = self.index(dst)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def execute(self):
        """execute for one round"""
        pass

    @staticmethod
    def __set_style(cache: dict, *styles: tuple[Any, Any]):
        for feature, style in styles:
            cache[feature] = style

    def set_node_style(self, *styles: tuple[Any, str]):
        self.__set_style(self.__node_styles, *styles)

    def set_route_style(self, *styles: tuple[Any, tuple[str, str]]):
        self.__set_style(self.__route_styles, *styles)

    @staticmethod
    def __get_style(
            cache: dict,
            feature_op: Callable[[Node], Any],
            default_styles: list[Any],
            exception_style: Any,
            node: Node,
    ):
        feature = feature_op(node)
        if feature in cache:
            return cache[feature]
        # automatically allocate a style in default styles, which has not been used
        for style in default_styles:
            if style not in cache.values():
                cache[feature] = style
                return style
        # if style is not specified, or default styles have been used up.
        cache[feature] = exception_style
        return exception_style

    def get_node_style(self, node: Node) -> str:
        return self.__get_style(
            self.__node_styles,
            self.node_feature_op,
            self.default_node_styles,
            "",
            node
        )

    def get_route_style(self, src: Node) -> tuple[str, str]:
        return self.__get_style(
            self.__route_styles,
            self.route_feature_op,
            self.default_route_styles,
            ("-", "k"),
            src
        )

    def plot_nodes(self):
        for node in self.nodes:
            style = self.get_node_style(node)
            self.plotter.plot_point(node.position, style)

    def plot_routes(self):
        for i, src in enumerate(self.nodes):
            dst = self.node(self.route[i])
            ls, c = self.get_route_style(src)
            self.plotter.plot_line(src.position, dst.position, linestyle=ls, color=c)

    def show(self):
        self.plotter.show()

    def plot(self):
        self.plot_nodes()
        self.plot_routes()
        self.show()


class Plotter2D:
    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float, margin: float):
        self.fig, self.ax = plt.subplots()
        self.plt = plt
        # self.set_font()
        self.set_bound(min_x - margin, min_y - margin, max_x + margin, max_y + margin)

    def plot_line(self, src, dst, **kwargs):
        self.ax.add_line(
            Line2D(
                (src[0], dst[0]), (src[1], dst[1]), linewidth=1, **kwargs
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


class SimpleRouter(Router):
    def initialize(self):
        sensors = []
        relays = []
        for node in self.non_sinks:
            match node.category:
                case NodeCategory.sensor:
                    sensors.append(node)
                case NodeCategory.relay:
                    relays.append(node)
                case _:
                    raise Exception("Unexpected node category.")

        # sensor nodes link to nearst relay
        for node in sensors:
            nearest_relay = argmin_(
                lambda x: distance(node.position, x.position),
                relays
            )
            self.route[self.index(node)] = self.index(nearest_relay)

        # relay nodes form a line
        node = self.sink
        free_relay = set(relays)
        while len(free_relay) > 0:
            nearest = argmin_(
                lambda x: distance(node.position, x.position),
                free_relay
            )
            self.route[self.index(nearest)] = self.index(node)
            node = nearest
            free_relay.remove(nearest)

    def execute(self):
        pass


class LEACH(Router):
    def __init__(
            self,
            sink: Node,
            non_sinks: Iterable[Node],
            *,
            n_cluster: int
    ):
        super().__init__(sink, non_sinks)
        self.n_cluster = n_cluster
        self.round = 0
        self.clusters: dict[Node, set[Node]] = dict()
        self.route_feature_op = lambda n: n in self.clusters

    def threshold(self, node: Node):
        if node in self.clusters:
            # if is a cluster head of last round
            return 0
        else:
            p = self.n_cluster / len(self.non_sinks)
            r = self.round
            return p / (1 - p * (r % int(1 / p)))

    def set_up_phase(self):
        # cluster head selection
        new_clusters = dict()
        non_heads = set()
        for node in filter(lambda n: n.energy > 0, self.non_sinks):
            T = self.threshold(node)
            t = rand()
            if t < T:
                # be selected as cluster head
                new_clusters[node] = set()
                self.set_route(node, self.sink)
                # broadcast announcement
            else:
                non_heads.add(node)

        if new_clusters:
            for node in non_heads:
                nearest = argmin_(
                    lambda x: distance(node.position, x.position), new_clusters
                )
                # send join-request
                new_clusters[nearest].add(node)
                self.set_route(node, nearest)

        self.clusters = new_clusters

    def steady_state_phase(self):
        pass

    def initialize(self):
        pass

    def execute(self):
        self.set_up_phase()
        self.steady_state_phase()


class APTEEN(Router):
    def initialize(self):
        pass

    def execute(self):
        pass


def test_leach():
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

    router = LEACH(sink, relays + sensors, n_cluster=n_relay)
    router.initialize()
    router.execute()
    print(f"cluster heads: {len(router.clusters)}")
    print(router.route)
    router.plot()
    print("")


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

    router = SimpleRouter(sink, relays + sensors)
    router.initialize()
    print(router.route)
    router.plot()
    print("")


if __name__ == "__main__":
    test_leach()
