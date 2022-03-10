from typing import Iterable
from collections import defaultdict
from itertools import combinations

from numpy.random import rand

from router import Router, Node
from .common import ClusterBased


class LEACH(Router, ClusterBased):
    """original LEACH"""
    def __init__(
            self,
            sink: Node,
            non_sinks: Iterable[Node],
            *,
            n_cluster: int,
            size_control: int = 32,
            size_data: int = 4096,
            energy_agg: float = 5e-9,
            agg_rate: float = 0.6
    ):
        super().__init__(sink, non_sinks)
        # super parameters
        self.n_cluster = n_cluster
        self.size_control = size_control  # size of different messages
        self.size_data = size_data
        self.energy_agg = energy_agg  # energy of aggregation per bit
        self.agg_rate = agg_rate

        # constants
        # max distance between two nodes
        self.max_dist = max(
            self.distance(n1, n2) for n1, n2 in combinations(self.nodes, 2)
        )
        self.route_feature_op = lambda n: n in self.clusters

        # private properties
        self.round = 0
        self.clusters: dict[Node, set[Node]] = dict()
        # number of rounds not as a cluster head
        # G in the paper
        self.rounds_non_head: dict[Node, int] = defaultdict(int)

    def clear_clusters(self):
        self.clusters = dict()
        self.clear_route()

    def add_cluster_head(self, head: Node):
        assert head not in self.clusters
        self.clusters[head] = set()
        self.set_route(head, self.sink)

    def add_cluster_member(self, head: Node, node: Node):
        assert head in self.clusters
        self.clusters[head].add(node)
        self.set_route(node, head)

    def get_cluster_heads(self):
        return iter(self.clusters)

    def get_cluster_members(self, head: Node):
        return self.clusters[head]

    def initialize(self):
        pass

    def execute(self):
        self.set_up_phase()
        self.steady_state_phase()
        self.round += 1

    def set_up_phase(self):
        # clustering until at least one cluster is generated.
        while len(self.alive_non_sinks) > 0:
            self.cluster_head_select()
            if self.clusters:
                self.cluster_member_join()
                break

    def cluster_head_select(self):
        self.clear_clusters()
        for node in self.alive_non_sinks:
            T = self.threshold(node)
            t = rand()
            if t < T:
                # be selected as cluster head
                # broadcast announcement
                node.broadcast(self.size_control, self.max_dist)
                if node.is_alive():
                    self.add_cluster_head(node)
                    self.rounds_non_head[node] = 0
            else:
                self.rounds_non_head[node] += 1

    def cluster_member_join(self):
        """members join clusters"""
        for node in filter(lambda n: n not in self.clusters, self.alive_non_sinks):
            node.recv_broadcast(self.size_control * len(self.clusters))
            # select nearest cluster head to join
            nearest = min(
                self.get_cluster_heads(),
                key=lambda x: self.distance(node, x),
            )
            # send join-request
            node.singlecast(self.size_control, nearest)
            self.add_cluster_member(nearest, node)

    def steady_state_phase(self):
        if not self.clusters:
            return

        for head in self.get_cluster_heads():
            members = self.get_cluster_members(head)
            if not members:
                head.singlecast(self.size_data, self.sink)
            else:
                size_total = self.size_data
                for node in members:
                    # cluster members send data
                    node.singlecast(self.size_data, head)
                    size_total += self.size_data
                # data aggregation
                head.singlecast(self.aggregation(head, size_total), self.sink)

    @property
    def non_head_protection(self) -> int:
        p = self.n_cluster / len(self.non_sinks)
        r = self.round
        return r % int(1 / p)

    def threshold(self, node: Node):
        if self.rounds_non_head[node] < self.non_head_protection:
            return 0
        else:
            p = self.n_cluster / len(self.non_sinks)
            r = self.round
            return p / (1 - p * (r % int(1 / p)))

    def aggregation(self, node: Node, size: int) -> int:
        node.energy -= size * self.energy_agg
        return int(self.agg_rate * size)

    def is_cluster_head(self, node: Node) -> bool:
        return node in self.clusters
