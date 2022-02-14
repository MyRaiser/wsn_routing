from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

from numpy.random import rand

from router.common import Router, Node, distance, argmin_


class LEACH(Router):
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
            distance(n1.position, n2.position) for n1, n2 in combinations(self.nodes, 2)
        )
        self.route_feature_op = lambda n: n in self.clusters

        # private properties
        self.round = 0
        self.clusters: dict[Node, set[Node]] = dict()
        # number of rounds not as a cluster head
        # G in the paper
        self.rounds_non_head: dict[Node, int] = defaultdict(int)

    def initialize(self):
        pass

    def execute(self):
        self.set_up_phase()
        self.steady_state_phase()

    def set_up_phase(self):
        # clustering until at least one cluster is generated.
        while True:
            self.cluster_head_select()
            if self.clusters:
                self.cluster_member_join()
                break

    def cluster_head_select(self):
        self.clusters = {}
        for node in self.alive_non_sinks:
            T = self.threshold(node)
            t = rand()
            if t < T:
                # be selected as cluster head
                # broadcast announcement
                node.broadcast(self.size_control, self.max_dist)
                if node.is_alive():
                    self.clusters[node] = set()
                    self.set_route(node, self.sink)
                    self.rounds_non_head[node] = 0
            else:
                self.rounds_non_head[node] += 1
        self.round += 1

    def cluster_member_join(self):
        """members join clusters"""
        for node in filter(lambda n: n not in self.clusters, self.alive_non_sinks):
            node.recv_broadcast(self.size_control * len(self.clusters))
            # select nearest cluster head to join
            nearest = argmin_(
                lambda x: distance(node.position, x.position), self.clusters
            )
            # send join-request
            node.singlecast(self.size_control, nearest)
            self.clusters[nearest].add(node)
            self.set_route(node, nearest)

    def steady_state_phase(self):
        if not self.clusters:
            return

        for head, members in self.clusters.items():
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