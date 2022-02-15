from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

from numpy.random import rand

from router.router import Router
from router.common import distance, argmin_
from router import Node


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
        while len(self.alive_non_sinks) > 0:
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
    
    def is_cluster_head(self, node: Node) -> bool:
        return node in self.clusters


class LEACHPrim(LEACH):
    """
    allow multi-hop in intra-cluster transmission
    and single-hop in inter-cluster transmission
    Using Prim algorithm to build route tree of cluster heads
    """
    def __init__(
            self,
            sink: Node,
            non_sinks: Iterable[Node],
            **kwargs
    ):
        super().__init__(
            sink, non_sinks, **kwargs)
        self.sink_cluster = set()  # view sink as a special cluster head

    def set_up_phase(self):
        while True:
            self.cluster_head_select()
            if self.clusters:
                self.cluster_head_organize()
                self.cluster_member_join()
                break

    def cluster_head_organize(self):
        """use Prim algorithm to form a tree of cluster heads"""
        visited = set()
        visited.add(self.sink)
        candidates = set(list(self.clusters.keys()))
        self.sink_cluster = set()

        while candidates:
            min_src, min_dst = min(
                [
                    min(
                        [(src, dst) for src in candidates],
                        key=lambda x: self.distance(x[0], x[1]),
                    ) for dst in visited
                ],
                key=lambda x: self.distance(x[0], x[1]),
            )
            visited.add(min_src)
            candidates.remove(min_src)
            self.set_route(min_src, min_dst)
            if min_dst == self.sink:
                self.sink_cluster.add(min_src)
            else:
                self.clusters[min_dst].add(min_src)

        # message exchange
        # organization of heads is done by sink
        for head in self.clusters:
            head.singlecast(self.size_control, self.sink)
            head.recv_broadcast(self.size_control)

    def steady_state_phase(self):
        self.cluster_run(self.sink)

    def cluster_run(self, head: Node) -> int:
        assert self.is_cluster_head(head) or head == self.sink

        if head == self.sink:
            members = self.sink_cluster
        else:
            members = self.clusters[head]
        size_agg = 0
        size_not_agg = self.size_data
        for member in members:
            if self.is_cluster_head(member):
                size_sub = self.cluster_run(member)
                member.singlecast(size_sub, head)
                size_agg += size_sub
            else:
                # cluster member send to head
                member.singlecast(self.size_data, head)
                size_not_agg += self.size_data
        return self.aggregation(head, size_not_agg) + size_agg
