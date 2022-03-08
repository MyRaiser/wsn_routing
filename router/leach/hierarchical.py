from abc import ABCMeta, abstractmethod
from typing import Iterable
from itertools import chain, combinations_with_replacement

from router import Node
from .leach import LEACH


class HierarchicalLEACH(LEACH, metaclass=ABCMeta):
    """
    use multi-hop in intra-cluster transmission  and single-hop in inter-cluster transmission.
    allow sink as a cluster head.
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

    def clear_clusters(self):
        self.clusters = dict()
        self.sink_cluster = set()
        self.clear_route()

    def add_cluster_head(self, head: Node):
        if (head != self.sink) and (head not in self.clusters):
            self.clusters[head] = set()
        # warning: route of cluster head is undetermined!

    def add_cluster_member(self, head: Node, node: Node):
        assert head in self.clusters or head == self.sink
        if head == self.sink:
            self.sink_cluster.add(node)
        else:
            self.clusters[head].add(node)
        self.set_route(node, head)

    def get_cluster_heads(self):
        return chain(iter(self.clusters), [self.sink])

    def get_cluster_members(self, head: Node):
        assert head in self.clusters or head == self.sink
        if head == self.sink:
            return self.sink_cluster
        else:
            return self.clusters[head]

    def set_up_phase(self):
        while len(self.alive_non_sinks) > 0:
            # clustering until at least one cluster is generated.
            self.cluster_head_select()
            if self.clusters:
                self.cluster_head_routing()
                self.cluster_member_join()
                break

    @abstractmethod
    def cluster_head_routing(self):
        """implement this to show how the cluster heads route."""
        pass

    def steady_state_phase(self):
        self.cluster_run(self.sink)

    def cluster_run(self, head: Node) -> int:
        members = self.get_cluster_members(head)
        size_agg = 0
        size_not_agg = self.size_data
        for member in members:
            if self.is_cluster_head(member):
                size_sub = self.cluster_run(member)
                member.singlecast(size_sub, head)
                size_agg += size_sub
                # size_not_agg += size_sub
            else:
                # cluster member send to head
                member.singlecast(self.size_data, head)
                size_not_agg += self.size_data
        return self.aggregation(head, size_not_agg) + size_agg


class LEACHPrim(HierarchicalLEACH):
    def cluster_head_routing(self):
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


class LEACHGreedy(HierarchicalLEACH):
    def route_cost(self, src: Node, dst: Node) -> float:
        d = self.distance(src, dst)
        d_sink = self.distance(src, self.sink)
        e_dst = dst.energy
        cost = (d ** 2 + d_sink ** 2) / e_dst
        return cost

    def cluster_head_routing(self):
        """use greedy """
        candidates = set(list(self.clusters.keys()))

        while candidates:
            min_src, min_dst = min(
                [(src, dst) for src, dst in combinations_with_replacement(candidates, 2)],
                key=lambda route: self.route_cost(*route)
            )
            if min_src == min_dst:
                self.add_cluster_member(self.sink, min_src)
            else:
                self.add_cluster_member(min_dst, min_src)
            candidates.remove(min_src)
