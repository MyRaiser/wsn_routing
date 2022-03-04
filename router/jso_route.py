"""clustering routing based on JSO"""
from typing import Iterable

import numpy as np

from .node import Node
from .leach import LEACH
from optimizer import optimize, jso


class JSORouter(LEACH):
    def __init__(
            self,
            sink: Node,
            non_sinks: Iterable[Node],
            *,
            n_cluster: int,
            size_control: int = 32,
            size_data: int = 4096,
            energy_agg: float = 5e-9,
            agg_rate: float = 0.6,
            n_pop: int,
            iter_max: int,
    ):
        super().__init__(
            sink, non_sinks,
            n_cluster=n_cluster, size_control=size_control, size_data=size_data,
            energy_agg=energy_agg, agg_rate=agg_rate
        )
        self.sink_cluster = set()
        self.jso_parameters = {
            "n_pop": n_pop,
            "iter_max": iter_max
        }
        self.jso_target = self.get_jso_target(n_cluster, len(self.nodes) - 1)

    def is_valid_route(self, sources: list[Node], destinations: list[Node]) -> bool:
        if self.sink not in destinations:
            return False
        route = {src: destinations[i] for i, src in enumerate(sources)}

        def search(src: Node, visited: set) -> bool:
            visited.add(src)

            dst = route[src]
            if dst == self.sink:
                ret = True
            elif dst in visited:
                ret = False
            else:
                ret = search(dst, visited)
            return ret
        # print("---")
        for src in sources:
            cache = set()
            valid = search(src, cache)
            # print(f"node {self.index(src)}: {valid}")
            if not valid:
                return False
        return True

    def get_jso_target(self, k: int, n: int):
        # idt is [ch_index] + [ch_route]
        # ch_index in [1, n) except sink node 0
        # ch_route in [0, k + 1) where k means route to sink

        def func(idt: np.ndarray) -> float:
            heads, routes = self.get_heads_and_routes(idt)
            if len(set(heads)) != len(heads):
                return float("inf")
            if not self.is_valid_route(heads, routes):
                return float("inf")
            s = 0
            for i, src in enumerate(heads):
                dst = routes[i]
                s += self.distance(src, dst)
            return s

        dim = k + k
        lb = np.array(
            [1] * k + [0] * k
        )
        ub = np.array(
            [n] * k + [k + 1] * k
        )
        return func, dim, lb, ub

    def get_heads_and_routes(self, idt: np.ndarray) -> tuple[list[Node], list[Node]]:
        k = len(idt) // 2
        ch_index = [int(i) for i in idt[:k]]
        ch_route = [int(i) for i in idt[k:]]
        heads = [self.node(i) for i in ch_index]
        routes = []
        for i, src_index in enumerate(ch_index):
            j = ch_route[i]
            if j == k:
                dst_index = 0
            else:
                dst_index = ch_index[j]
            routes.append(self.node(dst_index))
        return heads, routes

    def cluster_head_select(self):
        """select cluster head and route"""
        self.clusters = {}
        self.sink_cluster = set()
        while True:
            opt, val = optimize(jso(self.jso_target, **self.jso_parameters))
            if val != float("inf"):
                break
        # print(opt)
        print([int(i) for i in opt], val)
        heads, routes = self.get_heads_and_routes(opt)
        for i, src in enumerate(heads):
            dst = routes[i]
            self.clusters[src] = set()
            self.set_route(src, dst)

        for i, dst in enumerate(routes):
            src = heads[i]
            if dst == self.sink:
                self.sink_cluster.add(src)
            else:
                self.clusters[dst].add(src)

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
