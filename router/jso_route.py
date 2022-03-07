"""clustering routing based on JSO"""
from typing import Iterable

import numpy as np
from math import ceil

from .node import Node
from .leach import LEACHHierarchical
from optimizer import optimize, jso


class JSORouter(LEACHHierarchical):
    def __init__(
            self,
            sink: Node,
            non_sinks: Iterable[Node],
            *,
            n_pop: int,
            iter_max: int,
            **kwargs
    ):
        super().__init__(
            sink, non_sinks,
            **kwargs
        )
        self.sink_cluster = set()
        self.jso_parameters = {
            "n_pop": n_pop,
            "iter_max": iter_max
        }
        self.jso_target = None

    def get_jso_target(self, k: int, candidates: list[Node]):
        # idt is [ch_index] + [ch_route]
        # ch_index in [1, n) except sink node 0
        # ch_route in [0, k + 1) where k means route to sink
        n = len(candidates)
        # k = ceil(k / (len(self.nodes) - 1) * n)

        def f(heads: list[Node]) -> float:
            """judge the selection of cluster heads"""
            if len(heads) != len(set(heads)):
                return float("inf")
            e_mean = np.mean([node.energy for node in self.alive_non_sinks])
            e = np.mean([node.energy for node in heads])
            return 1 / (e / e_mean)

        def g(heads: list[Node], routes: dict[Node, Node]) -> float:
            visited: dict[Node, None | float] = dict()

            def discover(src: Node) -> float:
                if src in visited:
                    if visited[src] is None:
                        return float("inf")
                    else:
                        return visited[src]
                if src not in routes:  # end of route
                    if src == self.sink:
                        return 0
                    else:
                        return float("inf")
                visited[src] = None
                dst = routes[src]
                ret = self.distance(src, dst) + discover(dst)
                visited[src] = ret
                return ret

            dist = 0
            for head in heads:
                dist += discover(head)
            return dist

        def func(idt: np.ndarray) -> float:
            heads, routes = self.get_heads_and_routes(candidates, idt)
            fitness = f(heads) * 500 + g(heads, routes)
            return fitness

        # I think it's a total piece of shit
        dim = k + k
        lb = np.array(
            [1] * k + [0] * k
        )
        ub = np.array(
            [n] * k + [k + 1] * k
        )
        return func, dim, lb, ub

    def get_heads_and_routes(
            self, candidates: list[Node], idt: np.ndarray
    ) -> tuple[list[Node], dict[Node, Node]]:
        k = len(idt) // 2
        ch_indices = [int(i) for i in idt[:k]]
        ch_route_pointers = [int(i) for i in idt[k:]]
        heads = [candidates[i] for i in ch_indices]
        routes = {}
        for i, src_index in enumerate(ch_indices):
            src = candidates[src_index]
            j = ch_route_pointers[i]
            if j == k:
                dst = self.sink
            else:
                dst_index = ch_indices[j]
                dst = candidates[dst_index]
            routes[src] = dst
        return heads, routes

    def cluster_head_select(self):
        """select cluster head and route"""
        self.clear_clusters()
        while True:
            candidates = self.alive_non_sinks
            self.jso_target = self.get_jso_target(self.n_cluster, candidates)
            opt, val = optimize(jso(self.jso_target, **self.jso_parameters))
            if val != float("inf"):
                break
            else:
                print("invalid")
        # print(opt)
        print([int(i) for i in opt], val)

        heads, routes = self.get_heads_and_routes(candidates, opt)
        for src in heads:
            self.add_cluster_head(src)
        for src in heads:
            dst = routes[src]
            self.add_cluster_member(dst, src)

    def steady_state_phase(self):
        self.cluster_run(self.sink)
