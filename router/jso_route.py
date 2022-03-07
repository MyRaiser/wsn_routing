"""clustering routing based on JSO"""
from typing import Iterable

import numpy as np

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

    def is_valid_route(self, sources: list[Node], routes: dict[Node, Node]) -> bool:
        if self.sink not in (routes[src] for src in sources):
            return False

        def search(src: Node, visited: set) -> bool:
            visited.add(src)

            dst = routes[src]
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

    def get_jso_target(self, k: int, candidates: list[Node]):
        # idt is [ch_index] + [ch_route]
        # ch_index in [1, n) except sink node 0
        # ch_route in [0, k + 1) where k means route to sink
        n = len(candidates)

        def f(heads: list[Node]) -> float:
            """judge the selection of cluster heads"""
            e_mean = np.mean([node.energy for node in self.alive_non_sinks])
            e = np.mean([node.energy for node in heads])
            return e / e_mean

        def g(heads, routes: dict[Node, Node]) -> float:
            pass

        def func(idt: np.ndarray) -> float:
            heads, routes = self.get_heads_and_routes(candidates, idt)
            if len(set(heads)) != len(heads):
                return float("inf")
            if not self.is_valid_route(heads, routes):
                return float("inf")
            fitness = f(heads)
            for src in heads:
                dst = routes[src]
                fitness += self.distance(src, dst)
            return fitness

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
        self.clusters = {}
        self.sink_cluster = set()
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
            dst = routes[src]
            self.add_cluster_head(dst)
            self.add_cluster_member(dst, src)

    def steady_state_phase(self):
        self.cluster_run(self.sink)
