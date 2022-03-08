"""clustering routing based on JSO"""
from typing import Iterable
from itertools import combinations

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
            r_0: float,
            c: float,
            **kwargs
    ):
        super().__init__(
            sink, non_sinks,
            **kwargs
        )
        self.r_0 = r_0
        self.c = c

        self.jso_parameters = {
            "n_pop": n_pop,
            "iter_max": iter_max
        }
        self.jso_target = None

        self.dist = np.array(
            [
                [self.distance(ni, nj) for nj in self.non_sinks] for ni in self.non_sinks
            ]
        )

        self.d_to_sink = np.array(
            [self.distance(self.sink, node) for node in self.non_sinks]
        )

    def contention_radius(self, d_max: float, d_min: float, d: float) -> float:
        r = (1 - self.c * (d_max - d) / (d_max - d_min)) * self.r_0
        return r

    def get_jso_target(self, k: int, candidates: list[Node]):
        # idt is [ch_index] + [ch_route]
        # ch_index in [1, n) except sink node 0
        # ch_route in [0, k + 1) where k means route to sink
        n = len(candidates)
        energy = np.array([node.energy for node in candidates])
        e_mean = np.mean(energy)
        k = int(n * k / (len(self.nodes) - 1))
        if k <= 0:
            k = 1

        def f(indices: np.ndarray) -> float:
            """judge the selection of cluster heads"""
            # if len(heads) != len(set(heads)):
            #     return float("inf")
            e = np.mean([energy[i] for i in indices])
            return 1 / (e / e_mean)

        def g(indices: np.ndarray) -> float:
            ret = 0
            d = np.array([self.d_to_sink[i] for i in indices])
            d_max = np.max(d)
            d_min = np.min(d)
            for a, b in combinations((x for x in range(len(indices))), 2):
                i = indices[a]
                j = indices[b]
                ri = self.contention_radius(d_max, d_min, d[a])
                rj = self.contention_radius(d_max, d_min, d[b])
                tmp = self.distance(candidates[i], candidates[j]) - (ri + rj)
                ret += tmp ** 2
            return ret / len(indices) ** 2

        def func(idt: np.ndarray) -> float:
            # heads = self.get_heads_and_routes(candidates, idt)
            indices = np.array([int(i) for i in set(idt)])
            fitness = f(indices) + g(indices) * 1e-4
            return fitness

        dim = k
        lb = np.array(
            [0] * k
        )
        ub = np.array(
            [n] * k
        )
        return func, dim, lb, ub

    @staticmethod
    def get_heads_and_routes(
            candidates: list[Node], idt: np.ndarray
    ) -> list[Node]:
        ch_indices = [int(i) for i in idt]
        heads = [candidates[i] for i in ch_indices]
        return heads

    def cluster_head_select(self):
        """select cluster head and route"""
        self.clear_clusters()

        candidates = self.alive_non_sinks
        self.jso_target = self.get_jso_target(self.n_cluster, candidates)
        opt, val = optimize(jso(self.jso_target, **self.jso_parameters))
        # print(opt)
        print([int(i) for i in opt], val)

        heads = self.get_heads_and_routes(candidates, opt)
        for src in heads:
            self.add_cluster_head(src)
        self.cluster_head_organize()

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

    def steady_state_phase(self):
        self.cluster_run(self.sink)
