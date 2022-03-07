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

    @staticmethod
    def get_jso_target(k: int, candidates: list[Node]):
        # idt is [ch_index] + [ch_route]
        # ch_index in [1, n) except sink node 0
        # ch_route in [0, k + 1) where k means route to sink
        n = len(candidates)
        energy = [node.energy for node in candidates]
        e_mean = np.mean(energy)
        # k = ceil(k / (len(self.nodes) - 1) * n)

        def f(idt: np.ndarray) -> float:
            """judge the selection of cluster heads"""
            # if len(heads) != len(set(heads)):
            #     return float("inf")
            indices = [int(i) for i in idt]
            e = np.mean([energy[i] for i in indices])
            return 1 / (e / e_mean)

        def func(idt: np.ndarray) -> float:
            # heads = self.get_heads_and_routes(candidates, idt)
            fitness = f(idt)
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
        while True:
            candidates = self.alive_non_sinks
            self.jso_target = self.get_jso_target(self.n_cluster, candidates)
            opt, val = optimize(jso(self.jso_target, **self.jso_parameters))
            if val != float("inf"):
                break
            else:
                print("invalid")
        # print(opt)
        # print([int(i) for i in opt], val)

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
