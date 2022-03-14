from typing import Iterable
from itertools import combinations


import numpy as np

from .leach import LEACH
from .hierarchical import LEACHGreedy
from router.jso_route import JSOGreedy
from optimizer import optimize
from optimizer.optimizer import pso
from router.node import Node


def sigmoid(x: float) -> float:
    return 2 / (1 + np.exp(-x)) - 1


class LeachPSO(LEACHGreedy):
    def __init__(
            self,
            sink: Node,
            non_sinks: Iterable[Node],
            *,
            n_pop: int,
            iter_max: int,
            r_0: float,
            c: float,
            l1: float,
            **kwargs
    ):
        super().__init__(
            sink, non_sinks,
            **kwargs
        )
        self.l1 = l1
        self.r_0 = r_0
        self.c = c

        self.pso_parameters = {
            "n_pop": n_pop,
            "iter_max": iter_max
        }
        self.pso_target = None

        self.dist = np.array(
            [
                [self.distance(ni, nj) for nj in self.non_sinks] for ni in self.non_sinks
            ]
        )

        self.d_to_sink = np.array(
            [self.distance(self.sink, node) for node in self.non_sinks]
        )

    def contention_radius(self, d_max: float, d_min: float, d: float) -> float:
        if d_max == d_min:
            return self.r_0
        r = (1 - self.c * (d_max - d) / (d_max - d_min)) * self.r_0
        return r

    def get_pso_target(self, k: int, candidates: list[Node], energy: np.ndarray, e_mean):
        # idt is [ch_index] + [ch_route]
        # ch_index in [1, n) except sink node 0
        # ch_route in [0, k + 1) where k means route to sink
        n = len(candidates)
        if k <= 0:
            k = 1

        def f(indices: np.ndarray) -> float:
            """judge the selection of cluster heads"""
            # if len(heads) != len(set(heads)):
            #     return float("inf")
            e = np.mean([energy[i] for i in indices])
            # print(f"f - {1 / (e / e_mean)}")
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
            ret = ret / len(indices) ** 2
            # print(f"g = {sigmoid(ret)}")
            return sigmoid(ret)

        def func(idt: np.ndarray) -> float:
            # heads = self.get_heads_and_routes(candidates, idt)
            indices = np.array(list(set([int(i) for i in set(idt)])))
            # fitness = f(indices) + g(indices) * 2.5 * 1e-4
            fitness = self.l1 * f(indices) + (1 - self.l1) * g(indices)
            return fitness

        dim = k
        lb = np.array(
            [0] * k
        )
        ub = np.array(
            [n - 1] * k
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
        energy = np.array([node.energy for node in candidates])
        e_mean = np.mean(energy)
        k = int(len(candidates) * self.n_cluster / (len(self.nodes) - 1))
        self.pso_target = self.get_pso_target(k, candidates, energy, e_mean)
        opt, val = optimize(pso(self.pso_target, **self.pso_parameters))
        print(opt)
        print([int(i) for i in opt], val)

        heads = self.get_heads_and_routes(candidates, opt)
        for src in heads:
            if not self.is_cluster_head(src):
                self.add_cluster_head(src)
