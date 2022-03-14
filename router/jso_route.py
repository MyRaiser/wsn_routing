"""clustering routing based on JSO"""
from typing import Iterable
from itertools import combinations

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common.discretization import Q_discrete_white_noise

from .node import Node, dist_threshold
from .leach import LEACHGreedy
from .routing import prim
from optimizer import optimize, jso


def sigmoid(x: float) -> float:
    return 2 / (1 + np.exp(-x)) - 1


class JSOGreedy(LEACHGreedy):
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
        if d_max == d_min:
            return self.r_0
        r = (1 - self.c * (d_max - d) / (d_max - d_min)) * self.r_0
        return r

    def get_jso_target(self, k: int, candidates: list[Node], energy: np.ndarray, e_mean):
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
            return sigmoid(ret)

        def func(idt: np.ndarray) -> float:
            # heads = self.get_heads_and_routes(candidates, idt)
            indices = np.array([int(i) for i in set(idt)])
            # fitness = f(indices) + g(indices) * 2.5 * 1e-4
            fitness = self.l1 * f(indices) + (1 - self.l1) * g(indices)
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
        energy = np.array([node.energy for node in candidates])
        e_mean = np.mean(energy)
        k = int(len(candidates) * self.n_cluster / (len(self.nodes) - 1))
        self.jso_target = self.get_jso_target(k, candidates, energy, e_mean)
        opt, val = optimize(jso(self.jso_target, **self.jso_parameters))
        # print(opt)
        print([int(i) for i in opt], val)

        heads = self.get_heads_and_routes(candidates, opt)
        for src in heads:
            self.add_cluster_head(src)


class JSOPrim(JSOGreedy):
    def cluster_head_routing(self):
        routes = prim(lambda n1, n2: self.distance(n1, n2), self.get_cluster_heads(), self.sink)
        for src, dst in routes:
            self.add_cluster_member(dst, src)

        # message exchange
        # organization of heads is done by sink
        for head in self.get_cluster_heads():
            head.singlecast(self.size_control, self.sink)
            head.recv_broadcast(self.size_control)


class JSOKalman(JSOGreedy):
    def __init__(
            self,
            sink: Node,
            non_sinks: Iterable[Node],
            *,
            kalman_period: int = 10,
            kalman_warm_up: int = 2,
            kalman_p: float = 1000,
            kalman_r: float = 0.1,
            **kwargs,
    ):
        super().__init__(sink, non_sinks, **kwargs)
        self.kalman_period = kalman_period
        self.kalman_warm_up = kalman_warm_up
        self.kalman_used_heads = set()
        self.kalman = KalmanFilter(dim_x=2, dim_z=1)
        self.kalman.F = np.array([
            [1, 1],
            [0, 1],
        ])
        self.kalman.H = np.array([[1, 0]])
        self.kalman.P += kalman_p
        self.kalman.R = kalman_r
        self.kalman.Q = Q_discrete_white_noise(2, 1, .1)
        self.energy_cached = None  # update every kalman period
        self.e_mean_cached = None
        self.energy_estimated = None  # update every round

    def execute(self):
        self.update_energy_estimated()
        super().execute()

    def update_energy_estimated(self):
        if self.round == 0:
            self.energy_cached = {node: node.energy for node in self.non_sinks}
            self.energy_estimated = self.energy_cached

        # kalman estimation
        r = self.round - 1
        k, rem = divmod(r, self.kalman_period)
        z = np.mean([node.energy for node in self.non_sinks])  # get observation value
        if r == 0:  # kalman initialization
            z0 = np.mean(
                [self.energy_cached[node] for node in self.energy_cached]
            )
            self.e_mean_cached = z0
            self.kalman.x = np.array([
                [z],
                [(z - z0) * self.kalman_period]
            ])
            self.kalman.predict()
        else:
            if rem == 0:  # update kalman prediction
                self.e_mean_cached = self.kalman.x[0][0]
                self.kalman.update(z)
                self.kalman.predict()
        t = self.kalman.x[1][0] / self.kalman_period
        print(t)

        # update estimated value (every round)
        if k < self.kalman_warm_up:
            # use actual value
            self.energy_estimated = {node: node.energy for node in self.non_sinks}
        else:
            # use linear estimation
            self.energy_estimated = {
                node: self.energy_estimate(self.energy_cached[node], t, rem) for node in self.non_sinks
            }
        # update cached value (every kalman period)
        if rem == 0:
            self.kalman_used_heads = set()
            self.energy_cached = {node: node.energy for node in self.non_sinks}

    @staticmethod
    def energy_estimate(e0: float, t: float, k: int) -> float:
        return e0 + t * k

    def cluster_head_select(self):
        """select cluster head and route"""
        self.clear_clusters()

        # candidates = list(filter(
        #     lambda n: n not in self.kalman_used_heads,
        #     self.alive_non_sinks
        # ))
        # if not candidates:
        #     candidates = list(self.alive_non_sinks)

        candidates = list(self.alive_non_sinks)

        energy = np.array([
            self.energy_estimated[node] for node in candidates
        ])
        e_mean = self.kalman.x[0][0]
        k = int(len(candidates) * self.n_cluster / (len(self.nodes) - 1))
        self.jso_target = self.get_jso_target(k, candidates, energy, e_mean)
        opt, val = optimize(jso(self.jso_target, **self.jso_parameters))
        # print(opt)
        print([int(i) for i in opt], val)

        heads = self.get_heads_and_routes(candidates, opt)
        for src in heads:
            self.kalman_used_heads.add(src)
            self.add_cluster_head(src)

    @staticmethod
    def dist_cost(d0: float, d):
        if d <= d0:
            return d ** 2
        else:
            return d ** 4

    def route_cost(self, src: Node, dst: Node) -> float:
        d = self.distance(src, dst)
        d_sink = self.distance(src, self.sink)
        e_dst = self.energy_estimated[dst]
        cost = (self.dist_cost(dist_threshold, d) + self.dist_cost(dist_threshold, d_sink)) / e_dst
        return cost
