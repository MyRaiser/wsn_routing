from collections import defaultdict

import matplotlib.pyplot as plt

from distribution import *
from router.jso_route import JSORouter
from router.leach import LEACH, LEACHPrim


def test_jso_route():
    sink = (0, 0)
    # distribution = power_line_naive(4, 375, 0, 0, 25, 40, sink)
    distribution = uniform_in_square(200, 100, sink, "mid")
    clusters = 6
    cases = {
        "JSO": JSORouter,
        "LEACH": LEACH,
        "LEACH-P": LEACHPrim
    }
    parameters = {
        "JSO": {
            "n_pop": 50,
            "iter_max": 50,
            "r_0": 60,
            "c": 0.4
        }
    }
    parameters = defaultdict(dict, **parameters)

    n_alive = defaultdict(list)
    e_mean = defaultdict(list)
    for case, method in cases.items():
        parameter = parameters[case]
        router = method(
            *simple_loader(sink, distribution),
            n_cluster=clusters, **parameter
        )
        router.initialize()
        n_alive_rt = []
        e_mean_rt = []
        n = None
        count = 0
        while n is None or n > 0:
            router.execute()
            count += 1
            n = len(router.alive_non_sinks)
            n_alive_rt.append(n)
            c = len(router.clusters)
            print(f"round = {count}, alive = {n}, clusters = {c}")
            e = np.mean([node.energy for node in router.alive_non_sinks])
            e_mean_rt.append(e)
            # router.plot()
        n_alive[case] = n_alive_rt
        e_mean[case] = e_mean_rt

    with plt.style.context(["science", "ieee", "grid"]):
        fig, ax = plt.subplots()
        for case in n_alive:
            ax.plot(n_alive[case], label=case)

        ax.legend(title="protocols")
        ax.set(xlabel="Round")
        ax.set(ylabel="Number of nodes alive")
        ax.autoscale()

    with plt.style.context(["science", "ieee", "grid"]):
        fig, ax = plt.subplots()
        for case in e_mean:
            ax.plot(e_mean[case], label=case)

        ax.legend(title="protocols")
        ax.set(xlabel="Round")
        ax.set(ylabel="Mean energy")
        ax.autoscale()

    plt.show()


if __name__ == "__main__":
    test_jso_route()
