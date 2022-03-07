from collections import defaultdict

import matplotlib.pyplot as plt

from distribution import *
from router.jso_route import JSORouter
from router.leach import LEACH, LEACHPrim


def test_jso_route():
    sink = (0, 0)
    # distribution = power_line_naive(4, 375, 0, 0, 25, 40, sink)
    distribution = uniform_in_square(100, 200, sink, "mid")
    clusters = 14
    cases = {
        "LEACH": LEACH,
        "LEACH-P": LEACHPrim,
        "JSO": JSORouter
    }
    parameters = {
        "JSO": {
            "n_pop": 50,
            "iter_max": 100
        }
    }
    parameters = defaultdict(dict, **parameters)

    n_alive = defaultdict(list)
    for case, method in cases.items():
        parameter = parameters[case]
        router = method(
            *simple_loader(sink, distribution),
            n_cluster=clusters, **parameter
        )
        router.initialize()
        n_alive_rt = []
        n = None
        count = 0
        while n is None or n > clusters:
            router.execute()
            count += 1
            n = len(router.alive_non_sinks)
            n_alive_rt.append(n)
            print(f"round = {count}, alive = {n}")
            # router.plot()
        n_alive[case] = n_alive_rt

    with plt.style.context(["science", "ieee", "grid"]):
        fig, ax = plt.subplots()
        for case in n_alive:
            ax.plot(n_alive[case], label=case)

        ax.legend(title="protocols")
        ax.set(xlabel="Round")
        ax.set(ylabel="Number of nodes alive")
        ax.autoscale()
        plt.show()


if __name__ == "__main__":
    test_jso_route()
