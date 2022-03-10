from collections import defaultdict

import matplotlib.pyplot as plt

from distribution import *
from router.jso_route import *
from router.leach import LEACH, LEACHPrim


def test_jso_route():
    sink = (0, 0)
    # distribution = power_line_naive(4, 375, 0, 0, 25, 40, sink)
    distribution = uniform_in_square(200, 100, sink, "mid")
    clusters = 6
    instances = {
        "JSO-K": JSOKalman,
        # "JSO-P": JSOPrim,
        "JSO-G": JSOGreedy,
        "LEACH": LEACH,
        "LEACH-P": LEACHPrim
    }
    parameters = {
        "JSO-K": {
            "n_pop": 50,
            "iter_max": 50,
            "r_0": 60,
            "c": 0.4
        },
        "JSO-P": {
            "n_pop": 50,
            "iter_max": 50,
            "r_0": 60,
            "c": 0.4
        },
        "JSO-G": {
            "n_pop": 50,
            "iter_max": 50,
            "r_0": 60,
            "c": 0.4
        }
    }
    parameters = defaultdict(dict, **parameters)

    n_alive = defaultdict(list)
    e_mean = defaultdict(list)
    e_var = defaultdict(list)
    for case, method in instances.items():
        parameter = parameters[case]
        router = method(
            *simple_loader(sink, distribution),
            n_cluster=clusters, **parameter
        )
        router.initialize()
        n_alive_case = []
        e_mean_case = []
        e_var_case = []
        n = None
        count = 0
        while n is None or n > 0:
            router.execute()
            count += 1
            n = len(router.alive_non_sinks)
            n_alive_case.append(n)
            c = len(router.clusters)
            print(f"round = {count}, alive = {n}, clusters = {c}")
            energy = [node.energy for node in router.non_sinks]
            e_mean_case.append(np.mean(energy))
            e_var_case.append(np.std(energy))
            # router.plot()
        n_alive[case] = n_alive_case
        e_mean[case] = e_mean_case
        e_var[case] = e_var_case

    with plt.style.context(["science", "ieee", "grid", "no-latex", "cjk-sc-font"]):
        fig, ax = plt.subplots()
        for case in n_alive:
            ax.plot(n_alive[case], label=case)
            print(n_alive[case])

        ax.legend()
        ax.set(xlabel="轮数")
        ax.set(ylabel="节点存活数量")
        ax.autoscale(tight=True)
        fig.savefig("alive_nodes.png", dpi=300)

    with plt.style.context(["science", "ieee", "grid", "no-latex", "cjk-sc-font"]):
        fig, ax = plt.subplots()
        for case in e_mean:
            ax.plot(e_mean[case], label=case)
            print()

        ax.legend()
        ax.set(xlabel="轮数")
        ax.set(ylabel="平均节点能量")
        ax.autoscale(tight=True)
        fig.savefig("energy_mean.png", dpi=300)

    with plt.style.context(["science", "ieee", "grid", "no-latex", "cjk-sc-font"]):
        fig, ax = plt.subplots()
        for case in e_var:
            ax.plot(e_var[case], label=case)

        ax.legend()
        ax.set(xlabel="轮数")
        ax.set(ylabel="节点能量方差")
        ax.autoscale(tight=True)
        fig.savefig("energy_standard_deviation.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    test_jso_route()
