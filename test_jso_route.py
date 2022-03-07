import matplotlib.pyplot as plt

from distribution import *
from router.jso_route import JSORouter
from router.leach import LEACH, LEACHPrim


def test_jso_route():
    sink = (0, 0)
    # distribution = power_line_naive(4, 375, 0, 0, 25, 40, sink)
    distribution = uniform_in_square(200, 100, sink, "left-mid")
    c = 7

    leach = LEACHPrim(
        *simple_loader(sink, distribution),
        n_cluster=c
    )
    leach.initialize()
    n_alive_lc = []
    while len(leach.alive_non_sinks) > c:
        leach.execute()
        n = len(leach.alive_non_sinks)
        n_alive_lc.append(n)
        # if n < 20:
        #     leach.plot()

    jr = JSORouter(
        *simple_loader(sink, distribution),
        n_cluster=c,
        n_pop=200, iter_max=200
    )
    jr.initialize()
    n_alive_jr = []
    while len(jr.alive_non_sinks) > c:
        jr.execute()
        n = len(jr.alive_non_sinks)
        n_alive_jr.append(n)
        # print(f"cluster heads: {len(jr.clusters)}")
        print(f"nodes alive: {n}")
        # print(leach.route)
        # jr.plot()

    with plt.style.context(["science", "ieee", "grid"]):
        fig, ax = plt.subplots()
        ax.plot(n_alive_lc, label="LEACH")
        ax.plot(n_alive_jr, label="JSO")

        ax.legend(title="protocols")
        ax.set(xlabel="Round")
        ax.set(ylabel="Number of nodes alive")
        ax.autoscale()
        plt.show()


if __name__ == "__main__":
    test_jso_route()
