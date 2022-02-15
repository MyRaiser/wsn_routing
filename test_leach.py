import matplotlib.pyplot as plt

from distribution import *
from router import LEACH


def test_leach():
    sink = (0, 0)
    nodes = simple_loader(
        sink,
        uniform_in_square(100, 100, sink)
    )

    # leach = LEACH(*nodes_on_power_line_naive(), n_cluster=5)
    leach = LEACH(*nodes, n_cluster=6)
    leach.initialize()
    n_alive = []
    while len(leach.alive_non_sinks) > 0:
        leach.execute()
        n = len(leach.alive_non_sinks)
        n_alive.append(n)

        # print(
        #     {leach.index(head): [leach.index(n) for n in members] for head, members in leach.clusters.items()}
        # )
        print(f"cluster heads: {len(leach.clusters)}")
        print(f"nodes alive: {n}")
        # print(leach.route)
        # leach.plot()
    print("")
    rounds = list(range(len(n_alive)))
    plt.plot(rounds, n_alive)
    plt.xlim([rounds[0], rounds[-1] + (rounds[-1] - rounds[0]) / 5])
    plt.ylim([0, max(n_alive) + 5])
    plt.xlabel("round")
    plt.ylabel("number of alive nodes")
    plt.show()


if __name__ == "__main__":
    test_leach()
