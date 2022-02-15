import matplotlib.pyplot as plt

from distribution import *
from router import APTEEN


def test_apteen():
    sink = (0, 0)
    nodes = simple_loader(
        sink,
        uniform_in_square(100, 100, sink)
    )

    # apteen = LEACH(*nodes_on_power_line_naive(), n_cluster=5)
    apteen = APTEEN(*nodes, n_cluster=5)
    apteen.initialize()
    n_alive = []
    while len(apteen.alive_non_sinks) > 0:
        apteen.execute()
        n = len(apteen.alive_non_sinks)
        n_alive.append(n)

        # print(
        #     {apteen.index(head): [apteen.index(n) for n in members] for head, members in apteen.clusters.items()}
        # )
        # print(f"cluster heads: {len(apteen.clusters)}")
        # print(f"nodes alive: {n}")
        # print(apteen.route)
        if n < 10:
            apteen.plot()
    print("")
    rounds = list(range(len(n_alive)))
    plt.plot(rounds, n_alive)
    plt.xlim([rounds[0], rounds[-1] + (rounds[-1] - rounds[0]) / 5])
    plt.ylim([0, max(n_alive) + 5])
    plt.xlabel("round")
    plt.ylabel("number of alive nodes")
    plt.show()


if __name__ == "__main__":
    test_apteen()
