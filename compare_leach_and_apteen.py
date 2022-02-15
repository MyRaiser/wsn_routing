import matplotlib.pyplot as plt

from distribution import *
from router import APTEEN, LEACH


def main():
    sink = (0, 0)
    distribution = uniform_in_square(150, 100, sink)
    n_cluster = 5

    nodes_leach = simple_loader(sink, distribution)
    leach = LEACH(*nodes_leach, n_cluster=n_cluster)
    nodes_apteen = simple_loader(sink, distribution)
    apteen = APTEEN(*nodes_apteen, n_cluster=n_cluster)

    leach.initialize()
    alive_leach = []
    while (n := len(leach.alive_non_sinks)) > 0:
        alive_leach.append(n)
        leach.execute()
    print(alive_leach)

    apteen.initialize()
    alive_apteen = []
    while (n := len(apteen.alive_non_sinks)) > 0:
        alive_apteen.append(n)
        apteen.execute()
    print(alive_apteen)

    print("-" * 10)
    with plt.style.context(["science", "ieee", "grid"]):
        fig, ax = plt.subplots()
        ax.plot(list(range(len(alive_leach))), alive_leach, label="LEACH")
        ax.plot(list(range(len(alive_apteen))), alive_apteen, label="APTEEN")

        ax.legend(title="protocols")
        ax.set(xlabel="Round")
        ax.set(ylabel="Number of nodes alive")
        ax.autoscale()
        plt.show()


if __name__ == "__main__":
    main()
