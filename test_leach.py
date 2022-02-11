from distribution import nodes_on_power_line_naive
from router import LEACH


def test_leach():
    leach = LEACH(*nodes_on_power_line_naive(), n_cluster=10)
    leach.initialize()
    leach.execute()
    print(f"cluster heads: {len(leach.clusters)}")
    print(leach.route)
    leach.plot()
    print("")


if __name__ == "__main__":
    test_leach()
