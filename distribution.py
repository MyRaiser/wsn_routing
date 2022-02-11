import numpy as np
from numpy import pi, cos, sin
from numpy.random import rand
from router import Node, NodeCategory


def nodes_on_power_line_naive() -> tuple[Node, list[Node]]:
    sink_position = [0, 0]
    interval = 100
    d_max = 10
    phi_max = pi / 10
    n_relay = 5
    n_sensor_per_relay = 10
    r_max = 80

    relays = []
    x, y = sink_position
    sink = Node(np.array(sink_position), NodeCategory.sink)
    for _ in range(n_relay):
        d = d_max * rand()
        phi = phi_max * rand() - phi_max / 2
        r = interval + d
        x += r * cos(phi)
        y += r * sin(phi)
        relays.append(
            Node(np.array([x, y]), NodeCategory.relay)
        )

    sensors = []
    for relay in relays:
        for _ in range(n_sensor_per_relay):
            r = r_max * rand()
            theta = 2 * pi * rand()
            x = relay.position[0] + r * cos(theta)
            y = relay.position[1] + r * sin(theta)
            sensors.append(
                Node(np.array([x, y]), NodeCategory.sensor)
            )
    return sink, relays + sensors
