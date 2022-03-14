import numpy as np
from numpy.random import rand
from pyMetaheuristic.algorithm import particle_swarm_optimization

from . import Target
from .initializer import logistic_map


def bounded(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """boundary conditions"""
    while any(x > ub) or any(x < lb):
        x[x > ub] = ((x - ub) + lb)[x > ub]
        x[x < lb] = ((x - lb) + ub)[x < lb]
    return x


def jso(
        target: Target,
        *,
        n_pop: int,
        iter_max: int,
        gamma: float = 0.1,
        beta: float = 3,
        c_0: float = 0.5,
        eta: float = 4
):
    """
    Artificial Jellyfish Search Optimizer
    ref: Chou, J. S. , and D. N. Truong . "A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean."
        Applied Mathematics and Computation 389(2021):125535.
    """
    func, _, lb, ub = target
    pop, cost = logistic_map(target, n_pop, eta)  # population and cost
    i = np.argmin(cost)
    opt = np.copy(pop[i])  # historical optimum
    opt_cost = cost[i]
    yield opt, opt_cost

    t = 1  # time / round
    while t < iter_max:
        interest = np.arange(n_pop)
        np.random.shuffle(interest)
        for i in range(n_pop):
            # time control
            c = np.abs(
                (1 - t / iter_max) * (2 * rand() - 1)
            )
            if c >= c_0:  # jellyfish follows ocean current
                e_c = beta * rand()
                mu = np.mean(pop, axis=0)
                trend = opt - e_c * mu
                pop[i] = pop[i] + rand() * trend
            else:  # jellyfish moves inside swarm
                if rand() > (1 - c):  # do passive move
                    pop[i] = pop[i] + gamma * rand() * (ub - lb)
                else:  # do active move
                    j = interest[i]
                    if cost[i] >= cost[j]:
                        direction = pop[j] - pop[i]
                    else:
                        direction = pop[i] - pop[j]
                    step = rand() * direction
                    pop[i] = pop[i] + step
            pop[i] = bounded(pop[i], lb, ub)
            cost[i] = func(pop[i])
            if cost[i] < opt_cost:
                opt[:] = pop[i]  # copy instead of view
                opt_cost = cost[i]

        yield opt, opt_cost  # yield opt for this round
        t += 1


def pso(
        target: Target,
        *,
        n_pop: int,
        iter_max: int,
):
    func, _, lb, ub = target
    # func = np.sum
    ret = particle_swarm_optimization(
        swarm_size=n_pop, min_values=lb, max_values=ub, iterations=iter_max, target_function=func
    )
    # print(ret)
    yield ret[:-1], ret[-1]
