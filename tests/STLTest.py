from Entities.SpaceTimeLord import SpaceTimeLord
from pymobility.models.mobility import gauss_markov
from pymobility.models.mobility import random_walk
from pymobility.models.mobility import random_direction
from pymobility.models.mobility import random_waypoint
from pymobility.models.mobility import truncated_levy_walk
from pymobility.models.mobility import heterogeneous_truncated_levy_walk

import numpy as np


class MinutelyMovement():
    def __init__(self, movement_iter):
        self._move = movement_iter

    def __next__(self):
        for i in range(60):
            self_next = next(self._move)

        return self_next


def main():
    pop = 10
    dims = (3652, 3652)

    np.random.seed(1234)

    restrict_milan_rwk = random_walk(nr_nodes=pop,
                                     dimensions=dims
                                     )

    restrict_milan_rwp = random_waypoint(nr_nodes=pop,
                                         dimensions=dims,
                                         velocity=(0.1, 14),
                                         wt_max=100
                                         )

    restrict_milan_rdir = random_direction(nr_nodes=pop,
                                           dimensions=dims,
                                           wt_max=10,
                                           velocity=(0., 13.89)
                                           )

    restrict_milan_tlwk = truncated_levy_walk(nr_nodes=pop,
                                              dimensions=dims
                                              )

    restrict_milan_htlwk = heterogeneous_truncated_levy_walk(nr_nodes=pop,
                                                             dimensions=dims
                                                             )

    restrict_milan_gm = gauss_markov(nr_nodes=pop,
                                     dimensions=dims,
                                     velocity_mean=3.,
                                     variance=2.
                                     )

    movement_iter = restrict_milan_gm

    minute_iter = MinutelyMovement(movement_iter=movement_iter)

    test_stl = SpaceTimeLord(movements_iterable=minute_iter,
                             mo_count=1,
                             risk_thr=5,
                             area_sizes=dims
                             )

    test_stl.