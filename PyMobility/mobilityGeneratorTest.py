from pymobility.models.mobility import gauss_markov
from pymobility.models.mobility import random_walk
from pymobility.models.mobility import random_direction
from pymobility.models.mobility import random_waypoint
from pymobility.models.mobility import truncated_levy_walk
from pymobility.models.mobility import heterogeneous_truncated_levy_walk

rwk = random_walk(nr_nodes=7700,
                  dimensions=(1000, 1000)
                  )
itrange = 1

for i in range(itrange):
    poz = next(rwk)
    print(poz)
    print(poz.size)

"""
Code spits out an iterable object that at each iteration pops a list of 2-dim lists of x-y coords
One for each user.
Random walk is probably not the best model to be used for mobility generation.
It's not gambler's ruin. 
"""

rwp = random_waypoint(nr_nodes=7700,
                      dimensions=(1000, 1000),
                      velocity=(0.1, 14),
                      wt_max=100
                      )

for i in range(itrange):
    poz = next(rwp)
    print(poz.size)

rdir = random_direction(nr_nodes=7700,
                        dimensions=(1000, 1000),
                        wt_max=10,
                        velocity=(0., 13.89)
                        )

for i in range(itrange):
    poz = next(rdir)
    print(poz.size)

tlwk = truncated_levy_walk(nr_nodes=7700,
                           dimensions=(1000, 1000)
                           )

for i in range(itrange):
    poz = next(tlwk)
    print(poz.size)

htlwk = heterogeneous_truncated_levy_walk(nr_nodes=7700,
                                          dimensions=(1000, 1000)
                                          )

for i in range(itrange):
    poz = next(htlwk)
    print(poz.size)

gm = gauss_markov(nr_nodes=7700,
                  dimensions=(1000, 1000),
                  velocity_mean=3.,
                  variance=2.
                  )

for i in range(itrange):
    poz = next(gm)
    print(poz.size)
