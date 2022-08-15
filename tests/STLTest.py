from Entities.SpaceTimeLord import SpaceTimeLord
from pymobility.models.mobility import gauss_markov
from pymobility.models.mobility import random_walk
from pymobility.models.mobility import random_direction
from pymobility.models.mobility import random_waypoint
from pymobility.models.mobility import truncated_levy_walk
from pymobility.models.mobility import heterogeneous_truncated_levy_walk

def main():

    pop = 100000
    dims = ()
    test_stl =