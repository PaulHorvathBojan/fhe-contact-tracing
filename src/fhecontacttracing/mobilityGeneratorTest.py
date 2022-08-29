from pymobility.models.mobility import gauss_markov
from pymobility.models.mobility import random_walk
from pymobility.models.mobility import random_direction
from pymobility.models.mobility import random_waypoint
from pymobility.models.mobility import truncated_levy_walk
from pymobility.models.mobility import heterogeneous_truncated_levy_walk

import pandas as pd
import numpy as np
import gzip
import re
import time

# rwk = random_walk(nr_nodes=7700,
#                   dimensions=(1000, 1000)
#                   )
# itrange = 1
#
# for i in range(itrange):
#     poz = next(rwk)
#     print(poz)
#     print(poz.size)
#
# """
# Code spits out an iterable object that at each iteration pops a list of 2-dim lists of x-y coords
# One for each user.
# Random walk is probably not the best model to be used for mobility generation.
# It's not gambler's ruin.
# """
#
# rwp = random_waypoint(nr_nodes=7700,
#                       dimensions=(1000, 1000),
#                       velocity=(0.1, 14),
#                       wt_max=100
#                       )
#
# for i in range(itrange):
#     poz = next(rwp)
#     print(poz.size)
#
# rdir = random_direction(nr_nodes=7700,
#                         dimensions=(1000, 1000),
#                         wt_max=10,
#                         velocity=(0., 13.89)
#                         )
#
# for i in range(itrange):
#     poz = next(rdir)
#     print(poz.size)
#
# tlwk = truncated_levy_walk(nr_nodes=7700,
#                            dimensions=(1000, 1000)
#                            )
#
# for i in range(itrange):
#     poz = next(tlwk)
#     print(poz.size)
#
# htlwk = heterogeneous_truncated_levy_walk(nr_nodes=7700,
#                                           dimensions=(1000, 1000)
#                                           )
#
# for i in range(itrange):
#     poz = next(htlwk)
#     print(poz.size)
#
# gm = gauss_markov(nr_nodes=7700,
#                   dimensions=(1000, 1000),
#                   velocity_mean=3.,
#                   variance=2.
#                   )
#
# for i in range(itrange):
#     poz = next(gm)
#     print(poz.size)

dims = (13482, 13482)
pop = 1371498

milan_rwk = random_walk(nr_nodes=pop,
                        dimensions=dims
                        )

milan_rwp = random_waypoint(nr_nodes=pop,
                            dimensions=dims,
                            velocity=(0.1, 14),
                            wt_max=100
                            )

milan_rdir = random_direction(nr_nodes=pop,
                              dimensions=dims,
                              wt_max=10,
                              velocity=(0., 13.89)
                              )

milan_tlwk = truncated_levy_walk(nr_nodes=pop,
                                 dimensions=dims
                                 )

milan_htlwk = heterogeneous_truncated_levy_walk(nr_nodes=pop,
                                                dimensions=dims
                                                )

milan_gm = gauss_markov(nr_nodes=pop,
                        dimensions=dims,
                        velocity_mean=3.,
                        variance=2.
                        )

sim_duration = 1440  # the equivalent of 1 day in minutes
# Code for the random walk trace --- already in file and compressed.
# milan_rwk_df = pd.DataFrame(next(milan_rwk),
#                             index=index,
#                             columns=["x", "y"]
#                             )
# milan_rwk_df.to_csv('MilanRandWalk.csv')
#
# while ctr <= sim_duration:
#     ctr += 1
#     tick_no_array += 1
#     zippable = [tick_no_array, usr_array]
#     tuples = list(zip(*zippable))
#     index = pd.MultiIndex.from_tuples(tuples,
#                                       names=["tick_no", "uID"]
#                                       )
#     milan_rwk_next = pd.DataFrame(next(milan_rwk),
#                                   index=index,
#                                   columns=["x", "y"]
#                                   )
#
#     milan_rwk_next.to_csv('MilanRandWalk.csv',
#                           mode='a',
#                           index=False,
#                           header=False
#                           )

# Code for random waypoint trace
# ctr = 1
# tick_no_array = np.zeros((pop,),
#                          dtype=int
#                          )
# milan_rwp_df = pd.DataFrame(next(milan_rwp),
#                             index=index,
#                             columns=["x", "y"]
#                             )
# milan_rwp_df.to_csv('MilanRandWaypoint.csv')
#
# while ctr <= sim_duration:
#     ctr += 1
#     tick_no_array += 1
#     zippable = [tick_no_array, usr_array]
#     tuples = list(zip(*zippable))
#     index = pd.MultiIndex.from_tuples(tuples,
#                                       names=["tick_no", "uID"]
#                                       )
#     milan_rwp_next = pd.DataFrame(next(milan_rwp),
#                                   index=index,
#                                   columns=["x", "y"]
#                                   )
#
#     milan_rwp_next.to_csv('MilanRandWaypoint.csv',
#                           mode='a',
#                           index=False,
#                           header=False
#                           )
#
# # Code for random direction trace
# ctr = 1
# tick_no_array = np.zeros((pop,),
#                          dtype=int
#                          )
# milan_rdir_df = pd.DataFrame(next(milan_rdir),
#                              index=index,
#                              columns=["x", "y"]
#                              )
# milan_rdir_df.to_csv('MilanRandDir.csv')
#
# while ctr <= sim_duration:
#     ctr += 1
#     tick_no_array += 1
#     zippable = [tick_no_array, usr_array]
#     tuples = list(zip(*zippable))
#     index = pd.MultiIndex.from_tuples(tuples,
#                                       names=["tick_no", "uID"]
#                                       )
#     milan_rdir_next = pd.DataFrame(next(milan_rdir),
#                                    index=index,
#                                    columns=["x", "y"]
#                                    )
#
#     milan_rdir_next.to_csv('MilanRandDir.csv',
#                            mode='a',
#                            index=False,
#                            header=False
#                            )
#
# # Code for truncated Levy walk
# ctr = 1
# tick_no_array = np.zeros((pop,),
#                          dtype=int
#                          )
# milan_tlwk_df = pd.DataFrame(next(milan_tlwk),
#                              index=index,
#                              columns=["x", "y"])
# milan_tlwk_df.to_csv('MilanLevy.csv')
#
# while ctr <= sim_duration:
#     ctr += 1
#     tick_no_array += 1
#     zippable = [tick_no_array, usr_array]
#     tuples = list(zip(*zippable))
#     index = pd.MultiIndex.from_tuples(tuples,
#                                       names=["tick_no", "uID"]
#                                       )
#     milan_tlwk_next = pd.DataFrame(next(milan_tlwk),
#                                    index=index,
#                                    columns=["x", "y"]
#                                    )
#
#     milan_tlwk_next.to_csv('MilanLevy.csv',
#                            mode='a',
#                            index=False,
#                            header=False
#                            )
#
# # Code for heterogeneous truncated Levy walk
# ctr = 1
# tick_no_array = np.zeros((pop,),
#                          dtype=int
#                          )
# milan_htlwk_df = pd.DataFrame(next(milan_htlwk),
#                               index=index,
#                               columns=["x", "y"])
# milan_htlwk_df.to_csv('MilanHeteroLevy.csv')
#
# while ctr <= sim_duration:
#     ctr += 1
#     tick_no_array += 1
#     zippable = [tick_no_array, usr_array]
#     tuples = list(zip(*zippable))
#     index = pd.MultiIndex.from_tuples(tuples,
#                                       names=["tick_no", "uID"]
#                                       )
#     milan_htlwk_next = pd.DataFrame(next(milan_htlwk),
#                                     index=index,
#                                     columns=["x", "y"]
#                                     )
#
#     milan_htlwk_next.to_csv('MilanHeteroLevy.csv',
#                             mode='a',
#                             index=False,
#                             header=False
#                             )
#
# # Code for Gauss-Markov
# ctr = 1
# tick_no_array = np.zeros((pop,),
#                          dtype=int
#                          )
# milan_gm_df = pd.DataFrame(next(milan_gm),
#                            index=index,
#                            columns=["x", "y"])
# milan_gm_df.to_csv('MilanGaussMarkov.csv')
#
# while ctr <= sim_duration:
#     ctr += 1
#     tick_no_array += 1
#     zippable = [tick_no_array, usr_array]
#     tuples = list(zip(*zippable))
#     index = pd.MultiIndex.from_tuples(tuples,
#                                       names=["tick_no", "uID"]
#                                       )
#     milan_gm_next = pd.DataFrame(next(milan_gm),
#                                  index=index,
#                                  columns=["x", "y"]
#                                  )
#
#     milan_gm_next.to_csv('MilanGaussMarkov.csv',
#                          mode='a',
#                          index=False,
#                          header=False
#                          )

# Test viability of reading zipped csvs --- not viable for fully simulated Milan-like
# path = "mobilities/MilanGaussMarkov.csv.gz"
# tickrange = range(1440)
# adone = 0
# for tick in tickrange:
#     tick_no_array = np.zeros(shape=(pop,),
#                              dtype=int
#                              )
#
#     uID = range(pop)
#
#     test_df = pd.read_csv(filepath_or_buffer=path,
#                           header=0,
#                           skiprows=adone + tick * pop,
#                           nrows=pop
#                           )
#
#     print("Tick nr " + (str)(tick))
#     print(test_df.head())
#
#     adone = 1

# Reduced population sim:
# Have the population be 100k and restrict dimensions to preserve density (roughly 7.5k/sqkm)
pop = 100000
dims = (3652, 3652)

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


def mobility_trace_single_to_df(mob_iter, index=None, collection_interval=1):
    interv_ctr = 0
    while interv_ctr < collection_interval:
        curr_poss = next(mob_iter)
        interv_ctr += 1

    curr_poss_df = pd.DataFrame(data=curr_poss,
                                index=index,
                                columns=["x", "y"]
                                )

    return curr_poss_df


def mobility_df_to_file(mob_iter, collection_interval, population, path, sim_duration):
    curr_tick = 0

    tick_no_array = np.zeros(shape=(population,),
                             dtype=int
                             )
    tick_no_array += curr_tick

    uIDs = range(population)

    zippable = [tick_no_array, uIDs]

    tuples = list(zip(*zippable))

    curr_index = pd.MultiIndex.from_tuples(tuples,
                                           names=["tick_no", "uID"]
                                           )

    curr_df = mobility_trace_single_to_df(mob_iter=mob_iter,
                                          index=curr_index,
                                          collection_interval=collection_interval
                                          )

    curr_df.to_csv(path_or_buf=path)

    curr_tick += 1

    while curr_tick < sim_duration:
        tick_no_array = np.zeros(shape=(population,),
                                 dtype=int
                                 )
        tick_no_array += curr_tick

        uIDs = range(population)

        zippable = [tick_no_array, uIDs]

        tuples = list(zip(*zippable))

        curr_index = pd.MultiIndex.from_tuples(tuples,
                                               names=["tick_no", "uID"]
                                               )

        curr_df = mobility_trace_single_to_df(mob_iter=mob_iter,
                                              index=curr_index,
                                              collection_interval=collection_interval
                                              )

        curr_df.to_csv(path_or_buf=path,
                       mode='a',
                       index=False,
                       header=False
                       )

        curr_tick += 1


# Run the _to_file script
# mobility_df_to_file(mob_iter=restrict_milan_rwk,
#                     collection_interval=60,
#                     population=pop,
#                     path="MilanRandWalk100k.csv",
#                     sim_duration=sim_duration
#                     )

# Test viability of reading zipped csvs
path = "MilanRandWalk100k.csv.gz"
tickrange = range(1440)


def print_single_mobility_from_csv(tickrange, path):
    adone = 0
    for tick in tickrange:
        tick_no_array = np.zeros(shape=(pop,),
                                 dtype=int
                                 )
        tick_no_array = tick_no_array + tick
        uIDs = range(pop)
        zippable = [tick_no_array, uIDs]
        tuples = list(zip(*zippable))

        curr_index = pd.MultiIndex.from_tuples(tuples,
                                               names=["tick_no", "uID"]
                                               )

        test_df = pd.read_csv(filepath_or_buffer=path,
                              header=0,
                              skiprows=adone + tick * pop,
                              nrows=pop
                              )

        test_df.set_index(curr_index)

        print("Tick nr " + (str)(tick))
        print(test_df.head())

        adone = 1

def get_tick_no_from_gz(path, tick_no, pop_size):
    file = gzip.open(filename=path,
                     mode='rt'
                     )

    rez = []

    file.readline()
    for i in range(tick_no * pop_size):
        file.readline()

    for i in range(pop_size):
        content = file.readline()
        contents = re.split(pattern='[, , \n]+',
                            string=content
                            )
        contents = contents[:-1]
        floats = [float(x) for x in contents]
        ints = np.rint(floats)
        coords = [int(x) for x in ints]

        rez.append(coords)

    file.close()

    return rez

pre_from_file = time.time()
from_file = get_tick_no_from_gz(path=path,
                                tick_no=1000,
                                pop_size=pop
                                )
post_from_file = time.time()
time_from_file = post_from_file - pre_from_file

pre_from_iter = time.time()
from_iter = next(milan_rwk)
post_from_iter = time.time()
time_from_iter = post_from_iter - pre_from_iter

print("From file: " + str(time_from_file))
print("From iter: " + str(time_from_iter))