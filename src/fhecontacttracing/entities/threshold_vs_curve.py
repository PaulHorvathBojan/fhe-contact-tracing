import time

import numpy as np
import math
import cmath
import os
import unittest
import random
import gmpy2
import util.random_sample

from scipy.stats import bernoulli
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters
from gmpy2 import mpfr
from numpy import random

import util.matrix_operations as mat
from tests.helper import check_complex_vector_approx_eq
from util.plaintext import Plaintext
from util.random_sample import sample_random_complex_vector
from util.ciphertext import Ciphertext
from util.public_key import PublicKey

from pymobility.models.mobility import gauss_markov

from space_time_lord import *
from gov_agent import *
from mob_operator import *
from user import *

TICK_COUNT = [200, 100, 100, 100, 50, 50, 35, 30, 25, 20]

POPSIZE = 500  #7.5k/km**2 total, 5k/km**2 smartphone
TOTAL_AREA_SIZES = (200, 200)
AREA_SIDES = (50, 50)

RISK_THRESHOLD = 10
MO_COUNT = 2
CONTACT_TUPLES = [(1, 14), (2, 12), (3, 11), (4, 10), (5, 9), (6, 9), (7, 9), (8, 8), (9, 7), (10, 7)]

INFECTED_COUNT = 50

class DualIter:

    def __init__(self, init_iter):
        self._iterator = init_iter
        self._change = True
        self._next_val = None

    def __next__(self):
        if self._change:
            self._next_val = next(self._iterator)
        self._change = not self._change

        return self._next_val.copy()


class MinutelyMovement:
    def __init__(self, movement_iter):
        self._move = movement_iter

    def __next__(self):
        for i in range(60):
            self_next = next(self._move)

        return self_next

movements_iter = gauss_markov(nr_nodes=POPSIZE,
                              dimensions=TOTAL_AREA_SIZES,
                              alpha=.5,
                              velocity_mean=7,
                              variance=7)

minutely_gm = MinutelyMovement(movements_iter)

dual_gm = DualIter(minutely_gm)

for iter_ct in range(len(CONTACT_TUPLES)):

    contact_tuple = CONTACT_TUPLES[iter_ct]

    plain_stl = SpaceTimeLord(movements_iterable=dual_gm,
                              mo_count=MO_COUNT,
                              risk_thr=RISK_THRESHOLD,
                              area_sizes=AREA_SIDES,
                              max_sizes=TOTAL_AREA_SIZES,
                              exponent=contact_tuple[1])

    simple_stl = SimpleSTL(movements_iterable=dual_gm,
                           mo_count=MO_COUNT,
                           risk_thr=RISK_THRESHOLD,
                           area_sizes=AREA_SIDES,
                           max_sizes=TOTAL_AREA_SIZES,
                           contact_thr=contact_tuple[0])

    infected_vect = util.random_sample.sample_hamming_weight_vector(length=POPSIZE,
                                                                hamming_weight=INFECTED_COUNT)
    infected_vect = list(map(abs, infected_vect))

    plain_stl._ga._status = infected_vect
    simple_stl._ga._status = infected_vect.copy()

    for i in range(TICK_COUNT[iter_ct]):
        plain_stl.tick()
        simple_stl.tick()

    f = open("thrvscurve" + str(contact_tuple[0]) + "m.txt", 'a')
    for i in range(plain_stl.user_count):
        plain_stl.users[i].ping_mo_for_score()
        simple_stl.users[i].ping_mo_for_score()
        f.write(str(simple_stl.users[i]._score) + ", " + str(plain_stl.users[i]._score) + "\n")
    f.close()
