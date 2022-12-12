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


class TripleIter:

    def __init__(self, init_iter):
        self._iterator = init_iter
        self._change = 0
        self._next_val = None

    def __next__(self):
        if self._change == 0:
            self._next_val = next(self._iterator)
            self._change = 2
        self._change -= 1

        return self._next_val.copy()


class SameLocIter:
    def __init__(self, loc_val, loc_ct):
        self.vallist = []
        for i in range(loc_ct):
            self.vallist.append(loc_val)

    def __next__(self):
        return self.vallist.copy()


class MinutelyMovement:
    def __init__(self, movement_iter):
        self._move = movement_iter

    def __next__(self):
        for i in range(60):
            self_next = next(self._move)

        return self_next


class BatchIter:
    def __init__(self, batchsize, batchcount):
        self.const_list = []
        for i in range(batchcount):
            for j in range(batchsize):
                self.const_list.append([i * 100, i * 100])

    def __next__(self):
        return self.const_list.copy()


class IntercalIter:
    def __init__(self, batchsize, batchcount):
        self.const_list = []
        for i in range(batchcount):
            for j in range(batchsize):
                self.const_list.append([j * 100, j * 100])

    def __next__(self):
        return self.const_list.copy()


# Not sure if assigning random seed would be best.
# np.random.seed(1234)

# Simulation parameters
TICK_COUNT = 10

# Mobility parameters
POPSIZE = 40  # 7.5k/km**2
TOTAL_AREA_SIZES = (200, 200)
AREA_SIDES = (50, 50)

# Network parameters (contact threshold not used in current version)
RISK_THRESHOLD = 10  # more than RISK_THRESHOLD contact => at risk
CONTACT_THRESHOLD = 2  # less than CONTACT_THRESHOLD distance away => in contact
MO_COUNT = 2

# Encryption parameters
POLYNOMIAL_DEGREE = 256
CIPHERTEXT_MODULUS = 1 << 744
BIG_MODULUS = 1 << 930
SCALING_FACTOR = 1 << 49

# Infection param
INFECTED_COUNT = 5

# Movements iterable creation
movements_iter = gauss_markov(nr_nodes=POPSIZE,
                              dimensions=TOTAL_AREA_SIZES,
                              alpha=.5,
                              velocity_mean=7,
                              variance=7)

minutely_gm = MinutelyMovement(movements_iter)

triple_gm = TripleIter(minutely_gm)

params = CKKSParameters(poly_degree=POLYNOMIAL_DEGREE,
                        ciph_modulus=CIPHERTEXT_MODULUS,
                        big_modulus=BIG_MODULUS,
                        scaling_factor=SCALING_FACTOR)

pre_time = time.time()
simple_stl = SimpleSTL(movements_iterable=triple_gm,
                       mo_count=MO_COUNT,
                       risk_thr=RISK_THRESHOLD,
                       area_sizes=AREA_SIDES,
                       max_sizes=TOTAL_AREA_SIZES,
                       contact_thr=CONTACT_THRESHOLD)

plain_stl = SpaceTimeLord(movements_iterable=triple_gm,
                          mo_count=MO_COUNT,
                          risk_thr=RISK_THRESHOLD,
                          area_sizes=AREA_SIDES,
                          max_sizes=TOTAL_AREA_SIZES,
                          exponent=12)

pre_time = time.time()
fhe_stl = SpaceTimeLordUntrustedGA(movements_iterable=triple_gm,
                                   mo_count=MO_COUNT,
                                   area_sizes=AREA_SIDES,
                                   max_sizes=TOTAL_AREA_SIZES,
                                   params=params)
print("Cipher STL setup time: " + str(time.time() - pre_time))

infected_vect = util.random_sample.sample_hamming_weight_vector(length=POPSIZE, hamming_weight=INFECTED_COUNT)
infected_vect = list(map(abs, infected_vect))

simple_stl._ga._status = infected_vect
plain_stl._ga._status = infected_vect.copy()
fhe_stl._ga._status = infected_vect.copy()

tot_error = 0

f = open("usrcountsimfirsttick.txt", 'a')
pre_time = time.time()
fhe_stl.tick()
f.write(str(time.time() - pre_time) + '\n')
f.close()

f = open("usrcountsimtick.txt", 'a')
for i in range(TICK_COUNT):
    pre_time = time.time()
    fhe_stl.tick()
    f.write(str(time.time() - pre_time) + '\n')

    plain_stl.tick()
    simple_stl.tick()
f.close()

f = open("usercountsimerror.txt", 'a')
for j in range(plain_stl.user_count):
    simple_stl._users[j].ping_mo_for_score()
    plain_stl._users[j].ping_mo_for_score()
    fhe_stl._users[j].ping_mo_for_score()

    f.write(str(simple_stl._users[j]._score) + "," + str(plain_stl._users[j]._score) + "," +
            str(fhe_stl._users[j]._score) + '\n')
f.close()
print("check result files")
