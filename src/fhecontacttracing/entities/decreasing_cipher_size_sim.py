import time

import numpy as np
import math
import cmath
import os
import unittest
import random
import gmpy2

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

# Mobility parameters
POPSIZE = 1500
TOTAL_AREA_SIZES = (500, 500)
AREA_SIDES = (50, 50)

# Network parameters (contact threshold not used in current version)
RISK_THRESHOLD = 4  # more than RISK_THRESHOLD contact => at risk
CONTACT_THRESHOLD = 2  # less than CONTACT_THRESHOLD distance away => in contact
MO_COUNT = 2

# Encryption parameters
POLYNOMIAL_DEGREE = 256
CIPHERTEXT_MODULUS = 1 << 744
BIG_MODULUS = 1 << 930
SCALING_FACTOR = 1 << 49

# Movements iterable creation
movements_iter = gauss_markov(nr_nodes=POPSIZE,
                              dimensions=TOTAL_AREA_SIZES,
                              alpha=.5,
                              velocity_mean=7,
                              variance=7)

params = CKKSParameters(poly_degree=POLYNOMIAL_DEGREE,
                        ciph_modulus=CIPHERTEXT_MODULUS,
                        big_modulus=BIG_MODULUS,
                        scaling_factor=SCALING_FACTOR)

keygen = CKKSKeyGenerator(params=params)
pk = keygen.public_key
sk = keygen.secret_key
rk = keygen.relin_key

enco = CKKSEncoder(params=params)

encr = CKKSEncryptor(params=params,
                     public_key=pk,
                     secret_key=sk)

ev = CKKSEvaluator(params=params)

decr = CKKSDecryptor(params=params,
                     secret_key=sk)

f = open("256-744-49multtimes.txt", 'a')
for i in range(2):
    rnx1 = random.randint(0, 50)
    rny1 = random.randint(0, 50)
    rnx2 = random.randint(-50, 100)
    rny2 = random.randint(-50, 100)
    encod = ev.create_constant_plain(const=1 - ((rnx1 - rnx2) ** 2 + (rny1 - rny2) ** 2) / 20000)
    encry = encr.encrypt(plain=encod)
    print(encry.modulus)
    for j in range(15):

        pre_time = time.time()
        encry = ev.multiply(ciph1=encry, ciph2=encry, relin_key=rk)
        f.write(str(time.time() - pre_time) + ', ')
        encry = ev.rescale(ciph=encry, division_factor=params.scaling_factor)

    print(encry.modulus)
    dec = decr.decrypt(ciphertext=encry)
    dcd = enco.decode(dec)
    print(dcd[0])
    f.write('\n')

f.close()
