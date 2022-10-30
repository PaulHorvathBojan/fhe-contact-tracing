import time
import numpy as np
import math
import cmath
import os
import unittest
import random
import gmpy2

from numpy import random
from scipy.stats import bernoulli
from gmpy2 import mpfr

from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters
import util.matrix_operations as mat
from tests.helper import check_complex_vector_approx_eq
from util.plaintext import Plaintext
from util.random_sample import sample_random_complex_vector
from util.ciphertext import Ciphertext
from util.public_key import PublicKey

from pymobility.models.mobility import gauss_markov

from user import *
from mob_operator import *
from gov_agent import *

class SpaceTimeLord:

    def __init__(self, movements_iterable, mo_count, risk_thr, area_sizes, max_sizes):
        self._movements_iterable = movements_iterable
        self._risk_threshold = risk_thr
        self._max_x = max_sizes[0]
        self._max_y = max_sizes[1]
        self._area_size_x = area_sizes[0]
        self._area_size_y = area_sizes[1]

        self._usr_count = 0
        self._mo_count = 0

        self._mos = []
        self._users = []
        self._current_locations = next(movements_iterable)
        self._current_locations = list(map(lambda y: list(map(lambda x: int(round(x)),
                                                              self._current_locations[y])),
                                           range(len(self._current_locations))))
        self._curr_time = 0

        self._ga = GovAgent(risk_threshold=self._risk_threshold)

        while self._mo_count < mo_count:
            new_mo = MobileOperator(ga=self._ga,
                                    mo_id=self._mo_count,
                                    area_side_x=self._area_size_x,
                                    area_side_y=self._area_size_y,
                                    max_x=self._max_x,
                                    max_y=self._max_y
                                    )
            for prev_mo in self._mos:
                new_mo.register_other_mo(new_mo=prev_mo)
                prev_mo.register_other_mo(new_mo=new_mo)
            self._mos.append(new_mo)
            self._mo_count += 1

        for i in range(len(self._current_locations)):
            self.add_user(location=self._current_locations[i])

    def add_user(self, location):
        new_user = ProtocolUser(init_x=location[0],
                                init_y=location[1],
                                mo=self._mos[self._usr_count % self._mo_count],
                                uid=self._usr_count,
                                ga=self._ga
                                )

        self._users.append(new_user)
        self._usr_count += 1

    def get_current_locations(self):
        return self._current_locations

    current_locations = property(fget=get_current_locations)

    def get_area_sizes(self):
        return self._area_size_x, self._area_size_y

    area_sizes = property(fget=get_area_sizes)

    def get_user_count(self):
        return self._usr_count

    user_count = property(fget=get_user_count)

    def get_mo_count(self):
        return self._mo_count

    mo_count = property(fget=get_mo_count)

    def get_mos(self):
        return self._mos

    mos = property(fget=get_mos)

    def get_users(self):
        return self._users

    users = property(fget=get_users)

    def get_current_time(self):
        return self._curr_time

    current_time = property(fget=get_current_time)

    def get_ga(self):
        return self._ga

    ga = property(fget=get_ga)

    def tick(self):
        for i in range(self._mo_count):
            self._mos[i].tick()

        if self._curr_time % 1440 == 0:
            self._ga.daily()

        self._current_locations = next(self._movements_iterable)
        self._current_locations = list(map(lambda y: list(map(lambda x: int(round(x)),
                                                              self._current_locations[y])),
                                           range(len(self._current_locations))))
        self._curr_time += 1
        for i in range(len(self._users)):
            self._users[i].move_to(new_x=self._current_locations[i][0],
                                   new_y=self._current_locations[i][1],
                                   update_time=self._curr_time
                                   )


class EncryptionSTL:
    def __init__(self, movements_iterable, mo_count, risk_thr, area_sizes, max_sizes, degree, cipher_modulus,
                 big_modulus, scaling_factor):
        self._movements_iterable = movements_iterable
        self._risk_threshold = risk_thr
        self._max_x = max_sizes[0]
        self._max_y = max_sizes[1]
        self._area_size_x = area_sizes[0]
        self._area_size_y = area_sizes[1]

        self._usr_count = 0
        self._mo_count = 0

        self._mos = []
        self._users = []

        self._current_locations = next(movements_iterable)
        self._current_locations = list(map(lambda y: list(map(lambda x: int(round(x)),
                                                              self._current_locations[y])),
                                           range(len(self._current_locations))))

        self._curr_time = 0

        self._ga = EncryptionGovAgent(risk_threshold=self._risk_threshold,
                                      degree=degree,
                                      cipher_modulus=cipher_modulus,
                                      big_modulus=big_modulus,
                                      scaling_factor=scaling_factor)

        while self._mo_count < mo_count:
            new_mo = EncryptionMO(ga=self._ga,
                                  mo_id=self._mo_count,
                                  area_side_x=self._area_size_x,
                                  area_side_y=self._area_size_y,
                                  max_x=self._max_x,
                                  max_y=self._max_y
                                  )
            for prev_mo in self._mos:
                new_mo.register_other_mo(new_mo=prev_mo)
                prev_mo.register_other_mo(new_mo=new_mo)
            self._mos.append(new_mo)
            self._mo_count += 1

        for i in range(len(self._current_locations)):
            self.add_user(location=self._current_locations[i])

    def add_user(self, location):
        new_user = EncryptedUser(init_x=location[0],
                                 init_y=location[1],
                                 mo=self._mos[self._usr_count % self._mo_count],
                                 uid=self._usr_count,
                                 ga=self._ga
                                 )

        self._users.append(new_user)
        self._usr_count += 1

    def get_current_locations(self):
        return self._current_locations

    current_locations = property(fget=get_current_locations)

    def get_area_sizes(self):
        return self._area_size_x, self._area_size_y

    area_sizes = property(fget=get_area_sizes)

    def get_user_count(self):
        return self._usr_count

    user_count = property(fget=get_user_count)

    def get_mo_count(self):
        return self._mo_count

    mo_count = property(fget=get_mo_count)

    def get_mos(self):
        return self._mos

    mos = property(fget=get_mos)

    def get_users(self):
        return self._users

    users = property(fget=get_users)

    def get_current_time(self):
        return self._curr_time

    current_time = property(fget=get_current_time)

    def get_ga(self):
        return self._ga

    ga = property(fget=get_ga)

    def tick(self):
        for i in range(self._mo_count):
            self._mos[i].tick()

        if self._curr_time % 1440 == 0:
            self._ga.daily()

        self._current_locations = next(self._movements_iterable)
        self._current_locations = list(map(lambda y: list(map(lambda x: int(round(x)),
                                                              self._current_locations[y])),
                                           range(len(self._current_locations))))
        self._curr_time += 1
        for i in range(self._usr_count):
            self._users[i].move_to(new_x=self._current_locations[i][0],
                                   new_y=self._current_locations[i][1],
                                   update_time=self._curr_time
                                   )


class SpaceTimeLordUntrustedGA:
    def __init__(self, movements_iterable, mo_count, area_sizes, max_sizes, params):
        self._movements_iterable = movements_iterable
        self._max_x = max_sizes[0]
        self._max_y = max_sizes[1]
        self._area_size_x = area_sizes[0]
        self._area_size_y = area_sizes[1]
        self._CKKSParams = params

        self._usr_count = 0
        self._mo_count = 0

        self._mos = []
        self._users = []

        self._current_locations = next(movements_iterable)
        self._current_locations = list(map(lambda y: list(map(lambda x: int(round(x)),
                                                              self._current_locations[y])),
                                           range(len(self._current_locations))))

        self._curr_time = 0

        self._ga = EncryptionUntrustedGA(encryption_params=params)

        while self._mo_count < mo_count:
            new_mo = EncryptionMOUntrustedGA(ga=self._ga,
                                             id=self._mo_count,
                                             area_side_x=self._area_size_x,
                                             area_side_y=self._area_size_y,
                                             max_x=self._max_x,
                                             max_y=self._max_y,
                                             encryption_params=params
                                             )
            for prev_mo in self._mos:
                new_mo.register_other_mo(new_mo=prev_mo)
                prev_mo.register_other_mo(new_mo=new_mo)
            self._mos.append(new_mo)
            self._mo_count += 1

        for i in range(len(self._current_locations)):
            self.add_user(location=self._current_locations[i])

    def add_user(self, location):
        new_user = EncryptionUserUntrustedGA(x=location[0],
                                             y=location[1],
                                             mo=self._mos[self._usr_count % self._mo_count],
                                             uid=self._usr_count,
                                             ga=self._ga,
                                             encryption_params=self._CKKSParams
                                             )

        self._users.append(new_user)
        self._usr_count += 1

    def get_current_locations(self):
        return self._current_locations

    current_locations = property(fget=get_current_locations)

    def get_area_sizes(self):
        return self._area_size_x, self._area_size_y

    area_sizes = property(fget=get_area_sizes)

    def get_user_count(self):
        return self._usr_count

    user_count = property(fget=get_user_count)

    def get_mo_count(self):
        return self._mo_count

    mo_count = property(fget=get_mo_count)

    def get_mos(self):
        return self._mos

    mos = property(fget=get_mos)

    def get_users(self):
        return self._users

    users = property(fget=get_users)

    def get_current_time(self):
        return self._curr_time

    current_time = property(fget=get_current_time)

    def get_ga(self):
        return self._ga

    ga = property(fget=get_ga)

    def tick(self):
        for i in range(self._mo_count):
            self._mos[i].tick()

        if self._curr_time % 1440 == 0:
            self._ga.daily()

        self._current_locations = next(self._movements_iterable)
        self._current_locations = list(map(lambda y: list(map(lambda x: int(round(x)),
                                                              self._current_locations[y])),
                                           range(len(self._current_locations))))
        self._curr_time += 1
        for i in range(self._usr_count):
            self._users[i].move_to(new_x=self._current_locations[i][0],
                                   new_y=self._current_locations[i][1],
                                   update_time=self._curr_time
                                   )
