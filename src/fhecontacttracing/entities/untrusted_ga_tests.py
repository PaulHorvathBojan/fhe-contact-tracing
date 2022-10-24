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

from user import *
from mob_operator import *
from gov_agent import *
from space_time_lord import *


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


class UserTest(unittest.TestCase):
    def test_getters(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 300,
                                big_modulus=1 << 1200,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        dummy_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                           id=0,
                                           area_side_x=50,
                                           area_side_y=50,
                                           max_x=3652,
                                           max_y=3652,
                                           encryption_params=params)

        test_user = EncryptionUserUntrustedGA(x=12,
                                              y=13,
                                              mo=dummy_mo,
                                              uid=0,
                                              ga=dummy_ga,
                                              encryption_params=params)

        self.assertEqual([test_user], dummy_mo.users, "user not properly added in MO")

        self.assertEqual([test_user], dummy_ga._users, "user not properly added in GA")

        self.assertEqual(test_user.mo, dummy_mo, "MO property no work")

        self.assertEqual(test_user.uID, 0, "id property no work")

        self.assertEqual(test_user.ga, dummy_ga, "GA property no work")

        self.assertEqual(test_user.x, 12, "x property no work")

        self.assertEqual(test_user.y, 13, "y property no work")

        self.assertEqual(test_user.last_update, 0, "last update property no work")

    def test_moveto(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 300,
                                big_modulus=1 << 1200,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        dummy_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                           id=0,
                                           area_side_x=50,
                                           area_side_y=50,
                                           max_x=3652,
                                           max_y=3652,
                                           encryption_params=params)

        test_user = EncryptionUserUntrustedGA(x=12,
                                              y=13,
                                              mo=dummy_mo,
                                              uid=0,
                                              ga=dummy_ga,
                                              encryption_params=params)

        test_user.move_to(new_x=14,
                          new_y=15)

        self.assertEqual(test_user.x, 14, "move_to does not properly set x")

        self.assertEqual(test_user.y, 15, "move_to does not properly set y")

        self.assertEqual(test_user.last_update, 1, "move_to does not properly increment local time")

        self.assertEqual(dummy_mo._curr_locations[0][0], 14, "move_to does not properly update location in MO")

    def test_scorefrommo(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 300,
                                big_modulus=1 << 1200,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        dummy_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                           id=0,
                                           area_side_x=50,
                                           area_side_y=50,
                                           max_x=3652,
                                           max_y=3652,
                                           encryption_params=params)

        test_user = EncryptionUserUntrustedGA(x=12,
                                              y=13,
                                              mo=dummy_mo,
                                              uid=0,
                                              ga=dummy_ga,
                                              encryption_params=params)

        aux_encryptor = CKKSEncryptor(params=params,
                                      public_key=test_user.pk)
        aux_evaluator = CKKSEvaluator(params=params)
        encode1234 = aux_evaluator.create_constant_plain(const=1234)
        encrypt1234 = aux_encryptor.encrypt(plain=encode1234)
        test_user.score_from_mo(encr_score=encrypt1234)

        self.assertLessEqual(abs(test_user._score - 1234), 7.9e-9, "score_from_mo no work")


class MOTest(unittest.TestCase):
    def test_getters(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 300,
                                big_modulus=1 << 1200,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        test_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                          id=0,
                                          area_side_x=50,
                                          area_side_y=50,
                                          max_x=3652,
                                          max_y=3652,
                                          encryption_params=params)

        self.assertEqual(dummy_ga._mos, [test_mo], "MO not properly added in GA")

        self.assertIsNotNone(test_mo.pk, "pk property no work")

        self.assertEqual(test_mo.user_pks, [], "user pks not properly initialized as empty")

        dummy_user = EncryptionUserUntrustedGA(x=12,
                                               y=13,
                                               mo=test_mo,
                                               uid=0,
                                               ga=dummy_ga,
                                               encryption_params=params)

        self.assertEqual(test_mo.user_pks, [dummy_user.pk], "user pks property no work")

    def test_useraddition(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 300,
                                big_modulus=1 << 1200,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        test_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                          id=0,
                                          area_side_x=50,
                                          area_side_y=50,
                                          max_x=3652,
                                          max_y=3652,
                                          encryption_params=params)

        dummy_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                           id=1,
                                           area_side_x=50,
                                           area_side_y=50,
                                           max_x=3652,
                                           max_y=3652,
                                           encryption_params=params)

        test_mo.register_other_mo(new_mo=dummy_mo)
        dummy_mo.register_other_mo(new_mo=test_mo)

        self.assertEqual(test_mo.user_pks, [], "user pks not properly initialized as empty")

        self.assertEqual(test_mo._scores, [], "score list not properly initialized as empty")

        self.assertEqual(dummy_mo._other_mo_user_pks, [[]], "pks in other MO not properly initialized")

        self.assertEqual(dummy_mo._user_relin_keys, [], "list of user relin keys not empty")

        users = []
        for i in range(10):
            users.append(EncryptionUserUntrustedGA(x=2 * i,
                                                   y=2 * i + 1,
                                                   mo=test_mo,
                                                   uid=i,
                                                   ga=dummy_ga,
                                                   encryption_params=params))

            self.assertEqual(test_mo.user_pks[-1], users[-1].pk, "pk property no work @ " + str(i))

            self.assertEqual(dummy_mo._other_mo_user_pks[0][-1], users[i].pk,
                             "pk not properly transmitted to other MO @ " + str(i))

            self.assertEqual(test_mo._user_relin_keys[-1], users[-1].relin_key,
                             "relin key list not properly updated @ " + str(i))

            aux_decr = users[-1]._decryptor.decrypt(ciphertext=test_mo._scores[i])
            aux_deco = users[-1]._encoder.decode(plain=aux_decr)[0].real
            self.assertLessEqual(abs(aux_deco), 7.9e-9, "score not properly update upon user addition")

            self.assertEqual(test_mo._usr_count, i + 1, "user count not properly updated upon user addition")

    def test_registerothermo(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 300,
                                big_modulus=1 << 1200,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        test_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                          id=0,
                                          area_side_x=50,
                                          area_side_y=50,
                                          max_x=3652,
                                          max_y=3652,
                                          encryption_params=params)

        mos = []
        for i in range(5):
            mos.append(EncryptionMOUntrustedGA(ga=dummy_ga,
                                               id=i + 1,
                                               area_side_x=50,
                                               area_side_y=50,
                                               max_x=3652,
                                               max_y=3652,
                                               encryption_params=params))
            test_mo.register_other_mo(new_mo=mos[-1])

        iter = 0
        for j in range(len(mos)):
            self.assertEqual(test_mo._other_mo_user_pks[iter], mos[j].user_pks,
                             "other MO user pks not updated upon MO addition")
            iter += 1

    def test_plainencrlocationscore(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 600,
                                big_modulus=1 << 1200,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        test_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                          id=0,
                                          area_side_x=50,
                                          area_side_y=50,
                                          max_x=3652,
                                          max_y=3652,
                                          encryption_params=params)
        i = 14
        j = 16

        plain_score = test_mo.location_pair_contact_score(location1=(0, 0),
                                                          location2=(i, j))
        encri = test_mo._encryptor.encrypt(plain=test_mo._evaluator.create_constant_plain(const=i))
        encrj = test_mo._encryptor.encrypt(plain=test_mo._evaluator.create_constant_plain(const=j))

        encr_score = test_mo.plain_encr_location_pair_contact_score(plain_location=(0, 0),
                                                                    encr_location=(encri, encrj),
                                                                    relin_key=test_mo._relin_key)
        decr_score = test_mo._decryptor.decrypt(ciphertext=encr_score)
        deco_score = test_mo._encoder.decode(plain=decr_score)

        self.assertLessEqual(abs(plain_score - deco_score[0].real), 5e-6,
                             "plain-encr score no work for " + str(i) + " " + str(j))

    def test_rcvdatafrommo(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 500,
                                big_modulus=1 << 700,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        test_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                          id=0,
                                          area_side_x=50,
                                          area_side_y=50,
                                          max_x=3652,
                                          max_y=3652,
                                          encryption_params=params)

        users = []
        for i in range(2):
            users.append(EncryptionUserUntrustedGA(x=0,
                                                   y=0,
                                                   mo=test_mo,
                                                   uid=i,
                                                   ga=dummy_ga,
                                                   encryption_params=params))

        self.assertEqual(test_mo._users, users, "inconsistent user lists")

        aux_evaluator = CKKSEvaluator(params=params)
        aux_plain00 = aux_evaluator.create_constant_plain(const=0)
        aux_plain01 = aux_evaluator.create_constant_plain(const=0)

        area_list = [(0, 0)]

        sts_list = [[]]
        for user in users:
            aux_pk = user.pk
            aux_encryptor = CKKSEncryptor(params=params,
                                          public_key=aux_pk)

            aux_encoded_sts = aux_evaluator.create_constant_plain(const=1)
            sts_list[0].append(aux_encryptor.encrypt(plain=aux_encoded_sts))

        loc_list = [[]]
        for user in users:
            aux_encryptor = CKKSEncryptor(params=params,
                                          public_key=user.pk)

            loc_list[0].append((aux_encryptor.encrypt(plain=aux_plain00), aux_encryptor.encrypt(plain=aux_plain01)))

        test_mo.rcv_data_from_mo(loc_list=loc_list,
                                 area_list=area_list,
                                 sts_list=sts_list)

        for i in range(len(test_mo._scores)):
            decr = test_mo._users[i]._decryptor.decrypt(ciphertext=test_mo._scores[i])
            deco = test_mo._users[i]._encoder.decode(plain=decr)
            self.assertLessEqual(abs(deco[0].real - 1), 0.1, "rcv data from mo no work")

    def test_senddatatomo(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 500,
                                big_modulus=1 << 700,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        test_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                          id=0,
                                          area_side_x=50,
                                          area_side_y=50,
                                          max_x=3652,
                                          max_y=3652,
                                          encryption_params=params)

        dummy_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                           id=1,
                                           area_side_x=50,
                                           area_side_y=50,
                                           max_x=3652,
                                           max_y=3652,
                                           encryption_params=params)

        users = []
        for i in range(10):
            users.append(EncryptionUserUntrustedGA(x=i * 50,
                                                   y=i * 50,
                                                   mo=test_mo,
                                                   uid=i,
                                                   ga=dummy_ga,
                                                   encryption_params=params))
        for i in range(10):
            users.append(EncryptionUserUntrustedGA(x=i * 50,
                                                   y=i * 50,
                                                   mo=dummy_mo,
                                                   uid=i + 10,
                                                   ga=dummy_ga,
                                                   encryption_params=params))

        dummy_ga._status = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dummy_ga.daily()# This low-key assumes uGA.daily() works, which in turn assumes ga.sts_to_mo works, which in turn assumes mo.from_ga_comm works

        dummy_mo.send_data_to_mo(other_mo=test_mo)

        for i in range(len(test_mo._scores)):
            decr = test_mo._users[i]._decryptor.decrypt(ciphertext=test_mo._scores[i])
            deco = test_mo._users[i]._encoder.decode(plain=decr)
            self.assertLessEqual(abs(deco[0].real - 1), 0.1, "send data to mo no work")

    def test_fromgacomm(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 500,
                                big_modulus=1 << 700,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        test_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                          id=0,
                                          area_side_x=50,
                                          area_side_y=50,
                                          max_x=3652,
                                          max_y=3652,
                                          encryption_params=params)

        users = []
        for i in range(10):
            users.append(EncryptionUserUntrustedGA(x=i,
                                                   y=i,
                                                   mo=test_mo,
                                                   uid=i,
                                                   ga=dummy_ga,
                                                   encryption_params=params))

        sts_list = []
        aux_sts = []
        for user in users:
            sts_list.append([])
            aux_sts.append(user.uID ** 2)
        aux_ev = CKKSEvaluator(params=params)
        for user in users:
            aux_encryptor = CKKSEncryptor(params=params,
                                          public_key=user.pk)
            for i in range(len(sts_list)):
                aux_plain = aux_ev.create_constant_plain(const=aux_sts[i])
                sts_list[i].append(aux_encryptor.encrypt(plain=aux_plain))

        test_mo.from_ga_comm(new_status=sts_list)

        for i in range(test_mo._usr_count):
            decr_Sts = test_mo._users[i]._decryptor.decrypt(ciphertext=test_mo._status[i][i])
            deco_sts = test_mo._encoder.decode(plain=decr_Sts)
            self.assertLessEqual(abs(deco_sts[0].real - i ** 2), 8e-9, "improper decoded-decrypted status @ " + str(i))

    def test_insidescoring(self):
        params = CKKSParameters(poly_degree=4,
                                ciph_modulus=1 << 600,
                                big_modulus=1 << 800,
                                scaling_factor=1 << 30
                                )

        dummy_ga = EncryptionUntrustedGA(encryption_params=params)

        test_mo = EncryptionMOUntrustedGA(ga=dummy_ga,
                                          id=0,
                                          area_side_x=50,
                                          area_side_y=50,
                                          max_x=3652,
                                          max_y=3652,
                                          encryption_params=params)

        users = []
        for i in range(10):
            users.append(EncryptionUserUntrustedGA(x=0,
                                                   y=0,
                                                   mo=test_mo,
                                                   uid=i,
                                                   ga=dummy_ga,
                                                   encryption_params=params))

        sts_list = []
        aux_sts = []
        for user in users:
            sts_list.append([])
            aux_sts.append(0)
        aux_sts[-1] = 1
        aux_ev = CKKSEvaluator(params=params)
        for user in users:
            aux_encryptor = CKKSEncryptor(params=params,
                                          public_key=user.pk)
            for i in range(len(sts_list)):
                aux_plain = aux_ev.create_constant_plain(const=aux_sts[i])
                sts_list[i].append(aux_encryptor.encrypt(plain=aux_plain))

        test_mo.from_ga_comm(new_status=sts_list)
        test_mo.inside_scoring()

        for i in range(test_mo._usr_count - 1):
            decr = test_mo._users[i]._decryptor.decrypt(ciphertext=test_mo._scores[i])
            deco = test_mo._users[i]._encoder.decode(plain=decr)
            self.assertLessEqual(abs(deco[0].real - 1), 0.1, "inside score improper @ " + str(i))
        decr = test_mo._users[-1]._decryptor.decrypt(ciphertext=test_mo._scores[-1])
        deco = test_mo._users[-1]._encoder.decode(plain=decr)
        self.assertLessEqual(abs(deco[0].real), 0.1, "inside score improper @ 10")


unittest.main()
