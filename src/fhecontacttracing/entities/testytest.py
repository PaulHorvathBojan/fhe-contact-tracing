import numpy as np
import math
import cmath
import os
import unittest

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

from pymobility.models.mobility import gauss_markov


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


class Itertest(unittest.TestCase):

    def test_functionality(self):
        for _ in range(10):
            dummy_gm = gauss_markov(nr_nodes=10,
                                    dimensions=(3652, 3652),
                                    velocity_mean=7.,
                                    alpha=.5,
                                    variance=7.)

            dual = DualIter(init_iter=dummy_gm)

            for _ in range(10):
                fst = next(dual)
                snd = next(dual)

                for i in range(len(fst)):
                    self.assertEqual(fst[i][0], snd[i][0], "Dual iterator no work")
                    self.assertEqual(fst[i][1], snd[i][1], "Dual iterator no work")


class EncryptionLibraryTest(unittest.TestCase):
    def test_encodedecode(self):
        poly_deg = 2
        scaling_fact = 1 << 30
        big_mod = 1 << 1200

        ciph_mod = 1 << 300
        params = CKKSParameters(poly_degree=poly_deg,
                                ciph_modulus=ciph_mod,
                                big_modulus=big_mod,
                                scaling_factor=scaling_fact)

        encoder = CKKSEncoder(params=params)

        for val in range(3653):
            e1 = encoder.encode(values=[val],
                                scaling_factor=scaling_fact)
            e2 = encoder.encode(values=[val + 0j],
                                scaling_factor=scaling_fact)
            e3 = encoder.encode(values=[complex(val, 0)],
                                scaling_factor=scaling_fact)

            self.assertEqual([val], encoder.decode(plain=e1), "encoder no work")
            self.assertEqual([val], encoder.decode(plain=e2), "encoder no work")
            self.assertEqual([val], encoder.decode(plain=e3), "encoder no work")

    def test_encodeencryptdecryptdecode(self):
        poly_deg = 2
        scaling_fact = 1 << 30
        big_mod = 1 << 1200

        ciph_mod = 1 << 300
        params = CKKSParameters(poly_degree=poly_deg,
                                ciph_modulus=ciph_mod,
                                big_modulus=big_mod,
                                scaling_factor=scaling_fact)

        keygen = CKKSKeyGenerator(params=params)
        pk = keygen.public_key
        sk = keygen.secret_key

        encoder = CKKSEncoder(params=params)

        encryptor = CKKSEncryptor(params=params,
                                  public_key=pk,
                                  secret_key=sk)

        decryptor = CKKSDecryptor(params=params,
                                  secret_key=sk)

        for val in range(3653):
            e1 = encoder.encode(values=[val],
                                scaling_factor=scaling_fact)
            e2 = encoder.encode(values=[val + 0j],
                                scaling_factor=scaling_fact)
            e3 = encoder.encode(values=[complex(val, 0)],
                                scaling_factor=scaling_fact)

            ee1 = encryptor.encrypt(plain=e1)
            ee2 = encryptor.encrypt(plain=e2)
            ee3 = encryptor.encrypt(plain=e3)

            eed1 = decryptor.decrypt(ciphertext=ee1)
            eed2 = decryptor.decrypt(ciphertext=ee2)
            eed3 = decryptor.decrypt(ciphertext=ee3)

            eedd1 = encoder.decode(plain=eed1)
            eedd2 = encoder.decode(plain=eed2)
            eedd3 = encoder.decode(plain=eed3)

            self.assertLessEqual(val - eedd1[0].real, 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
            self.assertLessEqual(abs(eedd1[0].imag), 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
            self.assertLessEqual(val - eedd2[0].real, 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
            self.assertLessEqual(abs(eedd2[0].imag), 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
            self.assertLessEqual(val - eedd3[0].real, 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
            self.assertLessEqual(abs(eedd3[0].imag), 3e-9, "encode-encrypt-decrypt-decode pipeline no work")



unittest.main()
