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


class ProtocolUser:
    # Initialize user at a certain location with a certain MO, user ID and GA.
    # After initializing class fields to the parameter and default values respectively, call add_user in the
    #   user's MO and the GA.
    # By default:
    #    - the user's last update is at tick 0
    #    - the user is not infected
    #    - the user's infection time is 0
    def __init__(self, init_x, init_y, mo, uid, ga):
        self._MO = mo
        self._uID = uid
        self._GA = ga
        self._threshold = self._GA.risk_threshold

        self._x = init_x
        self._y = init_y
        self._last_update = 0

        # intuitively going for a Susceptible(0)-Infected(1) model;
        # recovery is beyond the scope of this project
        self._score = 0
        self._nonce = 0
        self._risk = 0

        self._MO.add_user(self)
        self._GA.add_user(self)

    # Defined properties from getters for x, y, MO, GA, last update time and infection status.
    # No setters, as the respective fields are either immutable or implicitly changed by other methods.
    def get_x(self):
        return self._x

    x = property(fget=get_x)

    def get_y(self):
        return self._y

    y = property(fget=get_y)

    def get_mo(self):
        return self._MO

    MO = property(fget=get_mo)

    def get_uid(self):
        return self._uID

    uID = property(fget=get_uid)

    def get_ga(self):
        return self._GA

    GA = property(fget=get_ga)

    def get_last_update_time(self):
        return self._last_update

    last_update = property(fget=get_last_update_time)

    # move_to updates the current location of the user and the time of the last update.
    # The final part produces a Bernoulli variable that models the chance to ask the MO for the score:
    #   - the chance to ask for score is 1/(the number of seconds in 2 days)
    def move_to(self, new_x, new_y, update_time):
        if update_time > self._last_update:
            self._x = new_x
            self._y = new_y
            self._last_update = update_time

            ask_score = bernoulli.rvs(0.00034722222)  # bernoulli argument is 1/2880, which attempts to model a request
            if ask_score == 1:
                self.score_receipt_proc()

    # upd_to_mo updates the data that the MO has regarding the self
    # it essentially calls a function in the MO that performs the updating
    def upd_to_mo(self):
        self._MO.upd_user_data(user=self,
                               new_x=self.x,
                               new_y=self.y
                               )
        # Essentially in a setting that puts users in a database, this would be needed to:
        #   - update user locations in the db
        #   - trigger any eventual events inside the MO class

    # score_from_mo models receipt of score from MO during the MO to user comm phase
    # It sets the score field of the user to the score argument
    def score_from_mo(self, score):
        self._score = score

    # ping_mo_for_score models the request for score value from the user to the MO
    # It calls the rcv_user_ping method inside the MO object that the user is affiliated with
    def ping_mo_for_score(self):
        self._MO.rcv_user_ping(self)

    # decr_score_from_ga models the user to GA comm phase
    # It multiplies the score with a random integer between 2 and 2 ** 60 and sends it as argument for the
    #   appropriate method inside the GA class.
    def decr_score_from_ga(self):
        self._nonce = random.randint(2, 2 ** 60)
        self._GA.score_req(self, self._nonce * self._score)

    # rcv_score_from_ga models receipt of score from the affiliated GA
    # The received value is the score masked multiplicatively with self._nonce.
    # The nonce is removed and the score is updated.
    def score_from_ga(self, nonced_score):
        self._score = nonced_score / self._nonce
        if self._score >= self._threshold:
            self.at_risk()

    # at_risk sets the value of _risk to 1
    def at_risk(self):
        self._risk = 1

    # score_receipt_proc models the score receipt procedure
    # There are a number of methods in various classes that call each other once certain criteria are met.
    # ProtocolUser.score_receipt_proc ->
    # ProtocolUser.ping_mo_for_score ->
    # MobileOperator.rcv_ping_from_user ->
    # MobileOperator.to_user_comm ->
    # ProtocolUser.score_from_mo
    # Then, the GA-related side of comms is initiated:
    # call ProtocolUser.decr_score_from_ga ->
    # GA.rcv_score_req_from_user ->
    # ProtocolUser.rcv_score_from_GA
    def score_receipt_proc(self):
        self.ping_mo_for_score()
        self.decr_score_from_ga()


class EncryptedUser(ProtocolUser):

    def __init__(self, init_x, init_y, mo, uid, ga):
        self._evaluator = None
        self._encryptor = None
        self._encoder = None
        self._scaling_factor = None
        self._relin_key = None
        self._public_key = None
        self._encr_score = None

        super(EncryptedUser, self).__init__(init_x, init_y, mo, uid, ga)

    def get_encoder(self):
        return self._encoder

    encoder = property(fget=get_encoder)

    def get_evaluator(self):
        return self._evaluator

    evaluator = property(fget=get_evaluator)

    def get_encryptor(self):
        return self._encryptor

    encryptor = property(fget=get_encryptor)

    def get_encr_score(self):
        return self._encr_score

    encr_score = property(fget=get_encr_score)

    def get_scaling_factor(self):
        return self._scaling_factor

    scaling_factor = property(fget=get_scaling_factor)

    def get_relin_key(self):
        return self._relin_key

    relin_key = property(fget=get_relin_key)

    def get_public_key(self):
        return self._public_key

    public_key = property(fget=get_public_key)

    def score_from_mo(self, score):
        self._encr_score = score

    def decr_score_from_ga(self):
        self._nonce = random.randint(2, 2 ** 60)
        enco_nonce = self._evaluator.create_constant_plain(const=self._nonce)

        nonced_score = self._evaluator.multiply_plain(ciph=self._encr_score,
                                                      plain=enco_nonce)

        self._GA.score_req(self, nonced_score)

    def first_time_fhe_setup(self, new_evaluator, new_encryptor, new_encoder, new_scaling_factor, new_relin_key,
                             new_public_key):
        self._encryptor = new_encryptor
        self._encoder = new_encoder
        self._scaling_factor = new_scaling_factor
        self._relin_key = new_relin_key
        self._public_key = new_public_key
        self._evaluator = new_evaluator

        self._encr_score = self._encryptor.encrypt(plain=self._evaluator.create_constant_plain(const=0))

    def refresh_fhe_keys(self, new_encryptor, new_relin_key, new_public_key):
        self._encryptor = new_encryptor
        self._relin_key = new_relin_key
        self._public_key = new_public_key
        self._encr_score = self._evaluator.switch_key(ciph=self._encr_score,
                                                      key=self._public_key)


class EncryptionUserUntrustedGA:
    def __init__(self, x, y, mo, uid, ga, encryption_params):
        self._MO = mo
        self._uID = uid
        self._GA = ga
        self._threshold = self._GA.risk_threshold

        self._x = x
        self._y = y
        self._last_update = 0

        # intuitively going for a Susceptible(0)-Infected(1) model;
        # recovery is beyond the scope of this project
        self._score = 0
        self._nonce = 0
        self._risk = 0

        self._CKKSParams = encryption_params
        self._keygen = CKKSKeyGenerator(params=self._CKKSParams)
        self._pk = self._keygen.public_key
        self._sk = self._keygen.secret_key
        self._relin_key = self._keygen.relin_key
        self._evaluator = CKKSEvaluator(params=self._CKKSParams)
        self._encryptor = CKKSEncryptor(params=self._CKKSParams,
                                        public_key=self._pk,
                                        secret_key=self._sk)
        self._decryptor = CKKSDecryptor(params=self._CKKSParams,
                                        secret_key=self._sk)
        self._encoder = CKKSEncoder(params=self._CKKSParams)

        self._MO.add_user(self)
        self._GA.add_user(self)

    @property
    def mo(self):
        return self._MO

    @property
    def uID(self):
        return self._uID

    @property
    def ga(self):
        return self._GA

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def last_update(self):
        return self._last_update

    @property
    def pk(self):
        return self._pk

    @property
    def relin_key(self):
        return self._relin_key

    def move_to(self, new_x, new_y):
        self._x = new_x
        self._y = new_y
        self._last_update += 1

        self.upd_to_mo()

    def upd_to_mo(self):
        self._MO.upd_user_data(user=self,
                               new_x=self.x,
                               new_y=self.y
                               )

    def ping_mo_for_score(self):
        self._MO.rcv_user_ping(user=self)

    def score_from_mo(self, score):
        plain_score = self._decryptor.decrypt(ciphertext=score)
        self._score = self._encoder.decode(plain=plain_score)[0].real
        # in case all the 128 positions are filled up with actual values, this might be subject to change
