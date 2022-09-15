import numpy as np
import math
import cmath
import os
import unittest
import random

from scipy.stats import bernoulli
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


class MinutelyMovement:
    def __init__(self, movement_iter):
        self._move = movement_iter

    def __next__(self):
        for i in range(60):
            self_next = next(self._move)

        return self_next


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
        enco_nonce = self._encoder.encode(values=[complex(self._nonce, 0)],
                                          scaling_factor=self._scaling_factor)

        nonced_score = self._evaluator.multiply_plain(ciph=self._encr_score,
                                                      plain=enco_nonce)

        self._GA.score_req(self, nonced_score)

    def set_new_fhe_suite(self, new_evaluator, new_encryptor, new_encoder, new_scaling_factor, new_relin_key,
                          new_public_key):
        self._encryptor = new_encryptor
        self._encoder = new_encoder
        self._scaling_factor = new_scaling_factor
        self._relin_key = new_relin_key
        self._public_key = new_public_key
        if self._evaluator is not None and self._encr_score is not None:
            self._encr_score = self._evaluator.switch_key(ciph=self._encr_score,
                                                          key=self._public_key)
        self._evaluator = new_evaluator


class GovAgent:

    # initialise the class with empty user list, MO list, infected list, status list, score list
    def __init__(self, risk_threshold):
        self._risk_threshold = risk_threshold

        self._users = []
        self._MOs = []
        self._infect_list = []
        self._status = []
        self._scores = []

    # define getters for possibly public fields
    def get_mos(self):
        return self._MOs

    MOs = property(fget=get_mos)

    def get_users(self):
        return self._users

    users = property(fget=get_users)

    def get_infected(self):
        return self._infect_list

    infected = property(fget=get_infected)

    def get_risk_threshold(self):
        return self._risk_threshold

    risk_threshold = property(fget=get_risk_threshold)

    # add_user appends a new user to the user list
    # Additionally, it appends its score and status to the respective lists
    def add_user(self, new_user):
        self._users.append(new_user)
        self._status.append(0)
        self._scores.append(0)

    # add_mo registers a reference to a new MO.
    # It appends the new_mo object to the local list of MOs.
    def add_mo(self, new_mo):
        self._MOs.append(new_mo)

    # daily performs the daily status-score exchange routine between the GA and the MOs
    # It is called by the SpaceTimeLord when the timer reaches a multiple of 1440 (nr of minutes in a day).
    # It iterates through the local list of MOs
    # It fetches the global indices of all users of the respective MO.
    # It initialises an empty status list for every MO and appends every user's respective infection status
    #   to the status list.
    # It calls the appropriate method in the MO object with the status list as argument.
    def daily(self):
        for mo in self._MOs:
            index_list = []
            for user in mo.users:
                index_list.append(user.uID)

            status_list = []
            for i in index_list:
                status_list.append(self._status[i])

            mo.from_ga_comm(new_status=status_list)

    # rcv_scores models the MO to GA daily communication of user infection risk scores.
    # The GA receives from the MO a list of (uID, risk score) tuples.
    # At GA-level, every score of the respective user is updated with the current value.
    # Then, if the user's current score is higher than the risk_threshold, the user's ID is appended to a new
    #   risk auxiliary list.
    # After the loop ends, all users in the risk auxiliary list are advised.
    def rcv_scores(self, from_mo_package):

        risk_aux = []
        for tup in from_mo_package:
            self._scores[tup[0]] = tup[1]
            if tup[1] >= self.risk_threshold and self._status[tup[0]] == 0:
                risk_aux.append(self._users[tup[0]].uID)

        for uID in risk_aux:
            self._users[uID].at_risk()

    # Method to be overloaded in a cryptographic implementation
    # In crypto world, the GA would decrypt the received value from the user
    # AND THEN
    # send the result back to the user
    # It is assumed that the user would send the value masked under a nonce
    def score_req(self, user, nonced_score):
        user.score_from_ga(nonced_score)

    # sts_from_user updates the user's status inside the GA's respective list of user status
    def status_from_user(self, user, status):
        if status == 1:
            self._status[user.uID] = status
        elif status == 0:
            self._status[user.uID] = status


class EncryptionGovAgent(GovAgent):
    def __init__(self, risk_threshold, degree, cipher_modulus, big_modulus, scaling_factor):
        super(EncryptionGovAgent, self).__init__(risk_threshold=risk_threshold)

        self._ckks_params = CKKSParameters(poly_degree=degree,
                                           ciph_modulus=cipher_modulus,
                                           big_modulus=big_modulus,
                                           scaling_factor=scaling_factor
                                           )

        self._key_generator = CKKSKeyGenerator(self._ckks_params)

        self._public_key = self._key_generator.public_key
        self._secret_key = self._key_generator.secret_key
        self._relin_key = self._key_generator.relin_key
        self._scaling_factor = scaling_factor

        self._encoder = CKKSEncoder(self._ckks_params)
        self._encryptor = CKKSEncryptor(self._ckks_params, self._public_key, self._secret_key)
        self._decryptor = CKKSDecryptor(self._ckks_params, self._secret_key)
        self._evaluator = CKKSEvaluator(self._ckks_params)

    def get_public_key(self):
        return self._public_key

    public_key = property(fget=get_public_key)

    def get_relin_key(self):
        return self._relin_key

    relin_key = property(fget=get_relin_key)

    def get_encoder(self):
        return self._encoder

    encoder = property(fget=get_encoder)

    def get_encryptor(self):
        return self._encryptor

    encryptor = property(fget=get_encryptor)

    def get_evaluator(self):
        return self._evaluator

    evaluator = property(fget=get_evaluator)

    def get_scaling_factor(self):
        return self._scaling_factor

    scaling_factor = property(fget=get_scaling_factor)

    def new_encryption_suite(self):
        self._key_generator = CKKSKeyGenerator(self._ckks_params)

        self._public_key = self._key_generator.public_key
        self._secret_key = self._key_generator.secret_key
        self._relin_key = self._key_generator.relin_key

        self._encoder = CKKSEncoder(self._ckks_params)
        self._encryptor = CKKSEncryptor(self._ckks_params, self._public_key, self._secret_key)
        self._decryptor = CKKSDecryptor(self._ckks_params, self._secret_key)
        self._evaluator = CKKSEvaluator(self._ckks_params)

        self.distribute_fhe_suite()

    def distribute_fhe_suite(self):
        for mo in self._MOs:
            mo.set_new_fhe_suite(new_evaluator=self._evaluator,
                                 new_encryptor=self._encryptor,
                                 new_encoder=self._encoder,
                                 new_scaling_factor=self._scaling_factor,
                                 new_relin_key=self._relin_key,
                                 new_public_key=self._public_key)

        for user in self._users:
            user.set_new_fhe_suite(new_evaluator=self._evaluator,
                                   new_encryptor=self._encryptor,
                                   new_encoder=self._encoder,
                                   new_scaling_factor=self._scaling_factor,
                                   new_relin_key=self._relin_key,
                                   new_public_key=self._public_key)

    def add_user(self, new_user):
        super(EncryptionGovAgent, self).add_user(new_user)

        new_user.set_new_fhe_suite(new_evaluator=self._evaluator,
                                   new_encryptor=self._encryptor,
                                   new_encoder=self._encoder,
                                   new_scaling_factor=self._scaling_factor,
                                   new_relin_key=self._relin_key,
                                   new_public_key=self._public_key)

    def add_mo(self, new_mo):
        super(EncryptionGovAgent, self).add_mo(new_mo)

        new_mo.set_new_fhe_suite(new_evaluator=self._evaluator,
                                 new_encryptor=self._encryptor,
                                 new_encoder=self._encoder,
                                 new_scaling_factor=self._scaling_factor,
                                 new_relin_key=self._relin_key,
                                 new_public_key=self._public_key)

    def daily(self):
        for mo in self._MOs:
            index_list = []
            for user in mo.users:
                index_list.append(user.uID)

            status_list = []
            for i in index_list:
                pre_encode_list = [complex(self._status[i], 0)]
                encoded_sts = self._encoder.encode(values=pre_encode_list,
                                                   scaling_factor=self._scaling_factor)
                encrypted_sts = self._encryptor.encrypt(plain=encoded_sts)
                status_list.append(encrypted_sts)

            mo.from_ga_comm(new_status=status_list)

    def rcv_scores(self, from_mo_package):

        for tup in from_mo_package:
            decr = self._decryptor.decrypt(ciphertext=tup[1])
            deco = self._encoder.decode(plain=decr)
            score = deco[0].real

            self._scores[tup[0]] = score
            if score >= self.risk_threshold and self._status[tup[0]] == 0:
                self._users[tup[0]].at_risk()

    def score_req(self, user, encr_score):
        decr_score = self._decryptor.decrypt(ciphertext=encr_score)
        deco_score = self._encoder.decode(plain=decr_score)
        plain_score = deco_score[0].real

        user.score_from_ga(plain_score)


class MobileOperator:
    # Dictionary for edge case adjacency.
    # Used in det_adj_area_ranges.
    edge_case_range_dict = {0: ([-1, 0, 1], [-1, 0, 1]),
                            1: ([0, 1], [-1, 0, 1]),
                            2: ([-1, 0, 1], [0, 1]),
                            3: ([0, 1], [0, 1]),
                            4: ([-1, 0], [-1, 0, 1]),
                            6: ([-1, 0], [0, 1]),
                            7: ([-1, 0, 1], [-1, 0]),
                            8: ([0, 1], [-1, 0]),
                            11: ([-1, 0], [-1, 0])
                            }

    # Initialize MobileOperator as having given GovAgent, ID, and area sides
    # By default:
    #   - _usr_count tracks number of users (init 0)
    #   - _users keeps a reference list of users (init empty)
    #   - _curr_locations keeps a list of location tuples for each user
    #           respectively --- same order as _users (init empty)
    #   - _scores keeps a list of user risk scores respectively --- same order as _users (init empty)
    #   - _area_sides keep the sizes of the considered tesselation area
    #   - _curr_time is 0
    #   - _area_array is a 23000 x 19000 list of lists of empty sets --- roughly the size of Italy if
    #       divided into 50 x 50 m squares
    def __init__(self, ga, mo_id, area_side_x, area_side_y, max_x, max_y):
        self._GA = ga
        self._id = mo_id
        self._area_side_x = area_side_x
        self._area_side_y = area_side_y
        self._L_max = 2 * max(self._area_side_y, self._area_side_x)
        self._area_count_x = int(max_x // area_side_x) + 1
        self._area_count_y = int(max_y // area_side_y) + 1

        self._usr_count = 0
        self._users = []
        self._curr_locations = []
        self._scores = []
        self._curr_areas_by_user = []
        self._area_array = []
        self._status = []
        self._curr_time = 0

        self._other_mos = []

        for _ in range(self._area_count_x):
            ys = []
            for _ in range(self._area_count_y):
                ys.append(set())
            self._area_array.append(ys)

        self._GA.add_mo(self)

    # define getters for all potentially public attributes:
    #   - govAuth
    def get_ga(self):
        return self._GA

    GA = property(fget=get_ga)

    #   - user list
    def get_users(self):
        return self._users

    users = property(fget=get_users)

    #   - self ID
    def get_id(self):
        return self._id

    id = property(fget=get_id)

    #   - user count
    def get_usr_count(self):
        return self._usr_count

    usr_count = property(fget=get_usr_count)

    #   - area sides
    def get_area_side_x(self):
        return self._area_side_x

    area_side_x = property(fget=get_area_side_x)

    def get_area_side_y(self):
        return self._area_side_y

    area_side_y = property(fget=get_area_side_y)

    #   - max of the two area sides
    def get_lmax(self):
        return self._L_max

    L_max = property(fget=get_lmax)

    #   - time of last update
    def get_curr_time(self):
        return self._curr_time

    curr_time = property(fget=get_curr_time)

    #   - list of other MOs
    def get_other_mos(self):
        return self._other_mos

    other_MOs = property(fget=get_other_mos)

    # register_other_mo creates reference to another MobileOperators for joint contact tracing purposes
    def register_other_mo(self, new_mo):
        self._other_mos.append(new_mo)

    # search_user_db will act as an auxiliary function to perform more efficient searches
    # inside the list of users internally
    # The first tiny assertion is that the users are added to the internal db in order of IDs.
    # The second tiny assertion is that no one will ever try ever to search for things that are not there.
    # In any case, an error is thrown if the user is not in the db.
    def search_user_db(self, user):
        left = 0
        right = self._usr_count - 1

        while left <= right:
            mid = (left + right) // 2
            if self._users[mid].uID < user.uID:
                left = mid + 1
            elif self._users[mid].uID > user.uID:
                right = mid - 1
            else:
                return mid

        return -1

    # add_user adds a new user to the _users list
    # It increments _usr_count and updates the _curr_locations, _curr_areas, _scores, and _status
    #   lists accordingly:
    #   - _curr_locations gets the user's location appended
    #   - _curr_areas gets the appropriate area appended
    #   - _scores gets 0 appended
    #   - _status gets 0 appended
    def add_user(self, user):
        self._users.append(user)

        self._curr_locations.append((np.rint(user.x), np.rint(user.y)))

        area_aux = self.assign_area(loc_tuple=self._curr_locations[-1])
        loc_bucket = self._area_array[area_aux[0]][area_aux[1]]
        loc_bucket.add(self.usr_count)

        self._curr_areas_by_user.append(area_aux)

        self._scores.append(0)

        self._status.append(0)

        self._usr_count += 1

    # assign_area assigns to a location an area code for running contact tracing in neighbouring areas only
    def assign_area(self, loc_tuple):
        return int(loc_tuple[0] / self._area_side_x), int(loc_tuple[1] / self._area_side_y)

    # det_adj_area_ranges determines adjacency area ranges
    # Essentially, the ranges are [-1, 0, 1] on both x and y axes by default.
    # In edge cases, some numbers are excluded from the ranges:
    #   - -1 in the case of left-x, upper-y
    #   - 1 in the case of right-x, lower-y
    def det_adj_area_ranges(self, area_tuple):
        edge_area_aux = 0

        if area_tuple[0] == 0:
            edge_area_aux += 1
        elif area_tuple[0] == len(self._area_array) - 1:
            edge_area_aux += 4

        if area_tuple[1] == 0:
            edge_area_aux += 2
        elif area_tuple[1] == len(self._area_array[0]) - 1:
            edge_area_aux += 7

        return self.edge_case_range_dict[edge_area_aux]

    # upd_user_data records changes in new user locations
    # It changes self._current_locations to fit the location in the respective user class
    # It changes area assignment:
    #   - pops the user from the prior self._area_array bucket
    #   - updates the user's value in self._curr_areas_by_user
    #   - pushes the user's index in the appropriate self._area_array bucket
    def upd_user_data(self, user, new_x, new_y):
        # Find user index in local indexing
        user_index = self.search_user_db(user)

        # Remove user index from prior area bucket
        prior_area = self._curr_areas_by_user[user_index]
        self._area_array[prior_area[0]][prior_area[1]].remove(user_index)

        # Update current user location
        self._curr_locations[user_index] = (new_x, new_y)

        # Add user index to current area bucket
        post_area = self.assign_area(loc_tuple=(new_x, new_y))
        self._area_array[post_area[0]][post_area[1]].add(user_index)

        # Change current user area
        self._curr_areas_by_user[user_index] = post_area

    # location_pair_contacts generates contact approx score between 2 locations
    # For alternate formulae, create a class inheriting MobileOperator and overload this method.
    def location_pair_contact_score(self, location1, location2):
        return (1 - (
                (location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2) / self._L_max ** 2) ** 2048

    # search_mo_db is an aux function for binarily searching for MOs in the MO list
    # The first tiny assertion is that the MOs are added to the internal db in order of IDs.
    # The second tiny assertion is that no one will ever try ever to search for things that are not there.
    # In any case, an error is thrown if the MO is not in the db.
    def search_mo_db(self, mo):
        left = 0
        right = len(self._other_mos) - 1

        while left <= right:
            mid = (left + right) // 2
            if self._other_mos[mid].id < mo.id:
                left = mid + 1
            elif self._other_mos[mid].id > mo.id:
                right = mid - 1
            else:
                return mid

        return -1

    # rcv_data_from_mo models data transfer from the other mobile operators
    # The other_mo MO is searched for in the self._other_mos local MO list and its index is returned
    # The loc_list location list, area_list area list, and sts_list status list are iterated over in order.
    #   ---assumption--- All the said lists are of the same length and every respective entry is related
    #            to the same respective user (i.e. if loc_list[i] has the location of user_johnathan, then
    #            area_list[i] has its area and sts_list[i] its infection status).
    # The score of the users is updated as contacts are registered:
    #   - extract incoming user location from loc_list
    #   - iterate over adjacent location areas
    #   - extract own user buckets from all adjacent areas
    #   - calculate contact score with all users in buckets
    #   - multiply contact scores with infection status and add to each own user score respectively
    def rcv_data_from_mo(self, loc_list, area_list, sts_list):
        for i in range(len(loc_list)):
            adj_indices = self.det_adj_area_ranges(area_tuple=area_list[i])

            for j in adj_indices[0]:
                for k in adj_indices[1]:
                    curr_bucket = self._area_array[area_list[i][0] + j][area_list[i][1] + k]
                    for user_index in curr_bucket:
                        self._scores[user_index] += sts_list[i] * self.location_pair_contact_score(
                            location1=loc_list[i],
                            location2=self._curr_locations[user_index]
                        )

    # send_data_to_mo is a mask for calling rcv_data_from_mo in the argument MO
    def send_data_to_mo(self, other_mo):
        other_mo.rcv_data_from_mo(loc_list=self._curr_locations,
                                  area_list=self._curr_areas_by_user,
                                  sts_list=self._status
                                  )

    # inside_scoring registers scores of the MO's own users
    # It iterates through all users
    # fetches their area from the list of areas
    # determines possible adjacent areas
    # fetches all other users from the adjacent areas
    # adds other user status * contact score to the current user's score
    def inside_scoring(self):
        for i in range(len(self._users)):
            area = self._curr_areas_by_user[i]
            adj_indices = self.det_adj_area_ranges(area_tuple=area)

            for j in adj_indices[0]:
                for k in adj_indices[1]:
                    curr_bucket = self._area_array[area[0] + j][area[1] + k]

                    for user_index in curr_bucket:
                        if not user_index == i:
                            self._scores[i] += self._status[user_index] * self.location_pair_contact_score(
                                location1=self._curr_locations[i],
                                location2=self._curr_locations[user_index])

    # tick models the passage of time inside the MO, much like the homonymous function does in SpaceTimeLord
    # It triggers receipt of location data from self users and the sending of self user data to the other
    #   MOs.
    def tick(self):
        for user in self._users:
            user.upd_to_mo()
        self.inside_scoring()

        for other_mo in self._other_mos:
            self.send_data_to_mo(other_mo)

        self._curr_time += 1

    # to_ga_comm models communications to the GA
    # It builds a list to_ga_package that contains tuples of (uID, score)-type, where uID is the global uID.
    # It then calls the receipt function inside the GA.
    def to_ga_comm(self):
        to_ga_package = []
        for i in range(len(self._users)):
            to_ga_package.append((self._users[i].uID, self._scores[i]))
        self._GA.rcv_scores(to_ga_package)

    # from_ga_comm models incoming communications from the GA
    # The new_status argument replaces the _status field inside the MO class.
    # Then, to_ga_comm is triggered to send data to the GA from the current MO.
    def from_ga_comm(self, new_status):
        self._status = new_status

        self.to_ga_comm()

    # rcv_user_ping models receipt of user request for score
    # As soon as the user's request is received, the procedure of transmitting the score stored MO-side to
    #   the respective user is started.
    # That is, the method calls the to_user_comm method.
    def rcv_user_ping(self, user):
        self.to_user_comm(user=user)

    # to_user_comm models the MO-user score transmission phase
    # It fetches the index of the user from the local _users list.
    # It sends the self._scores[index] score to the user via the user's score_from_mo method.
    def to_user_comm(self, user):
        index = self.search_user_db(user=user)

        user.score_from_mo(score=self._scores[index])


class EncryptionMO(MobileOperator):

    def __init__(self, ga, mo_id, area_side_x, area_side_y, max_x, max_y):
        self._evaluator = None
        self._public_key = None
        self._relin_key = None
        self._scaling_factor = None
        self._encoder = None
        self._encryptor = None

        super(EncryptionMO, self).__init__(ga, mo_id, area_side_x, area_side_y, max_x, max_y)

    def get_encoder(self):
        return self._encoder

    encoder = property(fget=get_encoder)

    def get_evaluator(self):
        return self._evaluator

    evaluator = property(fget=get_evaluator)

    def get_encryptor(self):
        return self._encryptor

    encryptor = property(fget=get_encryptor)

    def get_scaling_factor(self):
        return self._scaling_factor

    scaling_factor = property(fget=get_scaling_factor)

    def get_relin_key(self):
        return self._relin_key

    relin_key = property(fget=get_relin_key)

    def get_public_key(self):
        return self._public_key

    public_key = property(fget=get_public_key)

    def set_new_fhe_suite(self, new_evaluator, new_encryptor, new_encoder, new_scaling_factor, new_relin_key,
                          new_public_key):
        self._encryptor = new_encryptor
        self._encoder = new_encoder
        self._scaling_factor = new_scaling_factor
        self._relin_key = new_relin_key
        self._public_key = new_public_key
        if self._evaluator is not None:
            for score in self._scores:
                self._evaluator.switch_key(ciph=score,
                                           key=self._public_key)
        self._evaluator = new_evaluator

    def add_user(self, user):
        self._users.append(user)

        self._curr_locations.append((user.x, user.y))

        area_aux = self.assign_area(loc_tuple=self._curr_locations[-1])
        loc_bucket = self._area_array[area_aux[0]][area_aux[1]]
        loc_bucket.add(self.usr_count)

        self._curr_areas_by_user.append(area_aux)

        enco_0 = self._evaluator.create_constant_plain(const=0)
        encr_0 = self._encryptor.encrypt(plain=enco_0)

        self._scores.append(encr_0)

        self._status.append(encr_0)

        self._usr_count += 1

    def plain_encr_location_pair_contact_score(self, plain_location, encr_location):

        sq_diff_xs = self._evaluator.add_plain(ciph=encr_location[0],
                                               plain=self._evaluator.create_constant_plain(const=-plain_location[0]))
        sq_diff_xs = self._evaluator.multiply(ciph1=sq_diff_xs,
                                              ciph2=sq_diff_xs,
                                              relin_key=self._relin_key)

        sq_diff_ys = self._evaluator.add_plain(ciph=encr_location[1],
                                               plain=self._evaluator.create_constant_plain(const=-plain_location[1]))
        sq_diff_ys = self._evaluator.multiply(ciph1=sq_diff_ys,
                                              ciph2=sq_diff_ys,
                                              relin_key=self._relin_key)

        sq_dist = self._evaluator.add(ciph1=sq_diff_xs,
                                      ciph2=sq_diff_ys)

        const = -1 / (self._L_max ** 2)
        const = self._evaluator.create_constant_plain(const=const)

        fraction = self._evaluator.multiply_plain(ciph=sq_dist,
                                                  plain=const)

        base = self._evaluator.add_plain(ciph=fraction,
                                         plain=self._evaluator.create_constant_plain(const=1))

        for i in range(11):
            base = self._evaluator.multiply(ciph1=base,
                                            ciph2=base,
                                            relin_key=self._relin_key)

        return base

    def rcv_data_from_mo(self, loc_list, area_list, sts_list):
        for i in range(len(loc_list)):
            adj_indices = self.det_adj_area_ranges(area_tuple=area_list[i])

            for j in adj_indices[0]:
                for k in adj_indices[1]:
                    curr_bucket = self._area_array[area_list[i][0] + j][area_list[i][1] + k]
                    for user_index in curr_bucket:
                        self._scores[user_index] += sts_list[i] * self.location_pair_contact_score(
                            location1=loc_list[i],
                            location2=self._curr_locations[user_index]
                        )
        for i in range(len(loc_list)):
            adj_indices = self.det_adj_area_ranges(area_tuple=area_list[i])

            for j in adj_indices[0]:
                for k in adj_indices[1]:
                    curr_bucket = self._area_array[area_list[i][0] + j][area_list[area_list[i][1] + k]]
                    for user_index in curr_bucket:
                        dist_score = self.plain_encr_location_pair_contact_score(
                            plain_location=self._curr_locations[user_index],
                            encr_location=loc_list[i])

                        add_val = self._evaluator.multiply(ciph1=sts_list[i],
                                                           ciph2=dist_score,
                                                           relin_key=self._relin_key)

                        self._scores[user_index] = self._evaluator.add(ciph1=self._scores[user_index],
                                                                       ciph2=add_val)

    def send_data_to_mo(self, other_mo):
        loc_list = []

        for user in self._users:
            enco_x = self._evaluator.create_constant_plain(const=user.x)
            encr_x = self._encryptor.encrypt(plain=enco_x)

            enco_y = self._evaluator.create_constant_plain(const=user.y)
            encr_y = self._encryptor.encrypt(plain=enco_y)

            loc_list.append((encr_x, encr_y))

        other_mo.rcv_data_from_mo(loc_list=loc_list,
                                  area_list=self._curr_areas_by_user,
                                  sts_list=self._status
                                  )

    def inside_scoring(self):
        for i in range(len(self._users)):
            area = self._curr_areas_by_user[i]
            adj_indices = self.det_adj_area_ranges(area_tuple=area)

            for j in adj_indices[0]:
                for k in adj_indices[1]:
                    curr_bucket = self._area_array[area[0] + j][area[1] + k]

                    for user_index in curr_bucket:
                        if not user_index == i:
                            contact_score = self.location_pair_contact_score(
                                location1=self._curr_locations[i],
                                location2=self._curr_locations[user_index])
                            self._scores[i] = self._evaluator.multiply_plain(ciph=self._status[user_index],
                                                                             plain=contact_score)


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
                new_mo.register_other_mo(prev_mo)
            self._mos.append(new_mo)
            self._mo_count += 1

        for i in range(len(self._current_locations)):
            self.add_user(location=self._current_locations[i],
                          uid=i
                          )
            self._usr_count += 1

    def add_user(self, location, uid):
        new_user = ProtocolUser(init_x=location[0],
                                init_y=location[1],
                                mo=self._mos[uid % self._mo_count],
                                uid=uid,
                                ga=self._ga
                                )

        self._users.append(new_user)

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
        self._current_locations = next(self._movements_iterable)
        self._curr_time += 1

        for i in range(len(self._users)):
            self._users[i].move_to(new_x=self._current_locations[i][0],
                                   new_y=self._current_locations[i][1],
                                   update_time=self._curr_time
                                   )

        for i in range(self._mo_count):
            self._mos[i].tick()

        if self._curr_time % 1440 == 0:
            self._ga.daily()


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
        for i in range(len(self._current_locations)):
            tup = self._current_locations[i]
            aux1 = int(np.rint(tup[0]))
            aux2 = int(np.rint(tup[1]))

            self._current_locations[i] = (aux1, aux2)

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
                new_mo.register_other_mo(prev_mo)
            self._mos.append(new_mo)
            self._mo_count += 1

        for i in range(len(self._current_locations)):
            self.add_user(location=self._current_locations[i],
                          uid=i
                          )
            self._usr_count += 1

    def add_user(self, location, uid):
        new_user = EncryptedUser(init_x=location[0],
                                 init_y=location[1],
                                 mo=self._mos[uid % self._mo_count],
                                 uid=uid,
                                 ga=self._ga
                                 )

        self._users.append(new_user)

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
        self._current_locations = next(self._movements_iterable)
        for i in range(len(self._current_locations)):
            tup = self._current_locations[i]
            aux1 = int(np.rint(tup[0]))
            aux2 = int(np.rint(tup[1]))

            self._current_locations[i] = (aux1, aux2)
        self._curr_time += 1

        for i in range(len(self._users)):
            self._users[i].move_to(new_x=self._current_locations[i][0],
                                   new_y=self._current_locations[i][1],
                                   update_time=self._curr_time
                                   )

        for i in range(self._mo_count):
            self._mos[i].tick()

        if self._curr_time % 1440 == 0:
            self._ga.daily()


# class IterTest(unittest.TestCase):
#
#     def test_functionality(self):
#         for _ in range(10):
#             dummy_gm = gauss_markov(nr_nodes=10,
#                                     dimensions=(3652, 3652),
#                                     velocity_mean=7.,
#                                     alpha=.5,
#                                     variance=7.)
#
#             dual = DualIter(init_iter=dummy_gm)
#
#             for _ in range(10):
#                 fst = next(dual)
#                 snd = next(dual)
#
#                 for i in range(len(fst)):
#                     self.assertEqual(fst[i][0], snd[i][0], "Dual iterator no work")
#                     self.assertEqual(fst[i][1], snd[i][1], "Dual iterator no work")


# class EncryptionLibraryTest(unittest.TestCase):
#     def test_encodedecode(self):
#         poly_deg = 2
#         scaling_fact = 1 << 30
#         big_mod = 1 << 1200
#
#         ciph_mod = 1 << 300
#         params = CKKSParameters(poly_degree=poly_deg,
#                                 ciph_modulus=ciph_mod,
#                                 big_modulus=big_mod,
#                                 scaling_factor=scaling_fact)
#
#         encoder = CKKSEncoder(params=params)
#
#         for val in range(3653):
#             e1 = encoder.encode(values=[val],
#                                 scaling_factor=scaling_fact)
#             e2 = encoder.encode(values=[val + 0j],
#                                 scaling_factor=scaling_fact)
#             e3 = encoder.encode(values=[complex(val, 0)],
#                                 scaling_factor=scaling_fact)
#
#             self.assertEqual([val], encoder.decode(plain=e1), "encoder no work")
#             self.assertEqual([val], encoder.decode(plain=e2), "encoder no work")
#             self.assertEqual([val], encoder.decode(plain=e3), "encoder no work")
#
#     def test_encodeencryptdecryptdecode(self):
#         poly_deg = 2
#         scaling_fact = 1 << 30
#         big_mod = 1 << 1200
#
#         ciph_mod = 1 << 300
#         params = CKKSParameters(poly_degree=poly_deg,
#                                 ciph_modulus=ciph_mod,
#                                 big_modulus=big_mod,
#                                 scaling_factor=scaling_fact)
#
#         keygen = CKKSKeyGenerator(params=params)
#         pk = keygen.public_key
#         sk = keygen.secret_key
#
#         encoder = CKKSEncoder(params=params)
#
#         encryptor = CKKSEncryptor(params=params,
#                                   public_key=pk,
#                                   secret_key=sk)
#
#         decryptor = CKKSDecryptor(params=params,
#                                   secret_key=sk)
#
#         for val in range(3653):
#             e1 = encoder.encode(values=[val],
#                                 scaling_factor=scaling_fact)
#             e2 = encoder.encode(values=[val + 0j],
#                                 scaling_factor=scaling_fact)
#             e3 = encoder.encode(values=[complex(val, 0)],
#                                 scaling_factor=scaling_fact)
#
#             ee1 = encryptor.encrypt(plain=e1)
#             ee2 = encryptor.encrypt(plain=e2)
#             ee3 = encryptor.encrypt(plain=e3)
#
#             eed1 = decryptor.decrypt(ciphertext=ee1)
#             eed2 = decryptor.decrypt(ciphertext=ee2)
#             eed3 = decryptor.decrypt(ciphertext=ee3)
#
#             eedd1 = encoder.decode(plain=eed1)
#             eedd2 = encoder.decode(plain=eed2)
#             eedd3 = encoder.decode(plain=eed3)
#
#             self.assertLessEqual(val - eedd1[0].real, 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
#             self.assertLessEqual(abs(eedd1[0].imag), 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
#             self.assertLessEqual(val - eedd2[0].real, 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
#             self.assertLessEqual(abs(eedd2[0].imag), 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
#             self.assertLessEqual(val - eedd3[0].real, 3e-9, "encode-encrypt-decrypt-decode pipeline no work")
#             self.assertLessEqual(abs(eedd3[0].imag), 3e-9, "encode-encrypt-decrypt-decode pipeline no work")


class EncryptedUserTest(unittest.TestCase):
    def test_encodergetters(self):
        dummy_ga = EncryptionGovAgent(risk_threshold=5,
                                      degree=2,
                                      cipher_modulus=1 << 300,
                                      big_modulus=1 << 1200,
                                      scaling_factor=1 << 30)

        dummy_ga.new_encryption_suite()

        dummy_mo = EncryptionMO(ga=dummy_ga,
                                mo_id=0,
                                area_side_x=50,
                                area_side_y=50,
                                max_x=3652,
                                max_y=3652)

        test_user = EncryptedUser(init_x=12,
                                  init_y=12,
                                  mo=dummy_mo,
                                  uid=0,
                                  ga=dummy_ga)

        test_user._encoder = 5
        self.assertEqual(test_user.get_encoder(), 5, "encoder getter not proper")
        self.assertEqual(test_user.encoder, 5, "encoder property not proper")

        test_user._encoder = dummy_ga.encoder
        johnathan = dummy_ga.encoder
        self.assertEqual(test_user.get_encoder(), johnathan, "encoder getter not proper")
        self.assertEqual(test_user.encoder, johnathan, "encoder property not proper")

        test_user._evaluator = 5
        self.assertEqual(test_user.get_evaluator(), 5, "evaluator getter not proper")
        self.assertEqual(test_user.evaluator, 5, "evaluator property not proper")

        test_user._evaluator = dummy_ga.evaluator
        johnathan = dummy_ga.evaluator
        self.assertEqual(test_user.get_evaluator(), johnathan, "evaluator getter not proper")
        self.assertEqual(test_user.evaluator, johnathan, "evaluator property not proper")

        test_user._encryptor = 5
        self.assertEqual(test_user.get_encryptor(), 5, "encryptor getter not proper")
        self.assertEqual(test_user.encryptor, 5, "encryptor property not proper")

        test_user._encryptor = dummy_ga.encryptor
        johnathan = dummy_ga.encryptor
        self.assertEqual(test_user.get_encryptor(), johnathan, "encryptor getter not proper")
        self.assertEqual(test_user.encryptor, johnathan, "encryptor property not proper")

        test_user._scaling_factor = 5
        self.assertEqual(test_user.get_scaling_factor(), 5, "scaling_factor getter not proper")
        self.assertEqual(test_user.scaling_factor, 5, "scaling_factor property not proper")

        test_user._scaling_factor = dummy_ga.scaling_factor
        johnathan = dummy_ga.scaling_factor
        self.assertEqual(test_user.get_scaling_factor(), johnathan, "scaling_factor getter not proper")
        self.assertEqual(test_user.scaling_factor, johnathan, "scaling_factor property not proper")

        test_user._relin_key = 5
        self.assertEqual(test_user.get_relin_key(), 5, "relin_key getter not proper")
        self.assertEqual(test_user.relin_key, 5, "relin_key property not proper")

        test_user._relin_key = dummy_ga.relin_key
        johnathan = dummy_ga.relin_key
        self.assertEqual(test_user.get_relin_key(), johnathan, "relin_key getter not proper")
        self.assertEqual(test_user.relin_key, johnathan, "relin_key property not proper")

        test_user._public_key = 5
        self.assertEqual(test_user.get_public_key(), 5, "public_key getter not proper")
        self.assertEqual(test_user.public_key, 5, "public_key property not proper")

        test_user._public_key = dummy_ga.public_key
        johnathan = dummy_ga.public_key
        self.assertEqual(test_user.get_public_key(), johnathan, "public_key getter not proper")
        self.assertEqual(test_user.public_key, johnathan, "public_key property not proper")

    def test_scorefrommo(self):
        dummy_ga = EncryptionGovAgent(risk_threshold=5,
                                      degree=2,
                                      cipher_modulus=1 << 300,
                                      big_modulus=1 << 1200,
                                      scaling_factor=1 << 30)

        dummy_ga.new_encryption_suite()

        dummy_mo = EncryptionMO(ga=dummy_ga,
                                mo_id=0,
                                area_side_x=50,
                                area_side_y=50,
                                max_x=3652,
                                max_y=3652)

        test_user = EncryptedUser(init_x=12,
                                  init_y=12,
                                  mo=dummy_mo,
                                  uid=0,
                                  ga=dummy_ga)

        test_user.score_from_mo(score=dummy_mo._scores[0])
        decr = dummy_ga._decryptor.decrypt(ciphertext=test_user.encr_score)
        deco = test_user._encoder.decode(plain=decr)
        self.assertLessEqual(abs(deco[0].real), 3e-9, "score from mo no work")

        test_user.score_from_mo(69)
        self.assertEqual(test_user._encr_score, 69, "score from mo no work")

    def test_decrscorefromga(self):
        dummy_ga = EncryptionGovAgent(risk_threshold=5,
                                      degree=2,
                                      cipher_modulus=1 << 300,
                                      big_modulus=1 << 1200,
                                      scaling_factor=1 << 30)

        dummy_ga.new_encryption_suite()

        dummy_mo = EncryptionMO(ga=dummy_ga,
                                mo_id=0,
                                area_side_x=50,
                                area_side_y=50,
                                max_x=3652,
                                max_y=3652)

        test_user = EncryptedUser(init_x=12,
                                  init_y=12,
                                  mo=dummy_mo,
                                  uid=0,
                                  ga=dummy_ga)

        plane = 1234.1234
        encoplane = dummy_ga._encoder.encode(values=[complex(plane, 0)],
                                             scaling_factor=dummy_ga._scaling_factor)
        encrplane = dummy_ga._encryptor.encrypt(plain=encoplane)
        test_user._encr_score = encrplane
        test_user.decr_score_from_ga()
        self.assertLessEqual(abs(plane - test_user._score), 3e-9, "decr score from ga no work")

    def test_setnewfhesuite(self):
        dummy_ga = EncryptionGovAgent(risk_threshold=5,
                                      degree=2,
                                      cipher_modulus=1 << 300,
                                      big_modulus=1 << 1200,
                                      scaling_factor=1 << 30)

        dummy_ga.new_encryption_suite()

        dummy_mo = EncryptionMO(ga=dummy_ga,
                                mo_id=0,
                                area_side_x=50,
                                area_side_y=50,
                                max_x=3652,
                                max_y=3652)

        test_user = EncryptedUser(init_x=12,
                                  init_y=12,
                                  mo=dummy_mo,
                                  uid=0,
                                  ga=dummy_ga)

        self.assertEqual(dummy_ga.encryptor, test_user.encryptor, "encryptor setter no work")
        self.assertEqual(dummy_ga.encoder, test_user.encoder, "encoder setter no work")
        self.assertEqual(dummy_ga.scaling_factor, test_user.scaling_factor, "scaling_factor setter no work")
        self.assertEqual(dummy_ga.relin_key, test_user.relin_key, "relin_key setter no work")
        self.assertEqual(dummy_ga.public_key, test_user.public_key, "public_key setter no work")
        self.assertEqual(dummy_ga.evaluator, test_user.evaluator, "evaluator setter no work")

        test_const = 1976
        test_ptext = test_user._evaluator.create_constant_plain(const=test_const)
        test_ctext = test_user._encryptor.encrypt(plain=test_ptext)
        dummy_ga.new_encryption_suite()
        test_decr = dummy_ga._decryptor.decrypt(ciphertext=test_ctext)
        test_deco = dummy_ga._encoder.decode(plain=test_decr)

        self.assertLessEqual(abs(test_deco[0] - test_const), 3e-9,
                             "new encryption suite no work OR distribute fhe suite no work")


class EncryptionMOTest(unittest.TestCase):
    def test_suitegetters(self):
        dummy_ga = EncryptionGovAgent(risk_threshold=5,
                                      degree=2,
                                      cipher_modulus=1 << 300,
                                      big_modulus=1 << 1200,
                                      scaling_factor=1 << 30)

        test_mo = EncryptionMO(ga=dummy_ga,
                               mo_id=0,
                               area_side_x=50,
                               area_side_y=50,
                               max_x=3652,
                               max_y=3652)

        self.assertEqual(test_mo.get_encryptor(), None, "encryptor getter no work")
        self.assertEqual(test_mo.encryptor, None, "encryptor property no work")

        self.assertEqual(test_mo.get_encoder(), None, "encoder getter no work")
        self.assertEqual(test_mo.encoder, None, "encoder property no work")

        self.assertEqual(test_mo.get_scaling_factor(), None, "scaling factor getter no work")
        self.assertEqual(test_mo.scaling_factor, None, "scaling factor property no work")

        self.assertEqual(test_mo.get_relin_key(), None, "relin key getter no work")
        self.assertEqual(test_mo.relin_key, None, "relin key property no work")

        self.assertEqual(test_mo.get_public_key(), None, "public key getter no work")
        self.assertEqual(test_mo.public_key, None, "public key property no work")

        self.assertEqual(test_mo.get_evaluator(), None, "evaluator getter no work")
        self.assertEqual(test_mo.evaluator, None, "evaluator property no work")



unittest.main()
