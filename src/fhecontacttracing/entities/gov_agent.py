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

        self._scaling_factor = scaling_factor
        self._ckks_params = CKKSParameters(poly_degree=degree,
                                           ciph_modulus=cipher_modulus,
                                           big_modulus=big_modulus,
                                           scaling_factor=scaling_factor
                                           )

        self._key_generator = CKKSKeyGenerator(self._ckks_params)

        self._public_key = self._key_generator.public_key
        self._secret_key = self._key_generator.secret_key
        self._relin_key = self._key_generator.relin_key

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

        self._encryptor = CKKSEncryptor(self._ckks_params, self._public_key, self._secret_key)
        self._decryptor = CKKSDecryptor(self._ckks_params, self._secret_key)

        self.distribute_fhe_suite()

    def distribute_fhe_suite(self):
        for mo in self._MOs:
            mo.refresh_fhe_keys(new_encryptor=self._encryptor,
                                new_relin_key=self._relin_key,
                                new_public_key=self._public_key)

        for user in self._users:
            user.refresh_fhe_keys(new_encryptor=self._encryptor,
                                  new_relin_key=self._relin_key,
                                  new_public_key=self._public_key)

    def add_user(self, new_user):
        super(EncryptionGovAgent, self).add_user(new_user)

        new_user.first_time_fhe_setup(new_evaluator=self._evaluator,
                                      new_encryptor=self._encryptor,
                                      new_encoder=self._encoder,
                                      new_scaling_factor=self._scaling_factor,
                                      new_relin_key=self._relin_key,
                                      new_public_key=self._public_key)

    def add_mo(self, new_mo):
        super(EncryptionGovAgent, self).add_mo(new_mo)

        new_mo.first_time_fhe_setup(new_evaluator=self._evaluator,
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
                self._evaluator.lower_modulus(ciph=encrypted_sts,
                                              division_factor=self._scaling_factor ** 13)
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
