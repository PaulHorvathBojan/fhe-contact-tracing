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


class EncryptionMO(MobileOperator):  # TODO: -key switch on fhe suite change

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
