class MobileOperator:  # TODO ga communication code
    # Initialize MobileOperator as having given GovAgent, ID, and area sides
    # By default:
    #   - _usr_count tracks number of users (init 0)
    #   - _users keeps a reference list of users (init empty)
    #   - _curr_locations keeps a list of location tuples for each user
    #           respectively --- same order as _users (init empty)
    #   - _scores keeps a list of user risk scores respectively --- same order as _users (init empty)
    #   - _area_sides keep the sizes of the considered tesselation area
    def __init__(self, ga, mo_id, area_side_x, area_side_y):
        self._GA = ga
        self._id = mo_id
        self._area_side_x = area_side_x
        self._area_side_y = area_side_y
        self._L_max = max(self._area_side_y, self._area_side_x)

        self._usr_count = 0
        self._users = []
        self._curr_locations = []
        self._scores = []
        self._curr_areas_by_user = []
        self._area_array = []
        self._status = []

        self._other_mos = []

        ys = []
        for i in range(19000):
            ys.append(set())
        for i in range(23000):
            self._area_array.append(ys)

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

    # register_other_mo creates reference to another MobileOperators for joint contact tracing purposes
    def register_other_mo(self, new_mo):
        self._other_mos.append(new_mo)

    # search_user_db will act as an auxiliary function to perform more efficient searches
    # inside the list of users internally
    # The first tiny assertion is that the users are added to the internal db in order of IDs.
    # The second tiny assertion is that no one will ever try ever to search for things that are not there.
    def search_user_db(self, user):
        stop_bool = 0
        left = 0
        right = self._usr_count - 1

        while stop_bool == 0:
            mid = (left + right) // 2
            if self._users[mid].uID <= user.uID:
                right = mid
            else:
                left = mid
            if left == right:
                stop_bool = 1

        if self._users[mid] is user:
            return mid
        else:
            AssertionError("User does not exist")

    # add_user adds a new user to the _users list
    # It increments _usr_count and updates the _curr_locations, _curr_areas, _scores, and _status
    #   lists accordingly:
    #   - _curr_locations gets the user's location appended
    #   - _curr_areas gets the appropriate area appended
    #   - _scores gets 0 appended
    #   - _status gets 0 appended
    def add_user(self, user):
        self._users.append(user)
        self._curr_locations.append((user.x, user.y))
        area_aux = self.assign_area(loc_tuple=self._curr_locations[-1])
        self._area_array[area_aux[0]][area_aux[1]].add({self.usr_count})
        self._curr_areas_by_user.append(area_aux)
        self._scores.append(0)
        self._status.append(0)
        self._usr_count += 1

    # assign_area assigns to a location an area code for running contact tracing in neighbouring areas only
    def assign_area(self, loc_tuple):
        return loc_tuple[0] // self.area_side_x, loc_tuple[1] // self.area_side_y

    # location_pair_contacts generates contact approx score between 2 locations
    # The actual formula may or may not be prone to changes; right now the general vibe is not exactly best.
    def location_pair_contact_score(self, location1, location2):
        return (1 - ((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)
                / self._L_max ** 2) ** 1024

    #
    def inside_scoring(self):
        for user in self.users:


    # tick models the passage of time inside the MO, much like the homonymous function does in SpaceTimeLord
    # TODO
    def tick(self):
        for user in self.users:
            user.upd_to_mo()

    def upd_user_data(self, user, new_x, new_y):

# def to_ga_comm # we dont know yet

# def from_ga_comm
