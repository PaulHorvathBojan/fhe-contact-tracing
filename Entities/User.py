class User:
    # Initialize user at a certain location with a certain MO, user ID and GA.
    # After initializing class fields to the parameter and default values respectively, call add_user in the
    #   user's MO.
    # By default:
    #    - the user's last update is at tick 0
    #    - the user is not infected
    #    - the user's infection time is 0
    def __init__(self, init_x, init_y, mo, uid, ga):
        self._x = init_x
        self._y = init_y
        self._MO = mo
        self._uID = uid
        self._GA = ga
        self._last_update = 0
        self._status = 0  # intuitively going for a Susceptible(0)-Infected(1)-Recovered(2) model
        self._infection_time = 0

        self._MO.add_user(self)

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

    def get_status(self):
        return self._status

    status = property(fget=get_status)

    # move_to updates the current location of the user and the time of the last update.
    # The latter part also triggers the user's passive healing from covid after 2 weeks from the initial infection.
    # (might scrap the healing part as it seems out of scope)
    def move_to(self, new_x, new_y, update_time):
        self._x = new_x
        self._y = new_y
        self._last_update = update_time
        self.recover_from_covid()
        self.lose_antibodies()

    # infect infects a user and sets the time of infection to the current user time
    def infect(self):
        if self.status == 0:
            self._status = 1
            self._infection_time = self.last_update

    # recover_from_covid simulates recovery from covid (after 10080 ticks, which is 2 weeks for minutely ticks)
    def recover_from_covid(self):
        if self.status == 1 and self.last_update - self._infection_time >= 20160:
            self._status = 2

    # lose_antibodies simulates loss of antibodies after 262800 ticks (6 months worth of minutes)
    def lose_antibodies(self):
        if self.status == 2 and self.last_update - self._infection_time >= 262800:
            self._status = 0

    # upd_to_mo updates the data that the MO has regarding the self
    # it essentially calls a function in the MO that performs the updating
    def upd_to_mo(self):
        self._MO.upd_user_data(user=self,
                               new_x=self.x,
                               new_y=self.y
                               )
        # --- essentially in a setting that puts users in a database, this would be needed to:
        #   - update user locations in the db
        #   - trigger any eventual events inside the MO class
    # TODO:
    # def score_from_mo(self, encr_score):
    #   self._encr_score = encr_score

    # def ping_mo_for_score(self):
    #   self._mo.rcv_ping(self) --- not 100% sure yet how this will work, throwing stuff around
