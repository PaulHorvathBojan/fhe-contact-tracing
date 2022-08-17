import random

from scipy.stats import bernoulli


class User:
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
        self._status = 0
        self._score = 0
        self._nonce = 0

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

    def get_status(self):
        return self._status

    status = property(fget=get_status)

    # move_to updates the current location of the user and the time of the last update.
    # The final part produces a Bernoulli variable that models the chance to ask the MO for the score:
    #   - the chance to ask for score is 1/(the number of seconds in 2 days)
    def move_to(self, new_x, new_y, update_time):
        self._x = new_x
        self._y = new_y
        self._last_update = update_time

        ask_score = bernoulli.rvs(0.00034722222)  # bernoulli argument is 1/2880
        if ask_score == 1:
            self.score_receipt_proc()

    # infect infects a user by setting status to 1
    def infect(self):
        self._status = 1

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
        self._GA.rcv_score_req_from_user(self._nonce * self._score)

    # send_sts_to_ga models the update procedure of the user's infection status inside the GA class
    # It calls the appropriate method inside the GA class with the user's infection status as argument
    def send_sts_to_ga(self):
        self._GA.infection_sts_from_user(self._status)

    # rcv_score_from_ga models receipt of score from the affiliated GA
    # The received value is the score masked multiplicatively with self._nonce.
    # The nonce is removed and the score is updated.
    def rcv_score_from_ga(self, nonced_score):
        self._score = nonced_score / self._nonce
        if self._score >= self._threshold:
            self.test_me()

    # test_me models requesting an infection test to the GA
    # It calls the test_user method in the _GA.
    def test_me(self):
        self._GA.test_user(self)

    #
    def at_risk(self):
        self.test_me()

    # score_receipt_proc models the score receipt procedure
    # There are a number of methods in various classes that call each other once certain criteria are met.
    # User.score_receipt_proc ->
    # User.ping_mo_for_score ->
    # MobileOperator.rcv_ping_from_user ->
    # MobileOperator.to_user_comm ->
    # User.score_from_mo
    # Then, the GA-related side of comms is initiated:
    # call User.decr_score_from_ga ->
    # GA.rcv_score_req_from_user ->
    # User.rcv_score_from_GA
    def score_receipt_proc(self):
        self.ping_mo_for_score()
        self.decr_score_from_ga()
