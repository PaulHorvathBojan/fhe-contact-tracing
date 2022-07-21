class MobileOperator: # TODO ga communication code

    def __init__(self, ga, mo_id):
        self._GA = ga
        self._users = []
        self._id = mo_id

    def get_ga(self):
        return self._GA

    GA = property(fget=get_ga)

    def get_users(self):
        return self._users

    users = property(fget=get_users)

    def get_id(self):
        return self._id

    id = property(fget=get_id)

    def add_user(self, user):
        self._users.append(user)

    #def to_ga_comm # we dont know yet

    #def from_ga_comm