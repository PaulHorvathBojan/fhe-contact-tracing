class GovAgent:

    def __init__(self):
        self._users = []
        self._MOs = []
        self._infect_list = []

    def add_user(self, new_user):
        self._users.append(new_user)

    def add_mo(self, new_mo):
        self._MOs.append(new_mo)

    def get_mos(self):
        return self._MOs

    MOs = property(fget=get_mos)

    def get_users(self):
        return self._users

    users = property(fget=get_users)

    def get_infected(self):
        return self._infect_list

    infected = property(fget=get_infected)



