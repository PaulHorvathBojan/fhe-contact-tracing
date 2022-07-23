from Entities.GovAgent import GovAgent
from Entities.MobileOperator import MobileOperator
from Entities.User import User


class SpaceTimeLord:

    def __init__(self, movements_iterable, mo_count):
        self._usr_count = 0
        self._mo_count = 0

        self._mos = []
        self._users = []
        self._ga = GovAgent()
        self._movements_iterable = movements_iterable
        self._current_locations = next(movements_iterable)
        self._curr_time = 0

        while self._mo_count <= mo_count:
            self._mos.append(MobileOperator(ga=self._ga,
                                            mo_id=self._mo_count
                                            )
                             )
            self._mo_count += 1

        for i in range(len(self._current_locations)):
            self.add_user(location=self._current_locations[i],
                          uid=i
                          )

    def get_current_locations(self):
        return self._current_locations

    current_locations = property(fget=get_current_locations)

    def tick(self):
        self._current_locations = next(self._movements_iterable)
        self._curr_time += 1

        for i in range(len(self._users)):
            self._users[i].move_to(new_x=self._current_locations[i][0],
                                   new_y=self._current_locations[i][1],
                                   update_time=self._curr_time
                                   )

        for i in range(self._mo_count):
            self._mos[i].tick() # TODO: Implement periodic signalling in MO class

        if self._curr_time % 1440 == 0:
            self._ga.daily()  # TODO: Implement daily signalling in GA class

    def add_user(self, location, uid):
        new_user = User(init_x=location[0],
                        init_y=location[1],
                        mo=self._mos[uid % self._mo_count],
                        uid=uid,
                        ga=self._ga
                        )

        self._users.append(new_user)
        self._mos[uid % self._mo_count].add_user(new_user)
