from Entities.GovAgent import GovAgent
from Entities.MobileOperator import MobileOperator
from Entities.User import User


class SpaceTimeLord:

    def __init__(self, movements_iterable, mo_count, risk_thr, area_sizes):
        self._movements_iterable = movements_iterable
        self._risk_threshold = risk_thr
        self._max_x = area_sizes[0]
        self._max_y = area_sizes[1]

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
                                    area_side_x=self._max_x,
                                    area_side_y=self._max_y
                                    )
            for prev_mo in self._mos:
                new_mo.register_other_mo(prev_mo)
            self._mos.append(new_mo)
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
            self._mos[i].tick()

        if self._curr_time % 1440 == 0:
            self._ga.daily()

    def add_user(self, location, uid):
        new_user = User(init_x=location[0],
                        init_y=location[1],
                        mo=self._mos[uid % self._mo_count],
                        uid=uid,
                        ga=self._ga
                        )

        self._users.append(new_user)
        self._mos[uid % self._mo_count].add_user(new_user)
