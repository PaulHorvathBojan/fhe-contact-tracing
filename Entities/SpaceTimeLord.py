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

        while self._mo_count <= mo_count:
            self._mos.append(MobileOperator(GA=self._ga)) # TODO MO IDs

        self.tick()
        for i in range(len(self._current_locations)):
            self.add_user(self._current_locations[i])   # TODO appropriate user constructor, uID in constructor

    def tick(self):
        self._current_locations = next(self._movements_iterable)

    def get_current_locations(self):
        return self._current_locations

    current_locations = property(fget=get_current_locations)

