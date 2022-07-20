class SpaceTimeLord:
    def __init__(self, movements_iterable):
        self._usr_count = 0
        self._movements_iterable = movements_iterable
        self.tick()
        for i in range(len(self._current_locations)):
            self.add_user(self._current_locations[i])

    def tick(self):
        self._current_locations = next(self._movements_iterable)

    def get_current_locations(self):
        return self._current_locations

    current_locations = property(fget=get_current_locations)