class User:

    def __init__(self, init_x, init_y, mo, uid, ga):
        self._x = init_x
        self._y = init_y
        self._MO = mo
        self._uID = uid
        self._GA = ga
        self._last_update = 0

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

    def move_to(self, new_x, new_y, update_time):
        self._x = new_x
        self._y = new_y
        self._last_update = update_time

