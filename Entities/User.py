class User:

    def __init__(self, init_x, init_y, MO, uID):
        self._x = init_x
        self._y = init_y
        self._MO = MO
        self._uID = uID

    def get_x(self):
        return self._x

    def set_x(self, new_x):
        self._x = new_x

    x = property(fget=get_x,
                 fset=set_x)

    def get_y(self):
        return self._y

    def set_y(self, new_y):
        self._y = new_y

    y = property(fget=get_y,
                 fset=set_y)

    def get_MO(self):
        return self._MO

    MO = property(fget=get_MO)

    def get_uID(self):
        return self._uID

    uID = property(fget=get_uID)