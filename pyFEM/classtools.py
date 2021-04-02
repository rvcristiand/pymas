"""Assorted class utilities and tools"""


class UniqueInstances(type):
    """
    UniqueInstances metaclass

    Methods
    -------
    __new__
        asd
    __call__
        asd
    setattr
        asd
    """

    def __new__(mcs, name, bases, dct):
        """
        Create a class

        Parameters
        ----------
        name : str
            Class name.
        bases : tuple
            Parent classes.
        dct : dict
            Namespace's class.
        """
        if '__slots__' in dct:
            dct['instances_attrs'] = set()
            dct['__setattr__'] = UniqueInstances.setattr
            dct['__del__'] = UniqueInstances.delete

            return type.__new__(mcs, name, bases, dct)
        else:
            print("Warning: " +
                  "Classes created with the UniqueInstances metaclass must implement the " +
                  "'__slots__ ' variable. The class was not created.")

    def __call__(cls, *args, **kwargs):
        """
        Return an instances if it does not already exist otherwise return None

        Arguments
        ---------
        args : tuple
            asd
        kwargs : dict
            asd
        """
        # get __init__ class
        init = cls.__init__

        # get init's arguments and default values
        varnames = getattr(getattr(init, '__code__'), 'co_varnames')[len(args) + 1:]
        default = getattr(init, '__defaults__')

        # create list with args
        instance_attrs = list(args)

        # fill instance_attrs with kwargs or init's default values
        for i, key in enumerate(varnames):
            instance_attrs.append(kwargs.get(key, default[i]))

        # from list to tuple
        instance_attrs = tuple(instance_attrs)  # FIXME: i don't need necessary check all params

        # get obj's attrs and instances attrs class
        instances_attrs = getattr(cls, 'instances_attrs')

        # check obj's attrs don't be in instances attrs class
        if instance_attrs in instances_attrs:
            print("Warning: " +
                  "There is another instance of the class " +
                  "'{}' ".format(cls.__name__) +
                  "with the same attributes. The object was not created.")
        else:
            # add obj's attrs to instances attrs
            instances_attrs.add(instance_attrs)

            # create and instantiate the object
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)

            return obj

    def setattr(self, key, value):
        """
        Set attribute object if doesn't collide with attributes another object

        Parameters
        ----------
        key : string
            Key's attribute to modified.
        value : value
            Value to assign.
        """
        if hasattr(self, key):
            # get instances attrs and instance attrs
            instances_attrs = getattr(self.__class__, 'instances_attrs')
            instance_attrs = tuple(getattr(self, name) for name in self.__slots__)

            # get possible new instance attrs
            _instance_attrs = tuple((getattr(self, _key) if _key != key
                                     else value for _key in self.__slots__))

            # add new instance attrs if not in instances attrs
            if _instance_attrs in instances_attrs:
                print("Warning: " +
                      "There is another instance of the class " +
                      "'{}'".format(self.__class__.__name__) +
                      " with the same attributes. The object was not changed.")

                return None
            else:
                instances_attrs.remove(instance_attrs)
                instances_attrs.add(_instance_attrs)

        self.__class__.__dict__[key].__set__(self, value)
    
    def delete(self):
        getattr(self.__class__, 'instances_attrs').remove(tuple(getattr(self, name) for name in self.__slots__))


class AttrDisplay:
    __slots__ = ()

    def __repr__(self):
        """
        Get representation object

        Returns
        -------
        str
            Object representation.
        """
        return "{}({})".format(self.__class__.__name__,', '.join([repr(getattr(self, name)) for name in self.__slots__]))


if __name__ == "__main__":
    class Coordinate(AttrDisplay, metaclass=UniqueInstances):
        __slots__ = ('x', 'y', 'z')

        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z


    # view instances' attrs
    print(getattr(Coordinate, 'instances_attrs'))  # set()

    # instantiates first object
    coordinate1 = Coordinate(0, 0, 0)
    print(coordinate1)  # Coordinate(0, 0, 0)

    # try to add a attr to first object
    try:
        setattr(coordinate1, 'a', 1)
    except Exception as e:
        print(e.__class__.__name__, e)  # 'Coordinate' object has no attribute 'a'

    # view instances' attrs
    print(getattr(Coordinate, 'instances_attrs'))

    # try instantiates second object with attrs first object
    coordinate2 = eval(repr(coordinate1))  # Warning: There is another instance...
    print(coordinate2)  # None

    # view instances' attrs
    print(getattr(Coordinate, 'instances_attrs'))

    # del first object
    del coordinate1

    # view instances' attrs
    print(getattr(Coordinate, 'instances_attrs'))  # set()

    # create second object
    coordinate2 = Coordinate(0, 0, 0)
    coordinate3 = Coordinate(1, 0, 0)
    print(coordinate2)
    
    # view instances' attrs
    print(getattr(Coordinate, 'instances_attrs'))  # {(1, 0, 0), (0, 0, 0)}

    # change value attrs
    coordinate2.x = 1
    print(coordinate2)

    # view instances' attrs
    print(getattr(Coordinate, 'instances_attrs'))

    # create third object
    print(coordinate3)

    # view instances' attrs
    print(getattr(Coordinate, 'instances_attrs'))

    # change value attrs
    coordinate3.x = 0  # Warning: There is another instance...
    print(coordinate3)  # Coordinate(1, 0, 0)
