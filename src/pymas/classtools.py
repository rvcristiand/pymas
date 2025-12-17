class AttrDisplay:
    """Provides a string representation for objects based on their attributes.

    This class offers a `__repr__` method that generates a dictionary-like string
    representation of an object's public attributes (those not starting with '_')
    and whose values are not None.
    """

    def __repr__(self):
        """Gets a string representation of the object's public attributes.

        The representation includes all attributes that do not start with '_'
        and have a non-None value, presented as a dictionary.

        Returns:
            str: A string representation of the object.
        """
        return "{}".format({
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and value is not None
        })
