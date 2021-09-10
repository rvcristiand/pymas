"""Assorted class utilities and tools"""


class AttrDisplay:
    def __repr__(self):
        """
        Get representation object.

        Returns
        -------
        str
            Object representation.
        """ 
        return "{}".format({key: value for key, value in self.__dict__.items() if not key.startswith('_') and value is not None})


if __name__ == "__main__":
    pass
