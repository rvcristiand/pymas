"""Assorted class utilities and tools"""


class AttrDisplay:
    """
    Provides and inheritable display overload method that shows
    instances with their class names and a name=value pair for
    each attribute stored on the instance itself (but no attrs
    inherited from its classes). Can be mixed into any class,
    and will work on any instance.
    """

    def gather_attrs(self):
        attrs = []
        for key in sorted(self.__dict__):
            attrs.append('%s=%s' % (key, getattr(self, key)))
        return ', '.join(attrs)

    def __repr__(self):
        return '[%s: %s]' % (self.__class__.__name__, self.gather_attrs())


class Collection:
    def __init__(self):
        self.collection = []

    def add(self, obj):
        already_added = [obj == item for item in self.collection]
        if not any(already_added):
            # index = already_added.index(True)

            # obj.label = self.collection[index].label
            # self.collection[index] = obj
            # pass
            # else:
            self.collection.append(obj)

    def _labels(self):
        return [obj.label for obj in self.collection]

    def __getitem__(self, item):
        return self.collection[self._labels().index(item)]

    def __iter__(self):
        for x in self.collection:
            yield x

    # def __contains__(self, item):
    #     return item in self._labels()

    def __len__(self):
        return len(self.collection)

    def __repr__(self):
        return '\n'.join([obj.__repr__() for obj in self.collection])


if __name__ == "__main__":
    class TopTest(AttrDisplay):
        count = 0

        def __init__(self):
            self.attr1 = TopTest.count
            self.attr2 = TopTest.count + 1
            TopTest.count += 2


    class SubTest(TopTest):
        pass


    X, Y = TopTest(), SubTest()
    print(X)
    print(Y)
