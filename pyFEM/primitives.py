import numpy as np

from numpy import linalg

from scipy.spatial import distance
from scipy.spatial.transform import Rotation

from pyFEM.classtools import AttrDisplay, Collection


class Material(AttrDisplay):
    def __init__(self, label, modulus):
        self.label = label
        self.E = modulus

    def __eq__(self, other):
        return self.label == other.label


class Section(AttrDisplay):
    def __init__(self, label, material, area):
        self.label = label
        self.material = material
        self.A = area

    def __eq__(self, other):
        return self.label == other.label


class Node(AttrDisplay):
    def __init__(self, label, x, y, z):
        self.label = label
        self.position = np.array([x, y, z])

        self.degrees_freedom = None
        self.displacements = Displacements()

    def set_degrees_freedom(self, u):
        self.degrees_freedom = u

    def __eq__(self, other):
        return np.all(self.position == other.position)


class Truss(AttrDisplay):
    def __init__(self, parent, label, node_i, node_j, section):
        self.parent = parent

        self.label = label
        self.node_i = node_i
        self.node_j = node_j
        self.section = section

        self.position = self.node_i.position
        self.orientation = self._quaternion_from_two_vectors(self.local_vector())

    def matrix_transformation(self):
        _t = self.orientation.as_dcm()
        t = np.zeros(2 * (2 * self.parent.number_degrees_freedom,))

        t[0:self.parent.number_degrees_freedom, 0:self.parent.number_degrees_freedom] = _t
        t[self.parent.number_degrees_freedom:, self.parent.number_degrees_freedom:] = _t

        return t

    def local_vector(self):
        return self.node_j.position - self.node_i.position

    @staticmethod
    def _quaternion_from_two_vectors(v_to):
        v_from = np.array([1, 0, 0])
        v_to = v_to / linalg.norm(v_to)

        if np.all(v_from == v_to):
            return Rotation.from_quat([0, 0, 0, 1])
        elif np.all(v_from == -v_to):
            return Rotation.from_quat([0, 0, 1, 0])
        else:
            w = np.cross(v_from, v_to)
            w = w / linalg.norm(w)
            theta = np.arccos(np.dot(v_from, v_to))

            return Rotation.from_quat([x * np.sin(theta/2)for x in w] + [np.cos(theta/2)])

    def get_k(self):
        k = self.get_modulus() * self.get_area() / self.get_length()
        k = k * np.array([[1, 0, 0, -1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [-1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]
                          ])

        t = self.matrix_transformation()

        return np.dot(np.dot(t, k), np.transpose(t))

    def get_modulus(self):
        return self.section.material.E

    def get_area(self):
        return self.section.A

    def get_length(self):
        return distance.euclidean(self.node_i.position, self.node_j.position)

    def __eq__(self, other):
        return self.node_i == other.node_i and self.node_j == other.node_j


class Support(AttrDisplay):
    def __init__(self, node, restrains):
        self.label = node.label
        self.node = node
        self.restrains = restrains

    def __eq__(self, other):
        return self.label == other.label


class PointLoad(AttrDisplay):
    def __init__(self, node, load):
        self.node = node
        self.load = load

    def __eq__(self, other):
        return False


class LoadPatter(AttrDisplay):
    def __init__(self, parent, label):
        self.parent = parent

        self.label = label
        self.point_loads = Collection()

    def add_point_load(self, node, load):
        node = self.parent.nodes[node]
        self.point_loads.add(PointLoad(node, load))

    def get_f(self):
        f = np.zeros((self.parent.number_degrees_freedom * len(self.parent.nodes), 1))

        for point_load in self.point_loads:
            degrees_freedom = point_load.node.degrees_freedom

            for i, item in enumerate(point_load.load):
                f[degrees_freedom[i], 0] += item

        return f


class Displacement(AttrDisplay):
    def __init__(self, load_pattern, displacement):
        self.load_patter = load_pattern
        self.displacement = displacement


class Displacements(Collection):
    def __init__(self):
        Collection.__init__(self)

    def add(self, load_pattern, displacement):
        Collection.add(self, Displacement(load_pattern, displacement))


if __name__ == "__main__":
    pass
