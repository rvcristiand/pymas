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
        self.coordinate = np.array([x, y, z])

        self.degrees_freedom = None  # Cambiar !!!
        self.displacements = Displacements()  # Cambiar !!!

    def set_degrees_freedom(self, u):
        self.degrees_freedom = u

    def __eq__(self, other):
        return self.label == other.label or np.all(self.coordinate == other.coordinate)


class Truss(AttrDisplay):
    number_nodes = 2
    number_degrees_freedom_per_node = 3

    def __init__(self, label, node_i, node_j, section):
        self.label = label
        self.node_i = node_i
        self.node_j = node_j
        self.section = section

    def get_local_vector(self):
        vector = self.node_j.coordinate - self.node_i.coordinate

        return vector / linalg.norm(vector)

    def get_orientation(self):
        v_from = np.array([1, 0, 0])
        v_to = self.get_local_vector()

        if np.all(v_from == v_to):
            return Rotation.from_quat([0, 0, 0, 1])

        elif np.all(v_from == -v_to):
            return Rotation.from_quat([0, 0, 1, 0])

        else:
            w = np.cross(v_from, v_to)
            w = w / linalg.norm(w)
            theta = np.arccos(np.dot(v_from, v_to))

            return Rotation.from_quat([x * np.sin(theta/2) for x in w] + [np.cos(theta/2)])

    def get_matrix_transformation(self):
        _t = self.get_orientation().as_dcm()
        t = np.zeros(2 * (self.number_nodes * self.number_degrees_freedom_per_node,))

        for i in range(self.number_nodes):
            t[i*self.number_degrees_freedom_per_node:(i + 1) * self.number_degrees_freedom_per_node,
              i*self.number_degrees_freedom_per_node:(i + 1) * self.number_degrees_freedom_per_node] = _t

        return t

    def get_local_stiff_matrix(self):
        k = self.get_modulus() * self.get_area() / self.get_length()
        k = k * np.array([[1, 0, 0, -1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [-1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]
                          ])

        return k

    def get_global_stiff_matrix(self):
        k = self.get_local_stiff_matrix()
        t = self.get_matrix_transformation()

        return np.dot(np.dot(t, k), np.transpose(t))

    def get_modulus(self):
        return self.section.material.E

    def get_area(self):
        return self.section.A

    def get_length(self):
        return distance.euclidean(self.node_i.coordinate, self.node_j.coordinate)

    def __eq__(self, other):
        return self.label == other.label or (self.node_i == other.node_i and self.node_j == other.node_j)


class Support(AttrDisplay):
    def __init__(self, node, ux, uy, uz):
        self.label = node.label
        self.node = node
        self.restrains = np.array([ux, uy, uz])

        self.reactions = Reactions()

    def __eq__(self, other):
        return self.node == other.node


class PointLoad(AttrDisplay):
    def __init__(self, node, fx, fy, fz):
        self.label = node.label
        self.node = node
        self.load = np.array([fx, fy, fz])

    def __eq__(self, other):
        return self.node == other.node


class Displacement(AttrDisplay):
    def __init__(self, load_pattern, ux, uy, uz):
        self.label = load_pattern.label
        # self.load_pattern = load_pattern
        self.displacement = np.array([ux, uy, uz])


class Reaction(AttrDisplay):
    def __init__(self, load_pattern, reactions):
        self.label = load_pattern.label
        # self.load_pattern = load_pattern
        self.reaction = np.array(reactions)


class LoadPattern(AttrDisplay):
    def __init__(self, label, parent):
        self.label = label
        self.parent = parent
        self.point_loads = PointLoads(self.parent)

    def get_f(self):
        f = np.zeros((self.parent.number_degrees_freedom_per_node * len(self.parent.nodes), 1))

        for point_load in self.point_loads:
            degrees_freedom = point_load.node.degrees_freedom

            for i, item in enumerate(point_load.load):
                f[degrees_freedom[i], 0] += item

        return f

    def __eq__(self, other):
        return self.label == other.label


class PointLoads(Collection):
    def __init__(self, parent):
        Collection.__init__(self)
        self.parent = parent

    def add(self, node, fx, fy, fz):
        point_load = PointLoad(self.parent.nodes[node], fx, fy, fz)
        Collection.add(self, point_load)

        return point_load


class Displacements(Collection):
    def __init__(self):
        Collection.__init__(self)

    def add(self, load_pattern, ux, uy, uz):
        displacement = Displacement(load_pattern, ux, uy, uz)
        Collection.add(self, displacement)

        return displacement


class Reactions(Collection):
    def __init__(self):
        Collection.__init__(self)

    def add(self, load_pattern, reactions):
        reaction = Reaction(load_pattern, reactions)
        Collection.add(self, reaction)

        return reaction


if __name__ == "__main__":
    pass
