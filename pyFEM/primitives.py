import numpy as np

from numpy import linalg

from scipy.spatial import distance
from scipy.spatial.transform import Rotation

from pyFEM.classtools import AttrDisplay, Collection


class Material(AttrDisplay):
    def __init__(self, label, modulus_elasticity, modulus_elasticity_shear):
        self.label = label
        self.E = modulus_elasticity
        self.G = modulus_elasticity_shear

    def __eq__(self, other):
        return self.label == other.label


class Section(AttrDisplay):
    def __init__(self, label, area, moment_inertia_y, moment_inertia_z, torsion_constant):
        self.label = label
        self.A = area
        self.Iy = moment_inertia_y
        self.Iz = moment_inertia_z
        self.J = torsion_constant

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
    number_dimensions = 3
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
        t0 = self.get_orientation().as_dcm()
        t = np.zeros(2 * (self.number_nodes * self.number_degrees_freedom_per_node,))

        for i in range(int(self.number_degrees_freedom_per_node * self.number_nodes / self.number_dimensions)):
            t[i * self.number_dimensions:(i + 1) * self.number_dimensions,
              i * self.number_dimensions:(i + 1) * self.number_dimensions] = t0

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

    def get_forces(self, load_pattern):
        displacements = np.append(self.node_i.displacements[load_pattern].displacement,
                                  self.node_j.displacements[load_pattern].displacement).reshape(-1, 1)
        return -np.dot(np.linalg.inv(self.get_matrix_transformation()), np.dot(self.get_global_stiff_matrix(),
                                                                               displacements))[0, 0]

    def __eq__(self, other):
        return self.label == other.label or (self.node_i == other.node_i and self.node_j == other.node_j)


class Frame(Truss):
    number_degrees_freedom_per_node = 6

    def get_local_stiff_matrix(self):
        e = self.section.material.E
        g = self.section.material.G

        a = self.section.A
        iy = self.section.Iy
        iz = self.section.Iz
        j = self.section.J

        length = self.get_length()

        el = e / length
        el2 = e / length ** 2
        el3 = e / length ** 3

        gl = g / length

        ael = a * el
        gjl = j * gl

        e_iy_l = iy * el
        e_iz_l = iz * el

        e_iy_l2 = 6 * iy * el2
        e_iz_l2 = 6 * iz * el2

        e_iy_l3 = 12 * iy * el3
        e_iz_l3 = 12 * iz * el3

        k = np.zeros(2 * (self.number_nodes * self.number_degrees_freedom_per_node, ))

        # AE / L
        k[0, 0] = k[6, 6] = ael
        k[0, 6] = k[6, 0] = -ael

        # GJ / L
        k[3, 3] = k[9, 9] = gjl
        k[3, 9] = k[9, 3] = -gjl

        # 12EI / L^3
        k[1, 1] = k[7, 7] = e_iz_l3
        k[1, 7] = k[7, 1] = -e_iz_l3

        k[2, 2] = k[8, 8] = e_iy_l3
        k[2, 8] = k[8, 2] = -e_iy_l3

        # 6EI / L^2
        k[1, 5] = k[5, 1] = k[1, 11] = k[11, 1] = e_iz_l2
        k[5, 7] = k[7, 5] = k[7, 11] = k[11, 7] = -e_iz_l2

        k[2, 4] = k[4, 2] = k[2, 10] = k[10, 2] = -e_iy_l2
        k[4, 8] = k[8, 4] = k[8, 10] = k[10, 8] = e_iy_l2

        # 4EI / L
        k[4, 4] = k[10, 10] = 4 * e_iy_l
        k[5, 5] = k[11, 11] = 4 * e_iz_l

        k[10, 4] = k[4, 10] = 2 * e_iy_l
        k[11, 5] = k[5, 11] = 2 * e_iz_l

        return k


class Support(AttrDisplay):
    def __init__(self, node, ux, uy, uz, rx, ry, rz):
        self.label = node.label
        self.node = node
        self.restrains = np.array([ux, uy, uz, rx, ry, rz])

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


class DistributedLoad(AttrDisplay):
    def __init__(self, frame, fx, fy, fz):
        self.label = frame.label
        self.frame = frame
        self.load = np.array([fx, fy, fz])

    def __eq__(self, other):
        return self.frame == other.frame


class Displacement(AttrDisplay):
    def __init__(self, load_pattern, ux, uy, uz, rx, ry, rz):
        self.label = load_pattern.label
        # self.load_pattern = load_pattern
        self.displacement = np.array([ux, uy, uz, rx, ry, rz])


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
        self.distributed_loads = DistributedLoads(self.parent)

    def get_f_fixed(self):
        f = np.zeros((self.parent.number_degrees_freedom_per_node * len(self.parent.nodes), 1))

        for distributed_load in self.distributed_loads:
            degrees_freedom = np.concatenate((distributed_load.frame.node_i.degrees_freedom,
                                              distributed_load.frame.node_j.degrees_freedom))
            length = distributed_load.frame.get_length()

            # fx = distributed_load.load[0]
            fy = distributed_load.load[1]
            fz = distributed_load.load[2]

            f_local = [0, -fy * length / 2, -fz * length / 2, 0, fz * length ** 2 / 12, -fy * length ** 2 / 12]
            f_local += [0, -fy * length / 2, -fz * length / 2, 0, -fz * length ** 2 / 12, fy * length ** 2 / 12]

            f_global = np.dot(distributed_load.frame.get_matrix_transformation(), f_local)

            for i, item in enumerate(f_global):
                f[degrees_freedom[i], 0] += item

        return f

    def get_f(self):
        f = np.zeros((self.parent.number_degrees_freedom_per_node * len(self.parent.nodes), 1))

        for point_load in self.point_loads:
            degrees_freedom = point_load.node.degrees_freedom

            for i, item in enumerate(point_load.load):
                f[degrees_freedom[i], 0] += item

        return f - self.get_f_fixed()

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


class DistributedLoads(Collection):
    def __init__(self, parent):
        Collection.__init__(self)
        self.parent = parent

    def add(self, frame, fx, fy, fz):
        distributed_load = DistributedLoad(self.parent.frames[frame], fx, fy, fz)
        Collection.add(self, distributed_load)

        return distributed_load


class Displacements(Collection):
    def __init__(self):
        Collection.__init__(self)

    def add(self, load_pattern, ux, uy, uz, rx, ry, rz):
        displacement = Displacement(load_pattern, ux, uy, uz, rx, ry, rz)
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
