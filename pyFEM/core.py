# import numpy as np

from pyFEM.primitives import *
from pyFEM.classtools import Collection


class Materials(Collection):
    def __init__(self, parent):
        Collection.__init__(self)
        self.parent = parent

    def add(self, label, modulus):
        Collection.add(self, Material(label, modulus))


class Sections(Collection):
    def __init__(self, parent):
        Collection.__init__(self)
        self.parent = parent

    def add(self, label, material, area):
        Collection.add(self, Section(label, self.parent.materials[material], area))


class Nodes(Collection):
    def __init__(self, parent):
        Collection.__init__(self)
        self.parent = parent

    def add(self, label, x, y, z):
        Collection.add(self, Node(label, x, y, z))


class Trusses(Collection):
    def __init__(self, parent):
        Collection.__init__(self)
        self.parent = parent

    def add(self, node_i, node_j, section):
        node_i = self.parent.nodes[node_i]
        node_j = self.parent.nodes[node_j]
        section = self.parent.sections[section]

        Collection.add(self, Truss(self.parent, len(self), node_i, node_j, section))


class Supports(Collection):
    def __init__(self, parent):
        Collection.__init__(self)
        self.parent = parent

    def add(self, node, restrains):
        node = self.parent.nodes[node]

        Collection.add(self, Support(node, restrains))


class LoadPatterns(Collection):
    def __init__(self, parent):
        Collection.__init__(self)
        self.parent = parent

    def add(self, label):
        Collection.add(self, LoadPatter(self.parent, label))


class Structure:
    number_degrees_freedom = 3  # number degrees freedom per node

    def __init__(self):
        self.materials = Materials(self)
        self.sections = Sections(self)

        self.nodes = Nodes(self)
        self.trusses = Trusses(self)
        self.supports = Supports(self)

        self.load_patterns = LoadPatterns(self)

    def set_degrees_freedom(self):
        for i, node in enumerate(self.nodes):
            node.set_degrees_freedom(np.arange(self.number_degrees_freedom * i,
                                               self.number_degrees_freedom * (i + 1)))

    def get_k(self):
        k = np.zeros(2 * (self.number_degrees_freedom * len(self.nodes),))

        for truss in self.trusses:
            degrees_freedom = np.append(truss.node_i.degrees_freedom,
                                        truss.node_j.degrees_freedom)

            for i, row in enumerate(truss.get_k()):
                for j, item in enumerate(row):
                    k[degrees_freedom[i], degrees_freedom[j]] += item

        return k

    def solve(self):
        self.set_degrees_freedom()

        k = self.get_k()

        for load_pattern in self.load_patterns:
            k_load_pattern = k
            f = load_pattern.get_f()

            for support in self.supports:
                degrees_freedom = support.node.degrees_freedom

                for i, item in enumerate(support.restrains):
                    if item:
                        f[degrees_freedom[i], 0] = 0
                        k_load_pattern[degrees_freedom[i]] = np.zeros(np.shape(k)[0])
                        k_load_pattern[:, degrees_freedom[i]] = np.zeros(np.shape(k)[0])
                        k_load_pattern[degrees_freedom[i], degrees_freedom[i]] = 1

            print(np.linalg.solve(k_load_pattern, f))

    def __repr__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    # structure
    structure = Structure()

    # add material
    structure.materials.add("material1", 2040e4)

    # add sections
    structure.sections.add("section1", "material1", 30e-4)
    structure.sections.add("section2", "material1", 40e-4)
    structure.sections.add("section3", "material1", 100e-4)
    structure.sections.add("section4", "material1", 150e-4)

    # add nodes
    structure.nodes.add('0', 0, 0, 0)
    structure.nodes.add('1', 8, 0, 0)
    structure.nodes.add('2', 4, 3, 0)
    structure.nodes.add('3', 4, 0, 0)

    # add trusses
    structure.trusses.add(0, 2, "section3")
    structure.trusses.add(0, 3, "section2")
    structure.trusses.add(2, 1, "section4")
    structure.trusses.add(3, 1, "section2")
    structure.trusses.add(3, 2, "section1")

    # add support
    structure.supports.add(0, np.array([True, True, True]))
    structure.supports.add(1, np.array([False, True, True]))
    structure.supports.add(2, np.array([False, False, True]))
    structure.supports.add(3, np.array([False, False, True]))

    # add load pattern
    structure.load_patterns.add("point loads")

    # add point loads
    structure.load_patterns["point loads"].add_point_load(np.array([0, -20, 0]), 3)
    structure.load_patterns["point loads"].add_point_load(5 * np.array([0.8, 0.6, 0]), 2)

    # solve the problem
    structure.solve()

    # print
    # print("list of sections"), print(structure.sections, end='\n\n')
    # print("list of materials"), print(structure.materials, end='\n\n')
    # print("list of nodes"), print(structure.nodes, end='\n\n')
    # print("list of trusses"), print(structure.trusses, end='\n\n')
    # print("list of supports"), print(structure.supports, end='\n\n')
    # print("list of load patterns"), print(structure.load_patterns, end='\n\n')
