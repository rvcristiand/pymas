import numpy as np

from primitives import Material

class Structure:
    def __init__(self):
        self.xyz = []
        self.ele = []
        self.material = []

    def add_node(self, point):
        self.xyz.append(point)

    def add_element(self, element):
        self.ele.append(element)


if __name__ == '__main__':
    point1 = np.array([0, 0, 0])
    point2 = np.array([120, 0, 0])
    point3 = np.array([270, 0, 0])
