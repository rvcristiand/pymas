import numpy as np

from scipy.spatial import distance


class Material:
    def __init__(self, modulus):
        self.E = modulus


class Section:
    def __init__(self, material, area):
        self.material = material
        self.A = area


class Truss:
    def __init__(self,  p1, p2, section):
        self.p1 = p1
        self.p2 = p2
        self.section = section

    def getK(self):
        k = self.getE() * self.getA() / self.getL()

        return k * np.array([[1, -1], [-1, 1]])

    def getE(self):
        return self.section.material.E

    def getA(self):
        return self.section.A

    def getL(self):
        return distance.euclidean(self.p1, self.p2)


if __name__ == "__main__":
    p1 = np.array([0, 0])
    p2 = np.array([0, 0])

    truss1 = Truss
    print("holi")

