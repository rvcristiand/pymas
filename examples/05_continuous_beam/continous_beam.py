import numpy as np

import matplotlib.pyplot as plt
from pymas import Structure

""""Solution to problem 7.6 from 'Microcomputadores en Ingeniería Estructural'"""
# structure
model= Structure()

# add material
fc = 21.1  # MPa
E = 4700 * fc**0.5 * 1000  # kPa
v = 0.2
G = E / (2 * (1 + v))
model.add_material('material1', E, G)

# add sections
model.add_rectangular_section('section1', 1, 0.8)
# model.add_section('section2', 0.10, 1.2734e-3, 1.333e-3, 5.208e-4)

# add nodes
model.add_joint('1', 0)
model.add_joint('2', 1)
model.add_joint('3', 2)
model.add_joint('4', 3)
model.add_joint('5', 4)


# add frames
model.add_frame('1-2', '1', '2', 'material1', 'section1')
model.add_frame('2-3', '2', '3', 'material1', 'section1')
model.add_frame('3-4', '3', '4', 'material1', 'section1')
model.add_frame('4-5', '4', '5', 'material1', 'section1')

# add supports
model.add_support('1', r_ux=True, r_uy=True, r_uz=True, r_rx=True, r_ry=True)
model.add_support('2', r_ux=True, r_uy=True, r_uz=True, r_rx=True, r_ry=True)
model.add_support('3', r_ux=True, r_uy=True, r_uz=True, r_rx=True, r_ry=True)
model.add_support('4', r_ux=True, r_uy=True, r_uz=True, r_rx=True, r_ry=True)
model.add_support('5', r_ux=True, r_uy=True, r_uz=True, r_rx=True, r_ry=True)

# add load pattern
model.add_load_pattern("distributed loads")

# add distributed loads
model.add_uniformly_distributed_load('distributed loads', '1-2', wy=-1)
model.add_uniformly_distributed_load('distributed loads', '2-3', wy=-1)
model.add_uniformly_distributed_load('distributed loads', '3-4', wy=-1)
model.add_uniformly_distributed_load('distributed loads', '4-5', wy=-1)

# solve
model.solve()

# export
model.export('continuous_beam.json')

mz = []
for load_pattern in model.internal_forces.values():
    for element in load_pattern.values():
        mz+=element.mz

# print(mz)
plt.plot(list(range(len(mz))), -np.array(mz))

plt.show()


