import numpy as np

from pymas import Structure

""""Solution to problem 7.6 from 'Microcomputadores en Ingeniería Estructural'"""
# structure
model = Structure('beam')

# add material
fc = 21.1  # MPa
E = 4700 * fc**0.5 * 1000  # kPa
v = 0.2
G = E / (2 * (1 + v))
model.add_material('material1', E, G)

# add sections
model.add_rectangular_section('section1', dy=0.8, dz=1)
# model.add_section('section2', 0.10, 1.2734e-3, 1.333e-3, 5.208e-4)

# add nodes
model.add_joint('1', 0)
model.add_joint('2', 1)
model.add_joint('3', 2)
model.add_joint('4', 3)
model.add_joint('5', 4)


# add frames
model.add_frame('1-2', '1', '2', 'material1', 'section1',
                bending_z=True)
model.add_frame('2-3', '2', '3', 'material1', 'section1',
                bending_z=True)
model.add_frame('3-4', '3', '4', 'material1', 'section1',
                bending_z=True)
model.add_frame('4-5', '4', '5', 'material1', 'section1',
                bending_z=True)

# add supports
model.add_support('1', r_uy=True)
model.add_support('2', r_uy=True)
model.add_support('3', r_uy=True)
model.add_support('4', r_uy=True)
model.add_support('5', r_uy=True)

# add load pattern
model.add_load_pattern("distributed loads")

# add distributed loads
model.add_distributed_load('distributed loads', '1-2', fy=-1)
model.add_distributed_load('distributed loads', '2-3', fy=-1)
model.add_distributed_load('distributed loads', '3-4', fy=-1)
model.add_distributed_load('distributed loads', '4-5', fy=-1)

# solve
model.run_analysis()

# export
model.export('continuous_beam.json')

mz = []
for load_pattern in model.internal_forces.values():
    for element in load_pattern.values():
        mz+=element.mz

# plot the bending moment diagram if matplotlib is available
try:
    import matplotlib.pyplot as plt
except ImportError:
    print(f'Mmax: {max(mz):.4f}, Mmin: {min(mz):.4f}')
else:
    plt.plot(list(range(len(mz))), -np.array(mz))
    plt.show()
