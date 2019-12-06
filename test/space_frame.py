from pyFEM.core import Structure

import numpy as np


""""Solution to problem 7.6 from 'Microcomputadores en Ingeniería Estructural'"""
# structure
structure = Structure()

# add material
structure.materials.add('material1', 220e4, 85e4)

# add sections
structure.sections.add('section1', 'material1', 0.12, 9e-4, 1.6e-3, 1.944e-3)
structure.sections.add('section2', 'material1', 0.10, 1.333e-3, 5.208e-4, 1.2734e-3)

# add nodes
structure.joints.add('1', 0, 3, 3)
structure.joints.add('2', 5, 3, 3)
structure.joints.add('3', 0, 0, 3)
structure.joints.add('4', 0, 3, 0)

# add frames
structure.frames.add('1-2', '1', '2', 'section1')
structure.frames.add('3-1', '3', '1', 'section1')
structure.frames.add('4-1', '4', '1', 'section2')

# add supports
structure.supports.add('2', *6 * (True,))
structure.supports.add('3', *6 * (True,))
structure.supports.add('4', *6 * (True,))

# add load pattern
structure.load_patterns.add("distributed loads")

# add distributed loads
structure.load_patterns["distributed loads"].distributed_loads.add('1-2', 0, -2.4, 0)
structure.load_patterns["distributed loads"].distributed_loads.add('4-1', 0, -3.5, 0)

# solve
structure.solve()

np.set_printoptions(precision=3, suppress=True)
# displacements
print("displacement", end='\n\n')
for node in structure.joints:
    print("node:", node.label)

    for displacement in node.displacements:
        print("--> {}".format(displacement.label))
        print("\t", displacement.displacement)

print()

# supports
print("support", end='\n\n')
for support in structure.supports:
    print("support: ", support.label)

    for reaction in support.reactions:
        print("--> {}".format(reaction.label))
        print("\t", reaction.reaction)