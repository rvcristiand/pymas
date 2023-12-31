import makepath

from pymas import Structure

""""Solution to problem 7.6 from 'Microcomputadores en Ingeniería Estructural'"""
# structure
model= Structure()

# add material
model.add_material('material1', 220e4, 85e4)

# add sections
model.add_section('section1', 0.12, 1.944e-3, 9e-4, 1.6e-3)
model.add_section('section2', 0.10, 1.2734e-3, 1.333e-3, 5.208e-4)

# add nodes
model.add_joint('1', 0, 3, 3)
model.add_joint('2', 5, 3, 3)
model.add_joint('3', 0, 0, 3)
model.add_joint('4', 0, 3, 0)

# add frames
model.add_frame('1-2', '1', '2', 'material1', 'section1')
model.add_frame('4-1', '4', '1', 'material1', 'section2')
model.add_frame('3-1', '3', '1', 'material1', 'section1')

# add supports
model.add_support('2', *6 * (True,))
model.add_support('3', *6 * (True,))
model.add_support('4', *6 * (True,))

# add load pattern
model.add_load_pattern("distributed loads")

# add distributed loads
model.add_uniformly_distributed_load('distributed loads', '1-2', wy=-2.4)
model.add_uniformly_distributed_load('distributed loads', '4-1', wy=-3.5)

# solve
model.solve()

# export
model.export('space_frame.json')
