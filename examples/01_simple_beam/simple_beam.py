from pymas import Structure

# model simple beam

# model and analyze a simple concrete beam subjected to its self weight

# dimensions of the rectangular cross section
b = 0.5  # width, m
h = 1    # heigh, m

# length and stiffness modulus
L = 10                 # length, m
E = 4700*28**0.5*1000  # stiffness module, kN/m2

# cross-sectional area and self weight
A = b*h  # cross-sectional area, m2
w = 24*A   # self weight per length, kN/m

# create the model
model = Structure(type='beam')

# add materials
model.add_material('concrete 28 MPa', E)

# add sections
model.add_rectangular_section('0.5x1.0', base=b, height=h)

# add joints
model.add_joint('a', x=0)
model.add_joint('b', x=L)

# add frame
model.add_frame('beam', 'a', 'b', 'concrete 28 MPa', '0.5x1.0')

# add supports
model.add_support('a', r_uy=True)
model.add_support('b', r_uy=True)

# add load patterns
model.add_load_pattern('self weight')

# add distributed loads
model.add_distributed_load('self weight', 'beam', fy=-w)

# analyze the model
model.run_analysis()
model.export('simple_beam.json')

print(f'Θa: {model.displacements['self weight']['a'].rz:+.3e} rad')
print(f'Θb: {model.displacements['self weight']['b'].rz:+.3e} rad')
print(f'Ra: {model.reactions['self weight']['a'].fy:+.1f} kN')
print(f'Rb: {model.reactions['self weight']['b'].fy:+.1f} kN')
print(f'Mmax: {max(model.internal_forces['self weight']['beam'].mz):.1f} kN m')
print(f'νmax: {min(model.internal_displacements['self weight']['beam'].uy):.3e} m')
