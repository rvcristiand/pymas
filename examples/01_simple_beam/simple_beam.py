from pymas import Structure

# model an analyze a simple 10 m concrete beam and 28 MPa under its self weight

# create a model to analyze the beam (uy and rz degrees of freedoms activates)
model = Structure(type='beam')

# add a concrete of 28 MPa
model.add_material('concrete 28 MPa', E=4700*28**0.5*1000)  # kN/m2

# add a rectangular section of 0.5x1.0 m
model.add_rectangular_section('0.5x1.0', dy=1.0, dz=0.5)  # m

# add joints
model.add_joint('a', x=0.00)
model.add_joint('b', x=10.0)  # m

# add supports
model.add_support('a', r_uy=True)
model.add_support('b', r_uy=True)

# add frame
model.add_frame('1', 'a', 'b', 'concrete 28 MPa', '0.5x1.0', bending_z=True)

# add load pattern
model.add_load_pattern('self weight')

# add self weight
model.add_distributed_load('self weight', '1', fy=-24 * (0.5 * 1.0))  # kN/m

# analyze the model
model.run_analysis()
model.export('simple_beam.json')

print(f'Θa: {model.displacements['self weight']['a'].rz:+.3e} rad')
print(f'Ra: {model.reactions['self weight']['a'].fy:+.1f} kN')
print(f'Mmax: {max(model.internal_forces['self weight']['1'].mz):.1f} kN m')
