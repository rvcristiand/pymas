from pymas import Structure

# model a simple bar

# model and analyze a simple steel bar subjected to an axial force

# cross-sectional area
A = 198e-6  # area, m2

# length and stiffness modulus
L = 4            # length, m
E = 200_000_000  # stiffness module, kN/m2

# create the model
model = Structure(type='bar')

# add materials
model.add_material('steel', modulus_elasticity=E)

# add sections
model.add_section('5/8"', area=A)

# add joints
model.add_joint('a', x=0)
model.add_joint('b', x=L)

# add frame
model.add_frame('bar', 'a', 'b', 'steel', '5/8"', axial=True)

# add supports
model.add_support('a', r_ux=True)

# add load patterns
model.add_load_pattern('point loads')

# add point loads
model.add_joint_point_load('point loads', 'b', fx=42)  # kN

# analyze the model
model.run_analysis()
model.export('simple_bar.json')

# bar's elongation
print(f"ux_a: {model.displacements['point loads']['b'].ux:.3e} m")
