from pymas import Structure

"""
Solution to problem 7.2 from 'Microcomputadores en Ingeniería Estructural'
"""
# create the model
model = Structure(type='space_truss')

# add material
model.add_material("2100 t/cm2", 2100e4)

# add sections
model.add_section("10 cm2", 10e-4)
model.add_section("20 cm2", 20e-4)
model.add_section("40 cm2", 40e-4)
model.add_section("50 cm2", 50e-4)

# add joints
model.add_joint('1', 2.25, 6, 4.8)
model.add_joint('2', 3.75, 6, 2.4)
model.add_joint('3', 5.25, 6, 4.8)
model.add_joint('4', 0.00, 0, 6.0)
model.add_joint('5', 3.75, 0, 0.0)
model.add_joint('6', 7.50, 0, 6.0)

# add frames
model.add_frame('1-2', '1', '2', "2100 t/cm2", '20 cm2', axial=True)
model.add_frame('1-3', '1', '3', "2100 t/cm2", '20 cm2', axial=True)
model.add_frame('1-4', '1', '4', "2100 t/cm2", '40 cm2', axial=True)
model.add_frame('1-6', '1', '6', "2100 t/cm2", '50 cm2', axial=True)
model.add_frame('2-3', '2', '3', "2100 t/cm2", '20 cm2', axial=True)
model.add_frame('2-4', '2', '4', "2100 t/cm2", '50 cm2', axial=True)
model.add_frame('2-5', '2', '5', "2100 t/cm2", '40 cm2', axial=True)
model.add_frame('3-5', '3', '5', "2100 t/cm2", '50 cm2', axial=True)
model.add_frame('3-6', '3', '6', "2100 t/cm2", '40 cm2', axial=True)
model.add_frame('4-5', '4', '5', "2100 t/cm2", '10 cm2', axial=True)
model.add_frame('4-6', '4', '6', "2100 t/cm2", '10 cm2', axial=True)
model.add_frame('5-6', '5', '6', "2100 t/cm2", '10 cm2', axial=True)

# add supports
model.add_support('4', True, True, True)
model.add_support('5', True, True, True)
model.add_support('6', True, True, True)

# add load pattern
model.add_load_pattern("point loads")

# add point loads
model.add_joint_point_load("point loads", '1', 10, 15, -12)
model.add_joint_point_load("point loads", '2',  5, -3, -10)
model.add_joint_point_load("point loads", '3', -4, -2,  -6)

# analyze the model
model.run_analysis()
model.export('space_truss.json')

for joint in ['1', '2', '3']:
    print(f'joint: {joint}')
    print(f'\tux: {model.displacements["point loads"][joint].ux:.3e} m')
    print(f'\tux: {model.displacements["point loads"][joint].uy:.3e} m')
    print(f'\tux: {model.displacements["point loads"][joint].uz:.3e} m')
