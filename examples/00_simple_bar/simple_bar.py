"""
Model a simple bar under longitudinal tension.

Reference:
    Gabriel Valencia Clement - "Diseño básico de estructuras de acero de
    acuerdo con la NSR-10"
    Chapter 3 (Tension), Example 3.1, part b (p. 44).

Problem statement:
    Calculate the elongation of a 5/8" diameter round steel bar
    with a length of 4.0 m subjected to an axial tensile load of 42 kN.

Model units:
    Force:        kilonewtons (kN)
    Length:       meters (m)
    Stress:       kN/m²
"""

from pymas import Structure

# 1. Initialize the structure
# 'bar' configures a 1D structure with 1 active degree of freedom per joint: ux
model = Structure(type='bar')

# 2. Add material properties
# Steel elastic modulus: E = 200 GPa = 200,000,000 kN/m²
model.add_material('steel', E=200_000_000)

# 3. Add cross-section properties
# 5/8" bar: Cross-sectional area Ag = 198 mm² = 198e-6 m²
model.add_section('5/8"', A=198e-6)

# 4. Add joints (geometry defined along global x-axis)
# Total bar length L = 4.0 m
model.add_joint('a', x=0)
model.add_joint('b', x=4)

# 5. Connect joints with a frame member
# axial=True activates axial deformation for this frame
model.add_frame('bar', 'a', 'b', 'steel', '5/8"', axial=True)

# 6. Restrain boundary conditions
# Restrain longitudinal translation ux at joint 'a'
model.add_support('a', r_ux=True)

# 7. Define load pattern and apply point load
# Tensile force Pu = 42 kN applied at joint 'b' along +x
model.add_load_pattern('point loads')
model.add_joint_point_load('point loads', 'b', fx=42)

# 8. Run analysis and export model data
model.run_analysis()
model.export('simple_bar.json')

# 9. Query and display joint displacements
# Elongation corresponds to the longitudinal displacement ux at free joint 'b'
elongation_m = model.displacements['point loads']['b'].ux
elongation_mm = elongation_m * 1000

print(f"Bar elongation (ux_b): {elongation_mm:.2f} mm ({elongation_m:.3e} m)")
