import matplotlib.pyplot as plt

from pymas import Structure

# model and analyze a 10 m beam fixed at both ends, of 28 MPa concrete under
# its self weight

# create a model to analyze the beam (uy and rz degrees of freedoms activates)
model = Structure(type='beam')

# add a concrete of 28 MPa
model.add_material('concrete 28 MPa', E=4700*28**0.5*1000)  # kN/m2

# add a rectangular section of 0.5x1.0 m
model.add_rectangular_section('0.5x1.0', dy=1.0, dz=0.5)  # m

# add joints
model.add_joint('A', x=0.00)
model.add_joint('B', x=10.0)  # m

# add fixed supports at both ends
model.add_support('A', r_ux=True, r_uy=True, r_rz=True)
model.add_support('B', r_uy=True, r_rz=True)

# add frame
model.add_frame('1', 'A', 'B', 'concrete 28 MPa', '0.5x1.0', bending_z=True)

# add self weight
model.add_distributed_load('1', fy=-24 * (0.5 * 1.0))  # kN/m

# analyze the model
model.run_analysis()

# plot the dimensioned beam with its loads, reactions and support names, the
# shear force and bending moment diagrams, with their zero crossings
# dimensioned, and the deformed shape, one under the other (sharex lines up
# the beams)
fig, (loads, shear, moment, deformed) = plt.subplots(4, 1, figsize=(8, 11),
                                                     sharex=True)
model.plot_loads(ax=loads, dimensions=True, support_names=True)
model.plot_diagram('fy', ax=shear, labels=True)
model.plot_diagram('mz', ax=moment, labels=True, dimensions=True)
model.plot_deformed(ax=deformed, labels=True)
fig.tight_layout()
fig.savefig('fixed_beam.svg')
