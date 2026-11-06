import matplotlib.pyplot as plt

from pymas import Structure

# model and analyze a beam on three supports, with two 10 m spans and two
# 2.5 m overhangs, of 28 MPa concrete under its self weight

# create a model to analyze the beam (uy and rz degrees of freedoms activates)
model = Structure(type='beam')

# add a concrete of 28 MPa
model.add_material('concrete 28 MPa', E=4700*28**0.5*1000)  # kN/m2

# add a rectangular section of 0.5x1.0 m
model.add_rectangular_section('0.5x1.0', dy=1.0, dz=0.5)  # m

# add joints at the ends of the overhangs and at the supports
model.add_joint('A', x=0.00)
model.add_joint('B', x=2.50)
model.add_joint('C', x=12.5)
model.add_joint('D', x=22.5)
model.add_joint('E', x=25.0)  # m

# add supports at B, C and D; A and E are free
model.add_support('B', r_ux=True, r_uy=True)
model.add_support('C', r_uy=True)
model.add_support('D', r_uy=True)

# add frames: the overhangs and the spans
for frame, (joint_j, joint_k) in enumerate(('AB', 'BC', 'CD', 'DE'), 1):
    model.add_frame(str(frame), joint_j, joint_k, 'concrete 28 MPa',
                    '0.5x1.0', bending_z=True)

# add self weight
for frame in model.frames:
    model.add_distributed_load(frame, fy=-24 * (0.5 * 1.0))  # kN/m

# analyze the model
model.run_analysis()

# plot the dimensioned beam with its loads, reactions and support and frame
# names, the shear force and bending moment diagrams, with their zero
# crossings dimensioned, and the deformed shape, with the positions of its
# largest deflections dimensioned, one under the other (sharex lines up the
# beams; the figure is tall enough for the diagrams and their dimensions)
fig, (loads, shear, moment, deformed) = plt.subplots(4, 1, figsize=(8, 14),
                                                     sharex=True)
model.plot_loads(loads, dimensions=True, support_names=True, frame_names=True)
model.plot_diagram('fy', ax=shear, labels=True, dimensions=True)
model.plot_diagram('mz', ax=moment, labels=True, dimensions=True)
model.plot_deformed(ax=deformed, labels=True, dimensions=True)
fig.tight_layout()
fig.savefig('overhanging_beam.svg')
