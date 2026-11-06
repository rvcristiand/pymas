import matplotlib.pyplot as plt

from pymas import Structure

# model and analyze the continuous beam of problem 4.1 of 'Estructuras de
# Concreto I' (Segura): two spans of 7 and 8 m and a 2 m overhang, of a
# 0.60x0.45 m section of 21.1 MPa concrete, under a uniform load of 25 kN/m;
# the diagrams of the book are drawn for the factored load, 1.7 times it

# create a model to analyze the beam (uy and rz degrees of freedom activated)
model = Structure(type='beam')

# add a concrete of 21.1 MPa
model.add_material('concrete 21.1 MPa', E=4700*21.1**0.5*1000)  # kN/m2

# add a rectangular section of 0.60x0.45 m (0.45 m high)
model.add_rectangular_section('0.60x0.45', dy=0.45, dz=0.60)  # m

# add joints at the supports and at the end of the overhang
model.add_joint('A', x=0.00)
model.add_joint('B', x=7.00)
model.add_joint('C', x=15.0)
model.add_joint('D', x=17.0)  # m

# add supports
model.add_support('A', r_ux=True, r_uy=True)
model.add_support('B', r_uy=True)
model.add_support('C', r_uy=True)

# add frames: the two spans and the overhang
for frame in ('AB', 'BC', 'CD'):
    model.add_frame(frame, frame[0], frame[1], 'concrete 21.1 MPa',
                    '0.60x0.45', bending_z=True)

# add the factored load
for frame in model.frames:
    model.add_distributed_load(frame, fy=-1.7 * 25)  # kN/m

# analyze the model
model.run_analysis()

# compare with the design shears and moments of the book (figure 4.10),
# at the supports (on their left, l, and right, r) and in the spans
forces = model.internal_forces
book = [
    ('Vu A', forces['AB'].fy[0], 108.7),
    ('φMn AB', max(forces['AB'].mz), 139.1),
    ('Vu B,l', -forces['AB'].fy[-1], 188.8),
    ('φMn B', -forces['BC'].mz[0], 280.2),
    ('Vu B,r', forces['BC'].fy[0], 194.4),
    ('φMn BC', max(forces['BC'].mz), 164.4),
    ('Vu C,l', -forces['BC'].fy[-1], 145.6),
    ('φMn C', -forces['CD'].mz[0], 85.0),
    ('Vu C,r', forces['CD'].fy[0], 85.0),
]
print(f'{"":8}{"pymas":>8}{"book":>8}')
for name, value, expected in book:
    print(f'{name:8}{value:8.1f}{expected:8.1f}')

# plot the dimensioned beam with its loads, reactions and support names, the
# shear force and bending moment diagrams, with their zero crossings
# dimensioned, and the deformed shape, with the positions of its largest
# deflections dimensioned, one under the other (sharex lines up the beams)
fig, (loads, shear, moment, deformed) = plt.subplots(4, 1, figsize=(10, 16),
                                                     sharex=True)
model.plot_loads(ax=loads, dimensions=True, support_names=True)
model.plot_diagram('fy', ax=shear, labels=True, dimensions=True)
model.plot_diagram('mz', ax=moment, labels=True, dimensions=True)
model.plot_deformed(ax=deformed, labels=True, dimensions=True)
fig.tight_layout()
fig.savefig('concrete_beam.svg')
