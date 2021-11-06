import makepath

import ezdxf
import numpy as np
import matplotlib.pyplot as plt

from pyFEM import Structure
from matplotlib.ticker import MultipleLocator


# variables
l = 12  # length spans, m
no_div = 100  # no elementos por luz
no_spans = 2  # no luces

box_section = {
    '1' : {
        'width': 5.5,
        'depth': 7.5,
        'L1': 0.75,
        't1': 0.2,
        't2': 0.8,
        't3': 0.5,
        'f1h': 0.2,
        'f1v': 0.25,
        'f2h': 1,
        'f2v': 0.25,
        'f3h': 0,
        'f3v': 0
    },
    '2' : {
        'width': 11.3,
        'depth': 1.7,
        'L1': 3,
        't1': 0.25,
        't2': 0.23,
        't3': 0.4,
        'f1h': 1,
        'f1v': 0.2,
        'f2h': 1,
        'f2v': 0.35,
        'f3h': 1,
        'f3v': 0.24,
        't0': 0.2,
        't4': 0.4
    }
}

model = Structure(uy=True, rz=True)

# add material
model.add_material('mat', E=4800 * 35000 ** 0.5)

# add section
box_sect = model.add_box_section('sect', **box_section['2'])

# add main joints
supports = ['A', 'B', 'C']
for i, support in enumerate(supports):
    model.add_joint(support, i * l)

# divide spans in littler elements
for span in range(no_spans):
    joints = [supports[span]]
    x0 = span * l
    
    # add auxiliar joints
    for i in range(1, no_div):
        xi = x0 + (i / no_div) * l
        joint = model.add_joint(f'{xi:.3f}', xi)
        joints.append(joint.name)

    joints.append(supports[span + 1])

    # add frames
    for i in range(no_div):
        model.add_frame(f'{span+1}.{i+1}', joints[i], joints[i + 1] , 'mat', 'sect')

# add supports
for support in supports:
    model.add_support(support, uy=True)

# add load patterns
loadPattern = model.add_load_pattern('distributed')

# load_pattern = distributed
for frame in model.frames.values():
    model.add_distributed_load(loadPattern.name, frame.name, fy=1000.6)

# solution
model.solve()

x = []
vy = []
mz = []

for i, end_action in enumerate(model.end_actions[loadPattern.name].values()):
    joint_j = model.frames[end_action.frame].joint_j
    x.append(model.joints[joint_j].x)
    vy.append(end_action.fy_j)
    mz.append(end_action.mz_j)

joint_k = model.frames[end_action.frame].joint_k
x.append(model.joints[joint_k].x)
vy.append(end_action.fy_k)
mz.append(end_action.mz_k)

fig, ax = plt.subplots()

# plt.figure(2)
ax.plot(x, mz)
ax.axhline(y=0, color='k', lw=0.75)# ax.spines['bottom'].set_position(('axes', 0))
ax.set(xlabel='x', ylabel='M', title='Bending moment')
ax.set_xlim(xmin=0, xmax=no_spans*l)
ax.xaxis.set_major_locator(MultipleLocator(no_spans * l / 10))
# ax.xaxis.set_minor_locator(MultipleLocator())
ax.invert_yaxis()
ax.yaxis.set_major_locator(MultipleLocator((max(mz) - min(mz)) / 10))
# ax.yaxis.set_minor_locator(MultipleLocator())
ax.grid(True, which='both', axis='both')
# ax.xaxis.grid(True, which='minor')
# seaborn.despine(ax=ax, offset=0)
fig.savefig('mz.png')
plt.close(fig)

fig, ax = plt.subplots()

# plt.figure(1)
ax.plot(x, vy)
ax.axhline(y=0, color='k', lw=0.75)# ax.spines['bottom'].set_position(('axes', 0))
ax.set(xlabel='x', ylabel='V', title='Shear')
ax.set_xlim(xmin=0, xmax=no_spans*l)
ax.xaxis.set_major_locator(MultipleLocator(no_spans * l / 10))
# ax.xaxis.set_minor_locator(MultipleLocator())
ax.yaxis.set_major_locator(MultipleLocator((max(vy) - min(vy)) / 10))
# ax.yaxis.set_minor_locator(MultipleLocator())
ax.grid(True, which='both', axis='both')
# ax.xaxis.grid(True, which='minor')
# seaborn.despine(ax=ax, offset=0)
fig.savefig('vy.png')
plt.close(fig)

x = []
uy = []

for displacement in model.displacements[loadPattern.name].values():
    joint = displacement.joint
    x.append(model.joints[joint].x)
    uy.append(displacement.uy)

x, uy = zip(*sorted(zip(x, uy)))
    
fig, ax = plt.subplots()

for xi, uyi in zip(x, uy):
    print(xi, uyi)
# plt.figure(3)
ax.plot(x, uy)
ax.axhline(y=0, color='k', lw=0.75)# ax.spines['bottom'].set_position(('axes', 0))
ax.set(xlabel='x', ylabel='u', title='Displacement')
ax.set_xlim(xmin=0, xmax=no_spans*l)
ax.xaxis.set_major_locator(MultipleLocator(no_spans * l / 10))
# ax.xaxis.set_minor_locator(MultipleLocator())
ax.yaxis.set_major_locator(MultipleLocator((max(uy) - min(uy)) / 10))
# ax.yaxis.set_minor_locator(MultipleLocator())
ax.grid(True, which='both', axis='both')
# ax.xaxis.grid(True, which='minor')
# seaborn.despine(ax=ax, offset=0)
fig.savefig('uy.png')
plt.close(fig)
