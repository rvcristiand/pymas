"""
Model the Eiffel Tower as a space truss under self weight and wind.

This is a simplified, illustrative model, not a reproduction of the real
structure. Each of the four legs is reduced to a single chord, and the
lattice is idealized as pin-jointed bars (axial force only):

    - Height: 300 m (the roof of the top platform; antennas are ignored).
    - Plan: square, 125 m wide at the base. The width follows an
      exponential profile fitted to the 1st floor (about 70 m at 57.6 m)
      and the 2nd floor (about 41 m at 115.7 m), with a minimum of 10 m.
    - Members: four chords (the legs), a horizontal ring and two plan
      diagonals at every level, and an X-brace on every face of every panel.
    - Supports: the four feet are pinned.
    - Material: puddled iron, E = 200 GPa, unit weight 77 kN/m³.
    - Areas: chosen per height band so that the iron weighs about as much as
      the 7,300 t of the real tower.

Load patterns:
    - 'self weight': the weight of each bar, half at each end joint.
    - 'wind': along +x, with a pressure that grows linearly from 1.5 kN/m² at
      the base to 3.0 kN/m² at the top, acting on 25% of the outline of the
      windward face (the lattice is mostly empty), shared by the four joints
      of each level.

Model units:
    Force:        kilonewtons (kN)
    Length:       meters (m)
    Stress:       kN/m²

If matplotlib is installed, the script also plots the axial forces of both
load patterns and the deformed shape under wind, and saves the figure as
eiffel_tower.png.

Expected results (the exported JSON, about 17 MB, is not kept in the
repository):
    - 60 joints and 252 frames; the iron weighs 69,225 kN (7,057 t).
    - Self weight: the top settles 2.3 cm and each leg carries 20,523 kN in
      compression at the base.
    - Wind (5,385 kN in total): the top sways 4.2 cm; the legs at the base
      carry 2,953 kN, in compression on the leeward side and in tension on
      the windward side.
"""

import numpy as np

from pymas import Structure

# 1. Geometry
# heights of the levels: the floors (57.6, 115.7 and 276.1 m) are levels
heights = np.concatenate((
    np.linspace(0, 57.6, 4),          # base to 1st floor
    np.linspace(57.6, 115.7, 4)[1:],  # 1st to 2nd floor
    np.linspace(115.7, 276.1, 8)[1:],  # 2nd to 3rd floor
    [300.0],                          # top
))


def width(z):
    """Return the width of the tower at height z (exponential profile)."""
    return max(125 * np.exp(-z / 101), 10)


# corners of a level, counterclockwise from (+x, +y)
corners = {'a': (1, 1), 'b': (-1, 1), 'c': (-1, -1), 'd': (1, -1)}


def joint(level, corner):
    """Return the name of a joint, e.g. '3a'."""
    return f'{level}{corner}'


# 2. Initialize the structure
# 'space_truss' has 3 active degrees of freedom per joint: ux, uy, uz
model = Structure(type='space_truss')

# 3. Add material and sections
model.add_material('iron', E=200e6)
unit_weight = 77  # kN/m³

# chord (leg) areas by height band, and areas of horizontals and diagonals
model.add_section('chord 0-58', A=0.90)
model.add_section('chord 58-116', A=0.56)
model.add_section('chord 116-276', A=0.28)
model.add_section('chord 276-300', A=0.10)
model.add_section('horizontal', A=0.040)
model.add_section('diagonal', A=0.030)


def chord_section(z):
    """Return the chord section of the panel that starts at height z."""
    if z < 57.6:
        return 'chord 0-58'
    if z < 115.7:
        return 'chord 58-116'
    if z < 276.1:
        return 'chord 116-276'
    return 'chord 276-300'


# 4. Add joints
for level, z in enumerate(heights):
    half = width(z) / 2
    for corner, (sx, sy) in corners.items():
        model.add_joint(joint(level, corner), x=sx * half, y=sy * half, z=z)

# 5. Add frames (axial only, as in a truss)
names = list(corners)  # ['a', 'b', 'c', 'd']

for level, z in enumerate(heights):
    up = level + 1

    # chords (legs) of the panel above this level
    if up < len(heights):
        for corner in names:
            model.add_frame(f'leg {level}{corner}', joint(level, corner),
                            joint(up, corner), 'iron', chord_section(z),
                            axial=True)

        # X-brace on each face of the panel
        for i, corner in enumerate(names):
            nxt = names[(i + 1) % 4]
            model.add_frame(f'brace {level}{corner}{nxt}',
                            joint(level, corner), joint(up, nxt),
                            'iron', 'diagonal', axial=True)
            model.add_frame(f'brace {level}{nxt}{corner}',
                            joint(level, nxt), joint(up, corner),
                            'iron', 'diagonal', axial=True)

    # horizontal ring and plan diagonals (the feet are pinned instead)
    if level > 0:
        for i, corner in enumerate(names):
            nxt = names[(i + 1) % 4]
            model.add_frame(f'ring {level}{corner}{nxt}',
                            joint(level, corner), joint(level, nxt),
                            'iron', 'horizontal', axial=True)
        model.add_frame(f'plan {level}ac', joint(level, 'a'),
                        joint(level, 'c'), 'iron', 'horizontal', axial=True)
        model.add_frame(f'plan {level}bd', joint(level, 'b'),
                        joint(level, 'd'), 'iron', 'horizontal', axial=True)

# 6. Add supports: pinned feet
for corner in names:
    model.add_support(joint(0, corner), r_ux=True, r_uy=True, r_uz=True)

# 7. Add loads
# self weight of each bar, half at each end joint
model.add_load_pattern('self weight')

total_weight = 0
for frame in model.frames.values():
    area = model.sections[frame.section].A
    weight = unit_weight * area * frame.length()
    total_weight += weight
    for end in (frame.joint_j, frame.joint_k):
        model.add_joint_point_load('self weight', end, fz=-weight / 2)

# wind along +x, on the tributary height of each level
model.add_load_pattern('wind')

solidity = 0.25
total_wind = 0
for level, z in enumerate(heights):
    if level == 0:
        continue  # the load at the feet goes straight to the supports
    below = heights[level - 1]
    above = heights[level + 1] if level + 1 < len(heights) else z
    tributary = (above - below) / 2
    pressure = 1.5 + 1.5 * z / heights[-1]
    force = pressure * solidity * width(z) * tributary
    total_wind += force
    for corner in names:
        model.add_joint_point_load('wind', joint(level, corner),
                                   fx=force / 4)

# 8. Analyze the model
model.run_analysis()
model.export('eiffel_tower.json')

# 9. Results
top = len(heights) - 1


def axial_forces(load_pattern, prefix):
    """Return the axial forces of the frames whose name starts with prefix.

    Tension is positive.
    """
    return {name: forces.fx[0] for name, forces in
            model.internal_forces[load_pattern].items()
            if name.startswith(prefix)}


def kN(force):
    """Format a force in kN, without a negative sign on rounded zeros."""
    return f'{0 if abs(force) < 0.5 else force:,.0f} kN'


print(f'joints: {len(model.joints)}, frames: {len(model.frames)}')
print(f'weight of iron: {kN(total_weight)} '
      f'({total_weight / 9.81:,.0f} t, real tower: about 7,300 t)')

for load_pattern in model.load_patterns:
    reactions = model.reactions[load_pattern].values()
    print(f'\n{load_pattern}')
    print(f'\tsum of reactions: fx = {kN(sum(r.fx for r in reactions))}, '
          f'fz = {kN(sum(r.fz for r in reactions))}')

    top_a = model.displacements[load_pattern][joint(top, 'a')]
    print(f'\ttop displacement: ux = {top_a.ux * 100:+.1f} cm, '
          f'uz = {top_a.uz * 100:+.1f} cm')

    legs = axial_forces(load_pattern, 'leg 0')
    print('\taxial force in the legs at the base (tension +):')
    for name, force in legs.items():
        print(f'\t\t{name}: {force:+,.0f} kN')

print(f'\nwind force: {kN(total_wind)}')

# 10. Plot the results, if matplotlib is available
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.ticker import StrMethodFormatter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except ImportError:
    print('\ninstall matplotlib to plot the results')
    raise SystemExit

# colors: diverging blue (compression) - gray (zero) - red (tension)
surface, ink, muted, gray = '#fcfcfb', '#0b0b0b', '#52514e', '#c3c2b7'
blue, red = '#2a78d6', '#e34948'
diverging = LinearSegmentedColormap.from_list('axial', [blue, gray, red])


def segments(scale=0, load_pattern=None):
    """Return the end points of every frame, optionally displaced.

    Args:
        scale (float, optional): Factor applied to the joint displacements.
        load_pattern (str, optional): Load pattern of the displacements.

    Returns:
        ndarray: Array of shape (number of frames, 2, 3).
    """
    def position(name):
        p = model.joints[name].position()
        if scale:
            d = model.displacements[load_pattern][name]
            p = p + scale * np.array([d.ux, d.uy, d.uz])
        return p

    return np.array([[position(f.joint_j), position(f.joint_k)]
                     for f in model.frames.values()])


def setup(ax, title):
    """Set the title, limits and a recessive look of a 3D axes."""
    ax.set_title(title, color=ink, fontsize=11)
    ax.set_xlim(-65, 65)
    ax.set_ylim(-65, 65)
    ax.set_zlim(0, 300)
    ax.set_box_aspect((130, 130, 300))
    ax.view_init(elev=12, azim=-60)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([0, 57.6, 115.7, 276.1])
    ax.set_zticklabels(['0 m', '1st floor', '2nd floor', '3rd floor'],
                       color=muted, fontsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color(gray)
    ax.grid(False)


fig = plt.figure(figsize=(12, 7), facecolor=surface)
fig.suptitle('Eiffel Tower (simplified space truss)', color=ink, fontsize=13)
# three panels of the same size, with a color bar below the first two
grid = fig.add_gridspec(2, 3, height_ratios=(1, 0.025), left=0.03,
                        right=0.97, bottom=0.08, top=0.9, hspace=0,
                        wspace=0.1)

# axial forces, one panel per load pattern
for i, load_pattern in enumerate(model.load_patterns):
    ax = fig.add_subplot(grid[0, i], projection='3d', facecolor=surface)
    forces = np.array([model.internal_forces[load_pattern][name].fx[0]
                       for name in model.frames])
    limit = np.abs(forces).max()
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    lines = Line3DCollection(segments(), cmap=diverging, norm=norm,
                             linewidths=1.2)
    lines.set_array(forces)
    ax.add_collection(lines)
    setup(ax, f'Axial force: {load_pattern}')
    cax = fig.add_subplot(grid[1, i])
    box = cax.get_position()
    cax.set_position([box.x0 + 0.15 * box.width, box.y0, 0.7 * box.width,
                      box.height])
    bar = fig.colorbar(lines, cax=cax, orientation='horizontal')
    # ticks at zero and at the largest force rounded down to a thousand
    tick = np.floor(limit / 1000) * 1000
    bar.set_ticks([-tick, 0, tick])
    bar.ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    bar.set_label('kN (compression -, tension +)', color=muted, fontsize=8)
    bar.ax.tick_params(colors=muted, labelsize=8)
    bar.outline.set_visible(False)

# deformed shape under wind, exaggerated
scale = 500
ax = fig.add_subplot(grid[0, 2], projection='3d', facecolor=surface)
ax.add_collection(Line3DCollection(segments(), colors=gray, linewidths=0.8))
ax.add_collection(Line3DCollection(segments(scale, 'wind'), colors=blue,
                                   linewidths=1.2))
sway = model.displacements['wind'][joint(top, 'a')].ux
setup(ax, f'Deformed shape: wind (×{scale})\n'
          f'top sway {sway * 100:.1f} cm, gray: undeformed')

fig.savefig('eiffel_tower.png', dpi=150, facecolor=surface)
plt.show()
