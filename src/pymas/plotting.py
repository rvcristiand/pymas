"""Plots of the analysis results of a structure.

The functions of this module are used through the `plot_*` methods of
`Structure`. They require matplotlib, which is an optional dependency of
pymas (`pip install pymas[plot]`), and return the matplotlib figure or axes
they draw on, without showing it.

A structure is drawn in 2D when all its joints lie in the XY plane and the
translation uz and the rotations rx and ry are not active degrees of freedom;
otherwise it is drawn in 3D.
"""

import numpy as np

# colors of the plots: neutral gray for the geometry, a blue <-> red diverging
# pair for signed values (blue: negative, red: positive)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
MUTED = '#52514e'
GRAY = '#c3c2b7'
BLUE = '#2a78d6'
RED = '#e34948'

# flag of the frame that activates each internal force and displacement
FORCE_FLAGS = {'fx': 'axial', 'fy': 'bending_z', 'fz': 'bending_y',
               'mx': 'torsional', 'my': 'bending_y', 'mz': 'bending_z'}
DISPLACEMENT_FLAGS = {'ux': 'axial', 'uy': 'bending_z', 'uz': 'bending_y',
                      'rx': 'torsional', 'ry': 'bending_y', 'rz': 'bending_z'}

LABELS = {
    'fx': 'Axial force fx', 'fy': 'Shear force fy', 'fz': 'Shear force fz',
    'mx': 'Torque mx', 'my': 'Bending moment my', 'mz': 'Bending moment mz',
    'ux': 'Axial displacement ux', 'uy': 'Deflection uy',
    'uz': 'Deflection uz', 'rx': 'Twist rx', 'ry': 'Rotation ry',
    'rz': 'Rotation rz',
}

# local axis (1: y, 2: z) and side (+1 or -1) along which each internal force
# is drawn on the structure: bending moments on the tension side, forces and
# torque along the positive local axis
DIAGRAM_DIRECTIONS = {'fx': (1, 1), 'fy': (1, 1), 'fz': (2, 1),
                      'mx': (1, 1), 'my': (2, -1), 'mz': (1, -1)}


def _pyplot():
    """Return matplotlib.pyplot, or raise an ImportError with a hint."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('plotting requires matplotlib; install it with '
                          '`pip install pymas[plot]`') from None

    return plt


def _check_results(structure, load_pattern):
    """Raise a ValueError if the load pattern has no results."""
    if load_pattern not in structure.load_patterns:
        raise ValueError(f'there is no load pattern {load_pattern!r}')
    if load_pattern not in structure.displacements:
        raise ValueError(f'load pattern {load_pattern!r} has no results; '
                         'call run_analysis() first')


def is_planar(structure):
    """Return whether a structure is drawn in 2D.

    Args:
        structure (Structure): Structure.

    Returns:
        bool: True if all the joints lie in the XY plane and uz, rx and ry
            are not active degrees of freedom.
    """
    in_plane = all(not joint.z for joint in structure.joints.values())
    out_of_plane_dof = structure.get_degrees_freedom()[2:5].any()

    return in_plane and not out_of_plane_dof


def _style(ax, planar):
    """Give an axes a recessive look."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    if planar:
        for spine in ax.spines.values():
            spine.set_color(GRAY)
    else:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))
            axis.line.set_color(GRAY)
        ax.grid(False)


def _axes(structure, ax):
    """Return the axes to draw a structure on, creating it if needed."""
    if ax is not None:
        return ax

    plt = _pyplot()
    planar = is_planar(structure)
    fig = plt.figure(facecolor=SURFACE)
    ax = fig.add_subplot(projection=None if planar else '3d')
    _style(ax, planar)

    return ax


def _line(ax, points, **kwargs):
    """Draw a polyline of 3D points in a 2D or 3D axes."""
    if ax.name == '3d':
        return ax.plot(points[:, 0], points[:, 1], points[:, 2], **kwargs)

    return ax.plot(points[:, 0], points[:, 1], **kwargs)


def _set_limits(ax, points):
    """Fit the limits of an axes to some 3D points, with equal scales."""
    points = np.asarray(points)
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    size = max((high - low).max(), 1e-9) * 0.55

    if ax.name == '3d':
        # a flat direction keeps a small depth, so the view is not squashed
        span = np.maximum((high - low) / 2 * 1.1, 0.05 * size)
        ax.set_xlim(center[0] - span[0], center[0] + span[0])
        ax.set_ylim(center[1] - span[1], center[1] + span[1])
        ax.set_zlim(center[2] - span[2], center[2] + span[2])
        ax.set_box_aspect(2 * span)
    else:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(center[0] - size, center[0] + size)
        ax.set_ylim(center[1] - size, center[1] + size)


def _size(structure):
    """Return the size of a structure: the diagonal of its bounding box."""
    positions = np.array([joint.position()
                          for joint in structure.joints.values()])

    return np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))


def frame_positions(frame, n):
    """Return n equally spaced points along a frame, in global axes.

    Args:
        frame (Frame): Frame.
        n (int): Number of points, including both ends.

    Returns:
        ndarray: Points, of shape (n, 3).
    """
    start = frame._parent.joints[frame.joint_j].position()
    x = np.linspace(0, frame.length(), n)

    return start + x[:, None] * frame.direction_cosines_vector()


def deformed_positions(structure, load_pattern, frame, scale):
    """Return the deformed points of a frame, in global axes.

    The internal displacements are used for the local components that the
    frame models (e.g. uy with bending_z); the others are interpolated
    linearly between the displacements of the joints, as a rigid motion.

    Args:
        structure (Structure): Structure.
        load_pattern (str): Name of the load pattern.
        frame (Frame): Frame.
        scale (float): Factor applied to the displacements.

    Returns:
        ndarray: Points, of shape (number of internal points, 3).
    """
    internal = structure.internal_displacements[load_pattern][frame.name]
    displacements = structure.displacements[load_pattern]
    n = len(internal.ux)

    r = frame.rotation_matrix()
    u_j = r.T @ displacements[frame.joint_j].displacement_vector()[:3]
    u_k = r.T @ displacements[frame.joint_k].displacement_vector()[:3]
    t = np.linspace(0, 1, n)

    local = np.empty((n, 3))
    for i, component in enumerate(('ux', 'uy', 'uz')):
        if getattr(frame, DISPLACEMENT_FLAGS[component]):
            local[:, i] = getattr(internal, component)
        else:
            local[:, i] = (1 - t) * u_j[i] + t * u_k[i]

    return frame_positions(frame, n) + scale * local @ r.T


def _auto_scale(structure, load_pattern):
    """Return the scale that makes the largest displacement 10% of the size.

    Returns 1 if the structure does not move.
    """
    largest = 0
    for frame in structure.frames.values():
        moved = deformed_positions(structure, load_pattern, frame, 1)
        largest = max(largest, np.abs(moved - frame_positions(
            frame, len(moved))).max())

    if largest == 0:
        return 1

    return 0.1 * _size(structure) / largest


def plot_deformed(structure, load_pattern, scale=None, ax=None):
    """Plot the deformed shape of a structure.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        scale (float, optional): Factor applied to the displacements. By
            default, the largest displacement is drawn as 10% of the size of
            the structure.
        ax (Axes, optional): Axes to draw on (3D for a structure that is not
            planar). By default, a new figure is created.

    Returns:
        Axes: Axes of the plot.
    """
    _check_results(structure, load_pattern)
    ax = _axes(structure, ax)
    if scale is None:
        scale = _auto_scale(structure, load_pattern)

    points = []
    for i, frame in enumerate(structure.frames.values()):
        undeformed = frame_positions(frame, 2)
        deformed = deformed_positions(structure, load_pattern, frame, scale)
        _line(ax, undeformed, color=GRAY, linewidth=1,
              label='undeformed' if i == 0 else None)
        _line(ax, deformed, color=BLUE, linewidth=1.5,
              label='deformed' if i == 0 else None)
        points.extend((*undeformed, *deformed))

    _set_limits(ax, points)
    ax.set_title(f'Deformed shape: {load_pattern} (×{scale:.3g})',
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)

    return ax


def _plot_frame_series(structure, load_pattern, frame_name, results, flags,
                       kind):
    """Plot the active internal results of a frame against its length."""
    plt = _pyplot()
    frame = structure.frames[frame_name]
    active = [c for c, flag in flags.items() if getattr(frame, flag)]
    if not active:
        raise ValueError(f'frame {frame_name!r} has no active {kind}; set '
                         'its axial, torsional or bending flags')

    fig, axes = plt.subplots(len(active), 1, sharex=True, squeeze=False,
                             figsize=(7, 1.9 * len(active) + 0.6),
                             facecolor=SURFACE)
    fig.suptitle(f'Internal {kind} of frame {frame_name!r}: {load_pattern}',
                 color=INK, fontsize=11)

    values_of = results[load_pattern][frame_name]
    x = np.linspace(0, frame.length(), len(values_of.fx if kind == 'forces'
                                           else values_of.ux))

    for ax, component in zip(axes[:, 0], active):
        values = np.asarray(getattr(values_of, component))
        _style(ax, True)
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.fill_between(x, values, where=values >= 0, color=RED, alpha=0.2,
                        linewidth=0, interpolate=True)
        ax.fill_between(x, values, where=values <= 0, color=BLUE, alpha=0.2,
                        linewidth=0, interpolate=True)
        ax.plot(x, values, color=INK, linewidth=1.5)
        ax.set_ylabel(LABELS[component], color=MUTED, fontsize=8)

        # label the largest absolute value only
        i = int(np.argmax(np.abs(values)))
        if values[i] != 0:
            ax.annotate(f'{values[i]:.4g}', (x[i], values[i]),
                        textcoords='offset points', xytext=(4, 4),
                        color=INK, fontsize=8)

    axes[-1, 0].set_xlabel(f'x from joint {frame.joint_j!r}', color=MUTED,
                           fontsize=8)
    fig.tight_layout()

    return fig


def plot_internal_forces(structure, load_pattern, frame):
    """Plot the internal forces of a frame along its length.

    Only the forces that the frame models are plotted (e.g. fy and mz for a
    frame with bending_z).

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        frame (str): Name of the frame.

    Returns:
        Figure: Figure with one axes per internal force.
    """
    _check_results(structure, load_pattern)

    return _plot_frame_series(structure, load_pattern, frame,
                              structure.internal_forces, FORCE_FLAGS,
                              'forces')


def plot_internal_displacements(structure, load_pattern, frame):
    """Plot the internal displacements of a frame along its length.

    Only the displacements that the frame models are plotted (e.g. uy and rz
    for a frame with bending_z). They are given in the local axes of the
    frame.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        frame (str): Name of the frame.

    Returns:
        Figure: Figure with one axes per internal displacement.
    """
    _check_results(structure, load_pattern)

    return _plot_frame_series(structure, load_pattern, frame,
                              structure.internal_displacements,
                              DISPLACEMENT_FLAGS, 'displacements')


def diagram_positions(structure, load_pattern, frame, force, scale):
    """Return the points of the diagram of an internal force of a frame.

    Args:
        structure (Structure): Structure.
        load_pattern (str): Name of the load pattern.
        frame (Frame): Frame.
        force (str): Internal force: 'fx', 'fy', 'fz', 'mx', 'my' or 'mz'.
        scale (float): Length per unit of force.

    Returns:
        tuple: Points along the frame and points of the diagram, both of
            shape (number of internal points, 3), and the values.
    """
    values = np.asarray(getattr(
        structure.internal_forces[load_pattern][frame.name], force))
    axis, side = DIAGRAM_DIRECTIONS[force]
    if is_planar(structure):
        axis = 1  # local z is out of the plane
    direction = frame.rotation_matrix()[:, axis]

    base = frame_positions(frame, len(values))
    tip = base + side * scale * values[:, None] * direction

    return base, tip, values


def _same_sign_polygons(base, tip, values):
    """Split the area between a frame and its diagram by the sign of values.

    Args:
        base (ndarray): Points along the frame, of shape (n, 3).
        tip (ndarray): Points of the diagram, of shape (n, 3).
        values (ndarray): Values, of shape (n,).

    Returns:
        list: (polygon, sign) pairs, where polygon is an array of 3D points
            and sign is 1 or -1. Each polygon ends at the zeros of values.
    """
    polygons = []
    start_base, start_tip = [base[0]], [tip[0]]
    sign = 1 if values[0] >= 0 else -1

    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if a * b < 0:
            # the values cross zero between i and i + 1
            t = a / (a - b)
            zero = base[i] + t * (base[i + 1] - base[i])
            start_base.append(zero)
            start_tip.append(zero)
            polygons.append((np.array(start_base + start_tip[::-1]), sign))
            start_base, start_tip = [zero], [zero]
            sign = -sign
        elif a == 0 and b != 0:
            sign = 1 if b > 0 else -1
        start_base.append(base[i + 1])
        start_tip.append(tip[i + 1])

    polygons.append((np.array(start_base + start_tip[::-1]), sign))

    return polygons


def plot_diagram(structure, load_pattern, force, scale=None, ax=None):
    """Plot the diagram of an internal force on the whole structure.

    The values are drawn perpendicular to each frame: bending moments on the
    tension side, and forces and torque along the positive local y-axis (z
    for fz). Positive values are red and negative values blue. Frames that do
    not model the force are drawn without diagram.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        force (str): Internal force: 'fx', 'fy', 'fz', 'mx', 'my' or 'mz'.
        scale (float, optional): Length per unit of force. By default, the
            largest value is drawn as 15% of the size of the structure.
        ax (Axes, optional): Axes to draw on (3D for a structure that is not
            planar). By default, a new figure is created.

    Returns:
        Axes: Axes of the plot.
    """
    _check_results(structure, load_pattern)
    if force not in FORCE_FLAGS:
        raise ValueError(f'force must be one of {", ".join(FORCE_FLAGS)}, '
                         f'not {force!r}')

    ax = _axes(structure, ax)
    frames = [f for f in structure.frames.values()
              if getattr(f, FORCE_FLAGS[force])]
    largest = max((np.abs(getattr(structure.internal_forces[load_pattern]
                                  [f.name], force)).max() for f in frames),
                  default=0)
    if scale is None:
        scale = 0.15 * _size(structure) / largest if largest else 1

    points = []
    polygons, colors = [], []
    # largest positive and most negative values, with their points
    highest, lowest = (0, None), (0, None)
    for frame in structure.frames.values():
        line = frame_positions(frame, 2)
        _line(ax, line, color=GRAY, linewidth=1)
        points.extend(line)
        if frame not in frames:
            continue

        base, tip, values = diagram_positions(structure, load_pattern, frame,
                                              force, scale)
        _line(ax, tip, color=MUTED, linewidth=0.8)
        points.extend(tip)
        for polygon, sign in _same_sign_polygons(base, tip, values):
            polygons.append(polygon)
            colors.append(RED if sign > 0 else BLUE)
        i, j = int(np.argmax(values)), int(np.argmin(values))
        if values[i] > highest[0]:
            highest = (values[i], tip[i])
        if values[j] < lowest[0]:
            lowest = (values[j], tip[j])

    if ax.name == '3d':
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax.add_collection3d(Poly3DCollection(polygons, facecolors=colors,
                                             alpha=0.35, linewidths=0))
    else:
        from matplotlib.collections import PolyCollection
        ax.add_collection(PolyCollection([p[:, :2] for p in polygons],
                                         facecolors=colors, alpha=0.35,
                                         linewidths=0))

    # label the largest positive and negative values only
    for value, at in (highest, lowest):
        if at is None:
            continue
        if ax.name == '3d':
            ax.text(*at, f'{value:.4g}', color=INK, fontsize=8)
        else:
            ax.annotate(f'{value:.4g}', at[:2], textcoords='offset points',
                        xytext=(4, 4), color=INK, fontsize=8)

    _set_limits(ax, points)
    ax.set_title(f'{LABELS[force]}: {load_pattern}', color=INK, fontsize=11)

    return ax
