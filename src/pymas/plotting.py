"""Plots of the analysis results of a structure.

The functions of this module are used through the `plot_*` methods of the
analysis results (`Results` and `FrameResults`), of `LoadPattern` and of
`Structure`. They draw with matplotlib, which is imported only when something
is plotted, and return the matplotlib figure or axes they draw on, without
showing it.

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
DARK_BLUE = '#1f5fae'
RED = '#e34948'
DARK_RED = '#b8302f'

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
    """Return matplotlib.pyplot, imported only when something is plotted."""
    import matplotlib.pyplot as plt

    return plt


def _check_results(structure, load_pattern):
    """Raise a ValueError if the load pattern has no results."""
    if load_pattern not in structure.load_patterns:
        raise ValueError(f'there is no load pattern {load_pattern!r}')
    if load_pattern not in structure.results:
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


def _title(text, load_pattern):
    """Return a title naming its load pattern, unless it is 'default'."""
    return text if load_pattern == 'default' else f'{text}: {load_pattern}'


def _format(value):
    """Format a value for a label: thousands without exponent."""
    if abs(value) >= 1000:
        return f'{value:,.0f}'

    return f'{value:.4g}'


def _style(ax, planar):
    """Give an axes a recessive look, over the figure background."""
    ax.set_facecolor((1, 1, 1, 0))
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
    """Return the styled axes to draw a structure on, creating it if needed."""
    if ax is None:
        plt = _pyplot()
        fig = plt.figure(facecolor=SURFACE)
        ax = fig.add_subplot(projection=None if is_planar(structure)
                             else '3d')
    _style(ax, ax.name != '3d')

    return ax


def _line(ax, points, **kwargs):
    """Draw a polyline of 3D points in a 2D or 3D axes."""
    if ax.name == '3d':
        return ax.plot(points[:, 0], points[:, 1], points[:, 2], **kwargs)

    return ax.plot(points[:, 0], points[:, 1], **kwargs)


def _sign_parts(points, values):
    """Split a curve into its positive and negative parts.

    The points where the values change sign between two samples are added
    to both parts, so they join; elsewhere each part is NaN.

    Args:
        points (ndarray): Points of the curve, of shape (n, dimensions).
        values (ndarray): Values at those points, of shape (n,).

    Returns:
        tuple: Positive and negative parts, as arrays of points.
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    joined, signs = [points[0]], [values[0]]
    for i in range(len(values) - 1):
        if values[i] * values[i + 1] < 0:
            t = values[i] / (values[i] - values[i + 1])
            joined.append(points[i] + t * (points[i + 1] - points[i]))
            signs.append(0)
        joined.append(points[i + 1])
        signs.append(values[i + 1])
    joined, signs = np.array(joined), np.array(signs)

    return (np.where((signs >= 0)[:, None], joined, np.nan),
            np.where((signs <= 0)[:, None], joined, np.nan))


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
        # fit the bounding box with a margin; equal scales are kept by
        # widening the data limits, so a long beam fills a wide axes
        margin = 0.05 * size
        ax.update_datalim([low[:2] - margin, high[:2] + margin])
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale_view()


def support_kind(structure, support):
    """Return the kind of a support, for its symbol.

    A support is drawn only if it restrains an active degree of freedom.
    It is fixed if it restrains every active degree of freedom and some
    rotation is active. Otherwise it is pinned if it restrains every
    translation of the view (ux and uy in 2D, also uz in 3D), and a roller
    if not. A translation counts as restrained if it is active and
    restrained, or inactive but restrained as written by the user: in a
    beam, which does not model ux, a support with r_ux and r_uy is drawn
    pinned and one with r_uy alone a roller.

    Args:
        structure (Structure): Structure.
        support (Support): Support.

    Returns:
        str: 'fixed', 'pinned', 'roller', or None if the support restrains
            no active degree of freedom.
    """
    dof = structure.get_degrees_freedom()
    written = np.array([bool(getattr(support, f'r_{name}')) for name in
                        ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')])
    restrained = written & dof
    if not restrained.any():
        return None
    if restrained[dof].all() and dof[3:].any():
        return 'fixed'

    translations = 2 if is_planar(structure) else 3
    if written[:translations].all():
        return 'pinned'

    return 'roller'


def _support_size(structure):
    """Return the size of the support symbols of a structure."""
    return 0.035 * _size(structure)


def _support_direction(structure, support, kind):
    """Return the 2D direction from a joint to its support symbol.

    A fixed support lies on the side away from the frames of its joint; a
    pinned support or a roller against a restrained translation, uy first.
    """
    if kind == 'fixed':
        away = np.zeros(2)
        for frame in structure.frames.values():
            if support.joint in (frame.joint_j, frame.joint_k):
                sign = 1 if support.joint == frame.joint_j else -1
                away -= sign * frame.direction_cosines_vector()[:2]
        norm = np.linalg.norm(away)
        return away / norm if norm else np.array([0, -1])

    dof = structure.get_degrees_freedom()
    return np.array([0, -1]) if support.r_uy and dof[1] else np.array([-1, 0])


def _support_depth(kind):
    """Return how far a support symbol reaches from its joint, in sizes."""
    return 0.35 if kind == 'fixed' else 1.3


def _draw_supports(ax, structure):
    """Draw a symbol at every support, and return the points it covers.

    In 2D a pinned support is a triangle and a roller a circle, both on a
    hatched ground line and touching the joint on the side of a restrained
    translation; a fixed support is a hatched wall on
    the side away from its frames. In 3D they are markers: a triangle, or a
    square for a fixed support.
    """
    from matplotlib.collections import (LineCollection, PatchCollection,
                                        PolyCollection)
    from matplotlib.patches import Circle

    h = _support_size(structure)
    triangles, lines, circles, points = [], [], [], []
    for support in structure.supports.values():
        kind = support_kind(structure, support)
        if kind is None:
            continue
        p = structure.joints[support.joint].position()

        if ax.name == '3d':
            ax.scatter(*p, color=MUTED, s=30, depthshade=False,
                       marker='s' if kind == 'fixed' else '^')
            points.append(p)
            continue

        p = p[:2]
        d = _support_direction(structure, support, kind)
        n = np.array([-d[1], d[0]])

        if kind == 'fixed':
            lines.append([p - 0.7 * h * n, p + 0.7 * h * n])
            for t in np.linspace(-0.7, 0.7, 5):
                start = p + t * h * n
                lines.append([start, start + 0.35 * h * (d - n)])
            points.extend((p - 0.7 * h * n + 0.35 * h * (d - n),
                           p + 0.7 * h * n + 0.35 * h * d))
        else:
            # a triangle with its apex at the joint (a circle that touches
            # it for a roller), on a line
            base = p + h * d
            if kind == 'pinned':
                triangles.append([p, base - 0.6 * h * n,
                                  base + 0.6 * h * n])
            else:
                circles.append(Circle(p + 0.5 * h * d, 0.5 * h))
            lines.append([base - 0.9 * h * n, base + 0.9 * h * n])
            points.extend(lines[-1])
            # hatches at 45 degrees under the line
            for t in np.linspace(-0.8, 0.8, 5)[1:]:
                start = base + t * h * n
                lines.append([start, start + 0.3 * h * (d - n)])
            points.append(base - 0.9 * h * n + 0.3 * h * (d - n))

    if triangles:
        ax.add_collection(PolyCollection(triangles, facecolors=SURFACE,
                                         edgecolors=MUTED, linewidths=1,
                                         zorder=3))
    if lines:
        ax.add_collection(LineCollection(lines, colors=MUTED, linewidths=1,
                                         zorder=3))
    if circles:
        ax.add_collection(PatchCollection(circles, facecolors=SURFACE,
                                          edgecolors=MUTED, linewidths=1,
                                          zorder=3))

    return [np.append(point, 0) if len(point) == 2 else point
            for point in points]


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
    results = structure.results[load_pattern]
    internal = results.frames[frame.name].internal_displacements
    displacements = results.displacements
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


def largest_displacement(structure, load_pattern):
    """Return the point of a structure that moves the most.

    The displacement of a point is the magnitude of its translation,
    sqrt(ux^2 + uy^2 + uz^2), searched over the internal points of every
    frame (the joints included). Rotations are not considered.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.

    Returns:
        tuple: Magnitude of the displacement, frame and index of the
            internal point; (0, None, None) if the structure does not move.
    """
    largest = (0, None, None)
    for frame in structure.frames.values():
        moved = deformed_positions(structure, load_pattern, frame, 1)
        magnitudes = np.linalg.norm(
            moved - frame_positions(frame, len(moved)), axis=1)
        i = int(np.argmax(magnitudes))
        if magnitudes[i] > largest[0]:
            largest = (magnitudes[i], frame, i)

    return largest


def largest_deflections(structure, load_pattern, scale):
    """Return the largest deflection of each frame that bends.

    The deflections are the local uy of the frames with bending_z and the
    local uz of the frames with bending_y. A deflection shared by two frames
    at their joint is returned once, and zero deflections are left out.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        scale (float): Factor applied to the displacements.

    Returns:
        list: (value, point, direction, component, frame, distance) tuples,
            where value is the signed deflection, point is its deformed
            position, direction is the magnified displacement there, in
            global axes, component is 'uy' or 'uz', and distance is how far
            along the frame it is, from its joint j.
    """
    tolerance = 1e-6 * largest_displacement(structure, load_pattern)[0]
    results = structure.results[load_pattern].frames
    deflections = []
    for frame in structure.frames.values():
        internal = results[frame.name].internal_displacements
        moved = deformed_positions(structure, load_pattern, frame, scale)
        base = frame_positions(frame, len(moved))
        for component in ('uy', 'uz'):
            if not getattr(frame, DISPLACEMENT_FLAGS[component]):
                continue
            values = np.asarray(getattr(internal, component))
            i = int(np.argmax(np.abs(values)))
            if abs(values[i]) <= tolerance or any(
                    _format(values[i]) == _format(d[0]) and
                    np.allclose(moved[i], d[1]) for d in deflections):
                continue
            distance = i / (len(values) - 1) * frame.length()
            deflections.append((values[i], moved[i], moved[i] - base[i],
                                component, frame, distance))

    return deflections


def _auto_scale(structure, load_pattern):
    """Return the scale that makes the largest displacement 10% of the size.

    Returns 1 if the structure does not move.
    """
    largest, _, _ = largest_displacement(structure, load_pattern)
    if largest == 0:
        return 1

    return 0.1 * _size(structure) / largest


def plot_deformed(structure, load_pattern, ax=None, scale=None,
                  labels=False, dimensions=False):
    """Plot the deformed shape of a structure.

    The displacements are magnified by the scale, so the plot has no axes:
    its coordinates do not measure them. Optionally, the largest deflection
    of each frame that bends is labelled with its real, signed value: the
    local uy with bending_z, and the local uz with bending_y; and, in a
    plane structure, its position is dimensioned from the ends of its frame
    (horizontal and vertical frames only).

    The supports are drawn at their joints, on the undeformed structure.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        ax (Axes, optional): Axes to draw on (3D for a structure that is not
            planar). By default, a new figure is created.
        scale (float, optional): Factor applied to the displacements. By
            default, the largest displacement is drawn as 10% of the size of
            the structure.
        labels (bool, optional): Whether to label the largest deflection of
            each frame. False by default.
        dimensions (bool, optional): Whether to dimension the position of
            the largest deflection of each frame. False by default.

    Returns:
        Axes: Axes of the plot.
    """
    _check_results(structure, load_pattern)
    ax = _axes(structure, ax)
    if scale is None:
        scale = _auto_scale(structure, load_pattern)

    points = []
    moved = {}
    for i, frame in enumerate(structure.frames.values()):
        undeformed = frame_positions(frame, 2)
        deformed = deformed_positions(structure, load_pattern, frame, scale)
        moved[frame.name] = deformed
        _line(ax, undeformed, color=GRAY, linewidth=1,
              label='undeformed' if i == 0 else None)
        _line(ax, deformed, color=INK, linewidth=1.25,
              label='deformed' if i == 0 else None)
        points.extend((*undeformed, *deformed))

    points.extend(_draw_supports(ax, structure))

    # label the largest deflection of each frame beyond its point, and leave
    # room for the label, so it does not reach the title; near the right
    # end of the structure the label goes on the left of its point, so it
    # does not stick out of the plot
    size = _size(structure)
    x = [joint.position()[0] for joint in structure.joints.values()]
    right = min(x) + 0.9 * (max(x) - min(x))
    deflections = largest_deflections(structure, load_pattern, scale)
    for value, at, out, *_ in deflections if labels else []:
        label = _format(value)
        out = out / np.linalg.norm(out)
        points.append(at + 0.04 * size * out)
        if ax.name == '3d':
            ax.scatter(*at, color=INK, s=12)
            ax.text(*at, f' {label}', color=INK, fontsize=8)
        else:
            up, left = out[1] >= 0, at[0] > right
            ax.scatter(*at[:2], color=INK, s=12, zorder=3)
            ax.annotate(label, at[:2], textcoords='offset points',
                        xytext=(-4 if left else 4, 4 if up else -4),
                        ha='right' if left else 'left',
                        va='bottom' if up else 'top', color=INK, fontsize=8)
    if dimensions and ax.name != '3d':
        drawings = [(frame, frame_positions(frame, len(moved[frame.name])),
                     moved[frame.name], [distance])
                    for *_, frame, distance in deflections]
        points.extend(_draw_frame_dimensions(
            ax, drawings, size, 0.04 * size if labels else 0))

    _set_limits(ax, points)
    ax.set_axis_off()
    ax.set_title(f'{_title("Deformed shape", load_pattern)} '
                 f'(displacements ×{scale:.3g})', color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)

    return ax


def extreme_indices(values):
    """Return where some values reach their largest and most negative.

    A value reached at several separate places, e.g. at both ends of a
    fixed beam, gives the first index of each of them. The largest value is
    left out unless positive, and the most negative unless negative.

    Args:
        values (ndarray): Values.

    Returns:
        list: Sorted indices.
    """
    values = np.asarray(values, dtype=float)
    tolerance = 1e-6 * np.abs(values).max() if len(values) else 0
    indices = []
    for extreme, sign in ((values.max(), 1), (values.min(), -1)):
        if sign * extreme <= tolerance:
            continue
        reached = np.flatnonzero(np.abs(values - extreme) <= tolerance)
        # the first index of each run of consecutive indices
        indices += [int(k) for n, k in enumerate(reached)
                    if n == 0 or k != reached[n - 1] + 1]

    return sorted(set(indices))


def zero_crossings(x, values):
    """Return where some values sampled along a frame change sign.

    A crossing between two samples is interpolated linearly; zeros at the
    ends of the frame, and values that touch zero without changing sign,
    are not crossings.

    Args:
        x (ndarray): Distances along the frame.
        values (ndarray): Values at those distances.

    Returns:
        list: Distances of the crossings.
    """
    values = np.asarray(values, dtype=float)
    tolerance = 1e-6 * np.abs(values).max()
    signs = np.where(np.abs(values) <= tolerance, 0, np.sign(values))
    nonzero = np.flatnonzero(signs)

    crossings = []
    for p, q in zip(nonzero[:-1], nonzero[1:]):
        if signs[p] == signs[q]:
            continue
        if q == p + 1:
            crossings.append(x[p] - values[p] * (x[q] - x[p]) /
                             (values[q] - values[p]))
        else:
            # the zero falls on the samples between them
            crossings.append((x[p + 1] + x[q - 1]) / 2)

    return crossings


def _plot_frame_series(structure, load_pattern, frame_name, attribute,
                       flags, kind, axes=None, labels=False,
                       components=None):
    """Plot the active internal results of a frame against its length.

    The results are drawn on the given axes, one per component, each with
    its own title, or on a new figure; the bending moments are positive
    downwards. The components are
    the active ones, or the given ones, which must be active. Optionally,
    the largest positive and most negative values are marked with their
    coordinates, and so are the points where the values cross zero.
    """
    plt = _pyplot()
    frame = structure.frames[frame_name]
    active = [c for c, flag in flags.items() if getattr(frame, flag)]
    if not active:
        raise ValueError(f'frame {frame_name!r} has no active {kind}; set '
                         'its axial, torsional or bending flags')
    if components is not None:
        missing = [c for c in components if c not in active]
        if missing:
            raise ValueError(f'frame {frame_name!r} does not model '
                             f'{", ".join(missing)}; its active {kind} are '
                             f'{", ".join(active)}')
        active = [c for c in active if c in components]

    if axes is None:
        fig, axes = plt.subplots(len(active), 1, sharex=True, squeeze=False,
                                 figsize=(7, 2.2 * len(active) + 0.3),
                                 facecolor=SURFACE)
        axes = list(axes[:, 0])
        new = True
    else:
        axes = list(np.ravel(axes))
        if len(axes) != len(active):
            raise ValueError(f'frame {frame_name!r} has {len(active)} active '
                             f'{kind} ({", ".join(active)}), so it needs '
                             f'{len(active)} axes, not {len(axes)}')
        fig = axes[0].figure
        new = False

    values_of = getattr(
        structure.results[load_pattern].frames[frame_name], attribute)
    x = np.linspace(0, frame.length(), len(values_of.fx if kind == 'forces'
                                           else values_of.ux))

    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    for ax, component in zip(axes, active):
        values = np.asarray(getattr(values_of, component))
        _style(ax, True)
        # the y-axis stands at the start of the frame and the x-axis spans
        # its length, without a box around
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_bounds(0, x[-1])
        # a light grid behind the values, over the length of the frame only
        frame_band = Rectangle((0, 0), x[-1], 1, transform=
                               blended_transform_factory(ax.transData,
                                                         ax.transAxes))
        ax.grid(True, color=GRAY, linewidth=0.5, alpha=0.6,
                clip_path=frame_band)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot([0, x[-1]], [0, 0], color=GRAY, linewidth=1)
        ax.fill_between(x, values, where=values >= 0, color=BLUE, alpha=0.2,
                        linewidth=0, interpolate=True)
        ax.fill_between(x, values, where=values <= 0, color=RED, alpha=0.2,
                        linewidth=0, interpolate=True)
        # the curve is dark blue where positive and dark red where negative
        positive, negative = _sign_parts(np.column_stack((x, values)), values)
        ax.plot(*positive.T, color=DARK_BLUE, linewidth=1)
        ax.plot(*negative.T, color=DARK_RED, linewidth=1)
        # each component has its own title, so the y-axis gives its symbol
        ax.set_title(_title(f'{LABELS[component]} of frame {frame_name!r}',
                            load_pattern), color=INK, fontsize=11)
        ax.set_ylabel(component, color=MUTED, fontsize=8)
        # bending moments are positive downwards, on the tension side, as in
        # the diagrams on the structure
        inverted = kind == 'forces' and DIAGRAM_DIRECTIONS[component][1] < 0
        if inverted:
            ax.invert_yaxis()
        if not labels or not np.any(values):
            continue

        # mark the largest positive and most negative values and the zero
        # crossings, labelled with their coordinates beyond the curve, with
        # room for them inside the axes; near the far end of the frame the
        # label goes on the left of its point
        ax.margins(y=0.2)
        points = [(x[i], values[i]) for i in extreme_indices(values)]
        points += [(at, 0) for at in zero_crossings(x, values)]
        for at, value in points:
            up = value == 0 or (value > 0) != inverted
            left = at > 0.9 * x[-1]
            ax.scatter(at, value, color=MUTED, s=12, zorder=3)
            ax.annotate(f'({_format(at)}, {_format(value)})', (at, value),
                        textcoords='offset points',
                        xytext=(-4 if left else 4, 4 if up else -4),
                        ha='right' if left else 'left',
                        va='bottom' if up else 'top', color=INK, fontsize=8)

    # only the lowest of the element plots that share x shows the x axis:
    # the ones above lose it, and this one gets it unless one lies below
    last = axes[-1]
    shared = [ax for ax in fig.axes if ax is not last and
              getattr(ax, '_pymas_x_label', False) and
              last.get_shared_x_axes().joined(ax, last)]
    below = False
    for ax in shared:
        if ax.get_position().y0 < last.get_position().y0:
            below = True
        else:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
            ax._pymas_x_label = False
    if not below:
        last.set_xlabel(f'x from joint {frame.joint_j!r}', color=MUTED,
                        fontsize=8)
        # given axes may share x with others below that hide the tick
        # labels
        last.tick_params(labelbottom=True)
        last._pymas_x_label = True
    if new:
        fig.tight_layout()

    return fig


def plot_internal_forces(structure, load_pattern, frame, axes=None,
                         labels=False):
    """Plot the internal forces of a frame along its length.

    Only the forces that the frame models are plotted (e.g. fy and mz for a
    frame with bending_z). The bending moments are positive downwards, on
    the tension side. Optionally, the largest positive and most negative
    values are marked with their coordinates, and so are the points where
    they cross zero.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        frame (str): Name of the frame.
        axes (list, optional): Axes to draw on, one per internal force that
            the frame models, in the order fx, fy, fz, mx, my, mz. By
            default, a new figure is created.
        labels (bool, optional): Whether to label the largest values and
            the zero crossings. False by default.

    Returns:
        Figure: Figure with one axes per internal force.
    """
    _check_results(structure, load_pattern)

    return _plot_frame_series(structure, load_pattern, frame,
                              'internal_forces', FORCE_FLAGS, 'forces',
                              axes, labels)


def plot_internal_displacements(structure, load_pattern, frame, axes=None,
                                labels=False, components=None):
    """Plot the internal displacements of a frame along its length.

    Only the displacements that the frame models can be plotted (e.g. uy
    and rz for a frame with bending_z). By default, its translations are
    plotted, or its rotations if it models none (a torsional bar). They are
    given in the local axes of the frame. Optionally, the largest positive
    and most negative values are marked with their coordinates, and so are
    the points where they cross zero.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        frame (str): Name of the frame.
        axes (list, optional): Axes to draw on, one per plotted
            displacement, in the order ux, uy, uz, rx, ry, rz. By default, a
            new figure is created.
        labels (bool, optional): Whether to label the largest values and
            the zero crossings. False by default.
        components (list, optional): Displacements to plot, among those
            that the frame models, e.g. ('uy', 'rz'). By default, its
            translations.

    Returns:
        Figure: Figure with one axes per plotted displacement.
    """
    _check_results(structure, load_pattern)
    if components is None:
        f = structure.frames[frame]
        active = [c for c, flag in DISPLACEMENT_FLAGS.items()
                  if getattr(f, flag)]
        translations = [c for c in active if c.startswith('u')]
        components = translations or active

    return _plot_frame_series(structure, load_pattern, frame,
                              'internal_displacements', DISPLACEMENT_FLAGS,
                              'displacements', axes, labels, components)


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
        structure.results[load_pattern].frames[frame.name].internal_forces,
        force))
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
            and sign is 1 or -1. Each polygon ends at the zeros of values,
            whether they fall between two points or on a point; stretches
            where all the values are zero give no polygon.
    """
    polygons = []
    run_base, run_tip, run_values = [base[0]], [tip[0]], [values[0]]

    def close():
        """Add the current run as a polygon, signed by its first value."""
        nonzero = [value for value in run_values if value != 0]
        if nonzero:
            polygons.append((np.array(run_base + run_tip[::-1]),
                             1 if nonzero[0] > 0 else -1))

    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if a * b < 0:
            # the values cross zero between i and i + 1
            t = a / (a - b)
            zero = base[i] + t * (base[i + 1] - base[i])
            run_base.append(zero)
            run_tip.append(zero)
            run_values.append(0)
            close()
            run_base, run_tip, run_values = [zero], [zero], [0]
        run_base.append(base[i + 1])
        run_tip.append(tip[i + 1])
        run_values.append(b)
        if b == 0 and a != 0:
            # the values reach zero on point i + 1
            close()
            run_base, run_tip, run_values = [base[i + 1]], [tip[i + 1]], [0]

    close()

    return polygons


def _draw_frame_dimensions(ax, drawings, size, room):
    """Dimension some points along frames from their ends.

    Each horizontal or vertical frame with points inside it gets a chain of
    dimensions between its ends and those points; inclined frames get none.
    The frames on the same line share the line of their chains: below what
    is drawn on them if horizontal, and on the side away from most of it if
    vertical.

    Args:
        ax (Axes): 2D axes to draw on.
        drawings (list): (frame, base, tip, distances) tuples, where base
            and tip are the points along the frame and the points drawn
            from them, e.g. a diagram, and distances are those of the points
            to dimension, from the joint j of the frame.
        size (float): Size of the structure.
        room (float): Room to leave beyond the drawings, for their labels.

    Returns:
        list: 3D points that the chains cover.
    """
    gap = 0.06 * size
    lines = {}
    for frame, base, tip, distances in drawings:
        direction = frame.direction_cosines_vector()
        if abs(direction[1]) < 1e-9:
            axis = 0
        elif abs(direction[0]) < 1e-9:
            axis = 1
        else:
            continue
        # only the points inside the frame, not at its ends
        length = frame.length()
        inside = [d for d in distances
                  if 1e-6 * length < d < (1 - 1e-6) * length]
        if not inside:
            continue
        along = base[0][axis] + direction[axis] * np.array(
            [0, *inside, length])
        key = (axis, round(base[0][1 - axis], 9))
        lines.setdefault(key, []).append((np.sort(along), base, tip))

    drawn = []
    for (axis, _), chains in lines.items():
        across = np.concatenate([np.vstack((base, tip))
                                 for _, base, tip in chains])[:, 1 - axis]
        # a vertical chain goes on the side away from most of the diagram
        side = 1 if axis == 1 and sum(
            (tip[:, 0] - base[:, 0]).sum() for _, base, tip in chains) < 0 \
            else -1
        level = (across.max() if side > 0 else across.min()) + \
            side * (room + gap)
        for along, _, _ in chains:
            drawn += _dimension_chain(ax, axis, along, level, side * gap)

    return drawn


def plot_diagram(structure, load_pattern, force, ax=None, scale=None,
                 labels=False, dimensions=False):
    """Plot the diagram of an internal force on the whole structure.

    The values are drawn perpendicular to each frame: bending moments on the
    tension side, and forces and torque along the positive local y-axis (z
    for fz). Positive values are blue and negative values red. Frames that do
    not model the force are drawn without diagram. The values are scaled to
    the size of the structure, so the plot has no axes. Optionally, the
    largest positive and negative values of each frame, and those at its
    ends, e.g. the moments over the supports, are labelled, and the points
    where the diagrams of the horizontal and vertical frames cross zero are
    dimensioned from the ends of their frames.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        force (str): Internal force: 'fx', 'fy', 'fz', 'mx', 'my' or 'mz'.
        ax (Axes, optional): Axes to draw on (3D for a structure that is not
            planar). By default, a new figure is created.
        scale (float, optional): Length per unit of force. By default, the
            largest value is drawn as 15% of the size of the structure.
        labels (bool, optional): Whether to label the largest positive and
            negative values of each frame, and those at its ends. False by
            default.
        dimensions (bool, optional): Whether to dimension the zero
            crossings of a plane structure. False by default.

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
    internal = {name: results.internal_forces for name, results in
                structure.results[load_pattern].frames.items()}
    largest = max((np.abs(getattr(internal[f.name], force)).max()
                   for f in frames), default=0)
    if scale is None:
        scale = 0.15 * _size(structure) / largest if largest else 1

    points = []
    polygons, colors = [], []
    # values to label: the largest positive and most negative value of each
    # frame and those at its ends, with their points and the direction in
    # which the diagram grows there
    extremes = []
    diagrams = []
    for frame in structure.frames.values():
        line = frame_positions(frame, 2)
        _line(ax, line, color=GRAY, linewidth=1)
        points.extend(line)
        if frame not in frames:
            continue

        base, tip, values = diagram_positions(structure, load_pattern, frame,
                                              force, scale)
        # the diagram is dark blue where positive and dark red where negative
        positive, negative = _sign_parts(tip, values)
        _line(ax, positive, color=DARK_BLUE, linewidth=1)
        _line(ax, negative, color=DARK_RED, linewidth=1)
        points.extend(tip)
        diagrams.append((frame, base, tip, values))
        for polygon, sign in _same_sign_polygons(base, tip, values):
            polygons.append(polygon)
            colors.append(BLUE if sign > 0 else RED)
        # the extremes, and the values at the ends, e.g. the moments over
        # the supports, unless equal to an extreme, as along a constant
        # diagram
        indices = extreme_indices(values)
        for end in (0, len(values) - 1):
            if all(_format(values[end]) != _format(values[k])
                   for k in indices):
                indices.append(end)
        for k in indices:
            # a value shared by two frames at their joint is labelled once
            if abs(values[k]) > 1e-6 * largest and not any(
                    _format(values[k]) == _format(value) and
                    np.allclose(tip[k], at) for value, at, _ in extremes):
                extremes.append((values[k], tip[k], tip[k] - base[k]))

    points.extend(_draw_supports(ax, structure))

    if ax.name == '3d':
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax.add_collection3d(Poly3DCollection(polygons, facecolors=colors,
                                             alpha=0.35, linewidths=0))
    else:
        from matplotlib.collections import PolyCollection
        ax.add_collection(PolyCollection([p[:, :2] for p in polygons],
                                         facecolors=colors, alpha=0.35,
                                         linewidths=0))

    # label the extremes beyond their tip, and leave room for the label, so
    # it does not reach the title
    for value, at, out in extremes if labels else []:
        out = out / np.linalg.norm(out)
        points.append(at + 0.04 * _size(structure) * out)
        if ax.name == '3d':
            ax.text(*at, _format(value), color=INK, fontsize=8)
        else:
            up = out[1] >= 0
            ax.annotate(_format(value), at[:2], textcoords='offset points',
                        xytext=(4, 4 if up else -4),
                        va='bottom' if up else 'top', color=INK, fontsize=8)
    if dimensions and ax.name != '3d':
        room = 0.04 * _size(structure) if labels else 0
        drawings = [(frame, base, tip, zero_crossings(
            np.linspace(0, frame.length(), len(values)), values))
            for frame, base, tip, values in diagrams]
        points.extend(_draw_frame_dimensions(ax, drawings, _size(structure),
                                             room))

    _set_limits(ax, points)
    ax.set_axis_off()
    ax.set_title(_title(LABELS[force], load_pattern), color=INK, fontsize=11)

    return ax


def _vector(load, names):
    """Return the components of a load or reaction as a 3D vector."""
    return np.array([getattr(load, name) or 0 for name in names], dtype=float)


def load_arrows(structure, load_pattern):
    """Return the forces and moments of a load pattern, as arrows.

    The joint point loads, frame point loads and distributed loads are given
    in global axes, at the point where they act. A distributed load is a row
    of equal arrows along its frame, spaced alike on every frame. The
    distributed moments are not included.

    Args:
        structure (Structure): Structure.
        load_pattern (str): Name of the load pattern.

    Returns:
        list: (kind, point, vector, label) tuples, where kind is 'force',
            'moment' or 'distributed', point and vector are 3D arrays in
            global axes, and label is the text of the arrow. The point of a
            distributed load is an array of shape (n, 3), one row per arrow.
    """
    pattern = structure.load_patterns[load_pattern]
    arrows = []

    def add(point, force, moment, label=True):
        if np.any(force):
            arrows.append(('force', point, force,
                           _format(np.linalg.norm(force)) if label else None))
        if np.any(moment):
            arrows.append(('moment', point, moment,
                           _format(np.linalg.norm(moment))))

    for joint, loads in pattern.joint_point_loads.items():
        point = structure.joints[joint].position()
        for load in loads:
            add(point, _vector(load, ('fx', 'fy', 'fz')),
                _vector(load, ('mx', 'my', 'mz')))

    for name, loads in pattern.frame_point_loads.items():
        frame = structure.frames[name]
        r = frame.rotation_matrix()
        start = structure.joints[frame.joint_j].position()
        for load in loads:
            point = start + load.dist * frame.direction_cosines_vector()
            add(point, r @ _vector(load, ('fx', 'fy', 'fz')),
                r @ _vector(load, ('mx', 'my', 'mz')))

    # the arrows of every frame share the same spacing, about a twelfth of
    # the structure, with at least two intervals per frame
    size = _size(structure)
    for name, loads in pattern.frame_distributed_loads.items():
        frame = structure.frames[name]
        r = frame.rotation_matrix()
        intervals = max(2, round(12 * frame.length() / size))
        points = frame_positions(frame, intervals + 1)
        for load in loads:
            force = r @ _vector(load, ('fx', 'fy', 'fz'))
            if not np.any(force):
                continue
            arrows.append(('distributed', points, force,
                           f'q = {_format(np.linalg.norm(force))}'))

    return arrows


def _reaction_label(kind, axis, joint, value):
    """Return the label of a reaction, e.g. R_A^y = 108.7 or M_A^z = 12.

    The symbol is written in matplotlib mathtext: R for a force and M for a
    moment, with its joint as subscript and its global axis as superscript.
    """
    # the characters that mathtext would read as commands
    name = ''.join('\\' + c if c in '\\$_^{}%#&~' else c for c in joint)
    symbol = 'R' if kind == 'force' else 'M'

    return f'${symbol}_{{\\mathrm{{{name}}}}}^{{{"xyz"[axis]}}}$ = ' \
        f'{_format(abs(value))}'


def reaction_arrows(structure, load_pattern, offsets=None,
                    moment_offsets=None):
    """Return the reactions of a load pattern, as arrows.

    Each nonzero component is an arrow of its own, along its global axis,
    labelled with its symbol and value, e.g. R_A^y = 108.7.

    Args:
        structure (Structure): Analysed structure.
        load_pattern (str): Name of the load pattern.
        offsets (dict, optional): Offset of the point where the forces of
            each joint act, by joint name, e.g. to draw them beyond the
            support symbol: one vector for all of them, or three rows, one
            for the force along each global axis.
        moment_offsets (dict, optional): Offset of the point where the
            moments of each joint act, by joint name.

    Returns:
        list: (kind, point, vector, label) tuples, as in load_arrows(), with
            kind 'force' or 'moment'.
    """
    reactions = structure.results[load_pattern].reactions
    components = [_vector(reaction, names) for reaction in reactions.values()
                  for names in (('fx', 'fy', 'fz'), ('mx', 'my', 'mz'))]
    largest = max((np.abs(c).max() for c in components), default=0)

    offsets = offsets or {}
    moment_offsets = moment_offsets or {}
    arrows = []
    for joint, reaction in reactions.items():
        position = structure.joints[joint].position()
        for kind, names in (('force', ('fx', 'fy', 'fz')),
                            ('moment', ('mx', 'my', 'mz'))):
            offset = np.asarray((offsets if kind == 'force'
                                 else moment_offsets).get(joint, 0.0))
            for axis, value in enumerate(_vector(reaction, names)):
                if abs(value) <= 1e-9 * largest:
                    continue
                vector = np.zeros(3)
                vector[axis] = value
                point = position + (offset[axis] if offset.ndim == 2
                                    else offset)
                arrows.append((kind, point, vector,
                               _reaction_label(kind, axis, joint, value)))

    return arrows


def _draw_arrows(ax, arrows, color, size, radius=None, length=0.12,
                 centre=None, away=()):
    """Draw arrows, and return the points they cover.

    A force points at its point, from a tail at a fixed distance; a moment
    is an arc of the given radius around its point in 2D (counterclockwise
    if positive, open at the bottom, or a half circle open towards the
    centre at the away points) and an arrow along its axis in 3D. The
    length of the forces, a fraction of the size of the structure, does not
    depend on the value, which is given by the label.

    A distributed load is a row of shorter arrows, labeled at the middle
    one, whose tails are joined by a line.
    """
    planar = ax.name != '3d'
    points = []
    single = []
    for kind, point, vector, label in arrows:
        if kind != 'distributed':
            single.append((kind, point, vector, label))
            continue
        middle = len(point) // 2
        single.extend(('distributed', p, vector, label if i == middle else
                       None) for i, p in enumerate(point))
        unit = vector / np.linalg.norm(vector)
        _line(ax, point - 0.06 * size * unit, color=color, linewidth=1.2)

    for kind, point, vector, label in single:
        unit = vector / np.linalg.norm(vector)
        arrow = (0.06 if kind == 'distributed' else length) * size

        if kind == 'moment' and planar:
            # an arc of 270 degrees around the point, open at the bottom;
            # at one of the away points, a half circle on the side away from
            # the centre, open towards it; with its arrowhead
            r = radius or 0.04 * size
            sign = 1 if vector[2] > 0 else -1
            behind = centre is not None and any(
                np.allclose(point, p) for p in away)
            left = behind and point[0] < centre[0]
            if behind:
                facing = np.pi if left else 0.0
                angles = facing + sign * np.pi * np.linspace(-0.5, 0.5, 40)
            else:
                angles = np.pi * (0.5 + sign * np.linspace(-0.75, 0.75, 40))
            arc = point[:2] + r * np.column_stack(
                (np.cos(angles), np.sin(angles)))
            ax.plot(*arc[:-1].T, color=color, linewidth=1.2, zorder=4)
            ax.annotate('', xy=arc[-1], xytext=arc[-3],
                        arrowprops=dict(arrowstyle='-|>', color=color,
                                        linewidth=1.2), zorder=4)
            # the label goes up and right, or on the side away from the
            # centre for the moments that act at one of the away points
            text_at = point[:2] + r * np.array([-0.8 if left else 0.8, 0.8])
            points.extend((point[:2] - r, point[:2] + r))
            if label:
                ax.annotate(label, text_at, ha='right' if left else 'left',
                            color=INK, fontsize=8)
            continue

        # a force points at the point; a moment in 3D starts at it
        tail = point - arrow * unit if kind != 'moment' else point
        head = tail + arrow * unit
        points.extend((tail, head))
        if planar:
            ax.annotate('', xy=head[:2], xytext=tail[:2],
                        arrowprops=dict(arrowstyle='-|>', color=color,
                                        linewidth=1.2, shrinkA=0, shrinkB=0),
                        zorder=4)
            if label:
                ax.annotate(label, tail[:2], textcoords='offset points',
                            xytext=(3, 3), color=INK, fontsize=8)
        else:
            ax.quiver(*tail, *(head - tail), color=color, linewidth=1.2,
                      arrow_length_ratio=0.25)
            if label:
                # a reaction label already names its moment
                prefix = 'M = ' if kind == 'moment' and '=' not in label \
                    else ''
                ax.text(*tail, f' {prefix}{label}', color=INK, fontsize=8)

    return [np.append(p, 0) if len(p) == 2 else p for p in points]


def _draw_support_names(ax, structure):
    """Write the name of the joint of every support beside its symbol.

    In 2D the name goes on the left of the symbol, at half its depth (above
    it, if the symbol lies sideways); in 3D, by its marker.

    Returns:
        list: 3D points that the names are written at.
    """
    h = _support_size(structure)
    points = []
    for support in structure.supports.values():
        kind = support_kind(structure, support)
        if kind is None:
            continue
        p = structure.joints[support.joint].position()
        if ax.name == '3d':
            ax.text(*p, f' {support.joint}', color=MUTED, fontsize=8)
            points.append(p)
            continue

        d = _support_direction(structure, support, kind)
        # the side along the ground line that points left, or up
        side = np.array([-d[1], d[0]])
        if side[0] > 0 or (side[0] == 0 and side[1] < 0):
            side = -side
        # beyond the arc of the moment reaction of a fixed support
        away = 2.0 if kind == 'fixed' else 1.1
        at = p[:2] + 0.5 * _support_depth(kind) * h * d + away * h * side
        ax.annotate(support.joint, at, ha='right' if side[0] < 0 else
                    'center', va='center' if side[0] < 0 else 'bottom',
                    color=MUTED, fontsize=8)
        points.append(np.append(at, 0))

    return points


def _draw_frame_names(ax, structure):
    """Write the name of every frame, in a circle, beside its middle.

    The circle tells the names of the frames from those of the joints, as
    in Weaver and Gere. In 2D the name goes below the frame, or on its left
    if the frame is vertical, so it stays clear of the loads, which act
    from above; in 3D, by its middle.

    Returns:
        list: 3D points that the names are written at.
    """
    circle = dict(boxstyle='circle,pad=0.25', facecolor='white',
                  edgecolor=MUTED, linewidth=0.6)
    points = []
    for frame in structure.frames.values():
        start, end = frame_positions(frame, 2)
        middle = (start + end) / 2
        if ax.name == '3d':
            ax.text(*middle, frame.name, color=MUTED, fontsize=8,
                    ha='center', va='center', bbox=circle)
            points.append(middle)
            continue

        d = (end - start)[:2]
        d = d / np.linalg.norm(d)
        # the side across the frame that points down, or left
        side = np.array([-d[1], d[0]])
        if side[1] > 0 or (side[1] == 0 and side[0] > 0):
            side = -side
        # far enough that the circle does not touch the frame
        ax.annotate(frame.name, middle[:2], textcoords='offset points',
                    xytext=12 * side, ha='center', va='center', color=MUTED,
                    fontsize=8, bbox=circle)
        points.append(middle)

    return points


def _dimension_chain(ax, axis, values, level, step):
    """Draw a chain of dimensions between some coordinates.

    The dimension line lies across the given axis at level, with an
    extension line and a slash at each coordinate, and the length of each
    dimension, with two decimals at most, over it.

    Args:
        ax (Axes): 2D axes to draw on.
        axis (int): 0 for a horizontal chain (x coordinates), 1 for a
            vertical one (y coordinates).
        values (ndarray): Sorted coordinates along the axis.
        level (float): Coordinate of the dimension line across the axis.
        step (float): Length of the extension lines, signed away from what
            is dimensioned.

    Returns:
        list: 3D points that the chain covers.
    """
    from matplotlib.collections import LineCollection

    tick = 0.15 * abs(step)
    lines = [[(values[0], level), (values[-1], level)]]
    for value in values:
        # an extension line and a slash at each coordinate
        lines.append([(value, level - 0.4 * step),
                      (value, level + 0.4 * step)])
        lines.append([(value - tick, level - tick),
                      (value + tick, level + tick)])
    for start, stop in zip(values[:-1], values[1:]):
        # lengths with two decimals at most, e.g. 7 or 2.56
        text = f'{stop - start:.2f}'.rstrip('0').rstrip('.')
        middle = (start + stop) / 2
        if axis == 0:
            ax.annotate(text, (middle, level), textcoords='offset points',
                        xytext=(0, 2), ha='center', va='bottom', color=MUTED,
                        fontsize=8)
        else:
            # on the side of the chain away from what is dimensioned
            left = step < 0
            ax.annotate(text, (level, middle), textcoords='offset points',
                        xytext=(-2 if left else 2, 0),
                        ha='right' if left else 'left', va='center',
                        rotation=90, color=MUTED, fontsize=8)
    if axis == 1:
        lines = [[(y, x) for x, y in line] for line in lines]
    ax.add_collection(LineCollection(lines, colors=MUTED, linewidths=0.8))

    # the chain ends with its extension lines
    far = level + 0.4 * step
    ends = [(values[0], far), (values[-1], far)]
    if axis == 1:
        ends = [(y, x) for x, y in ends]

    return [np.array([x, y, 0]) for x, y in ends]


def _draw_dimensions(ax, structure, points):
    """Draw the dimensions of a plane structure, and return their points.

    A chain of horizontal dimensions between the x coordinates of the joints
    goes below the given points, and a chain of vertical dimensions between
    their y coordinates goes on their left.
    """
    positions = np.array([joint.position()
                          for joint in structure.joints.values()])
    points = np.vstack([positions, *np.reshape(points, (-1, 3))])
    gap = 0.06 * _size(structure)

    drawn = []
    for axis in (0, 1):
        values = np.unique(np.round(positions[:, axis], 9))
        if len(values) > 1:
            level = points[:, 1 - axis].min() - gap
            drawn += _dimension_chain(ax, axis, values, level, -gap)

    return drawn


def plot_loads(structure, load_pattern, ax=None, reactions=True,
               dimensions=False, support_names=False, frame_names=False):
    """Plot the loads of a load pattern on a structure, and its reactions.

    The loads are drawn in black and the reactions in red. Forces are arrows
    that point at the joint or the point where they act, and moments arcs
    around it (arrows along their axis in 3D); the arrows have a fixed
    length and are labelled with their values. Optionally, a plane
    structure is dimensioned: the distances between its joints along x
    below it, and along y on its left; the supports are named after
    their joints; and the frames are named, in circles, beside their middles.

    Args:
        structure (Structure): Structure.
        load_pattern (str): Name of the load pattern.
        ax (Axes, optional): Axes to draw on (3D for a structure that is not
            planar). By default, a new figure is created.
        reactions (bool, optional): Whether to draw the reactions, if the
            structure has been analysed.
        dimensions (bool, optional): Whether to dimension a plane
            structure. False by default.
        support_names (bool, optional): Whether to write the name of the
            joint of each support beside it. False by default.
        frame_names (bool, optional): Whether to write the name of each
            frame, in a circle, beside its middle. False by default.

    Returns:
        Axes: Axes of the plot.
    """
    if load_pattern not in structure.load_patterns:
        raise ValueError(f'there is no load pattern {load_pattern!r}')

    ax = _axes(structure, ax)
    size = _size(structure)

    points = []
    for frame in structure.frames.values():
        line = frame_positions(frame, 2)
        _line(ax, line, color=MUTED, linewidth=1)
        points.extend(line)
    points.extend(_draw_supports(ax, structure))

    points.extend(_draw_arrows(ax, load_arrows(structure, load_pattern), INK,
                               size))
    if reactions and load_pattern in structure.results:
        offsets, moment_offsets, radius, centre = {}, {}, None, None
        if ax.name != '3d':
            # the forces act just beyond the support symbols (beyond the
            # hatch of a wall), and the moments around them; against a
            # wall, a force along it acts at the end of its line, below it
            # if it points up (above it if down), and the moments go round
            # a point just behind the hatch, as a half circle open towards
            # the wall
            h = _support_size(structure)
            radius = 1.5 * h
            reactions_of = structure.results[load_pattern].reactions
            for support in structure.supports.values():
                symbol = support_kind(structure, support)
                if symbol is None:
                    continue
                d = _support_direction(structure, support, symbol)
                depth = _support_depth(symbol)
                gap = 0.4 if symbol == 'fixed' else 0.2
                beyond = np.append((depth + gap) * h * d, 0)
                offsets[support.joint] = beyond
                if symbol == 'fixed' and abs(d[0]) > abs(d[1]):
                    fy = reactions_of[support.joint].fy or 0
                    along = np.array([0, -np.sign(fy or 1) * h, 0])
                    offsets[support.joint] = np.array([beyond, along,
                                                       beyond])
                    moment_offsets[support.joint] = np.append(
                        (depth + 0.4) * h * d, 0)
            centre = np.mean([joint.position() for joint in
                              structure.joints.values()], axis=0)
        arrows = reaction_arrows(structure, load_pattern, offsets,
                                 moment_offsets)
        # the reactions are shorter than the loads, so they do not reach
        # far below the structure
        shifted = [structure.joints[joint].position() + offset
                   for joint, offset in moment_offsets.items()]
        points.extend(_draw_arrows(ax, arrows, RED, size, radius,
                                   length=0.07, centre=centre,
                                   away=shifted))
    if support_names:
        points.extend(_draw_support_names(ax, structure))
    if frame_names:
        points.extend(_draw_frame_names(ax, structure))
    if dimensions and ax.name != '3d':
        points.extend(_draw_dimensions(ax, structure, points))

    _set_limits(ax, points)
    ax.set_axis_off()
    title = 'Loads and reactions' if reactions and \
        load_pattern in structure.results else 'Loads'
    ax.set_title(_title(title, load_pattern), color=INK, fontsize=11)

    return ax
