"""
Tests for the plots of the analysis results.

The tests check the data drawn, not the look of the figures.

Methods of the results of a load pattern (Results):
    - plot_loads(ax=None, reactions=True)
    - plot_deformed(ax=None, scale=None, labels=False)
    - plot_diagram(force, ax=None, scale=None, labels=False)

Methods of the results of a frame (FrameResults):
    - plot_internal_forces(axes=None, labels=False)
    - plot_internal_displacements(axes=None, labels=False, components=None)

Typical usage:
    structure.run_analysis()
    ax = structure.results['dead'].plot_deformed()
    ax.figure.savefig('deformed.png')
"""

import matplotlib
import numpy as np
import pytest

from pymas import Structure, plotting

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Close the figures created by each test."""
    yield
    plt.close('all')


@pytest.fixture
def portal():
    """Create and analyse a fixed portal frame (plane frame).

    Structure:
        b ------ c
        |        |
        a        d

    Columns of 4 m and a 6 m beam with a uniform load and a lateral load.
    """
    s = Structure('plane_frame')
    s.add_material('steel', E=200e6)
    s.add_rectangular_section('section', dy=0.4, dz=0.3)
    s.add_joint('a', 0, 0)
    s.add_joint('b', 0, 4)
    s.add_joint('c', 6, 4)
    s.add_joint('d', 6, 0)
    for name, j, k in (('ab', 'a', 'b'), ('bc', 'b', 'c'), ('cd', 'c', 'd')):
        s.add_frame(name, j, k, 'steel', 'section', axial=True,
                    bending_z=True)
    s.add_support('a', r_ux=True, r_uy=True, r_rz=True)
    s.add_support('d', r_ux=True, r_uy=True, r_rz=True)
    s.add_load_pattern('dead')
    s.load_patterns['dead'].add_distributed_load('bc', fy=-10)
    s.load_patterns['dead'].add_joint_point_load('b', fx=20)
    s.run_analysis()

    return s


@pytest.fixture
def space_cantilever():
    """Create and analyse an oblique space frame cantilever with a tip load."""
    s = Structure('space_frame')
    s.add_material('steel', E=200e6, G=80e6)
    s.add_section('section', A=0.01, J=1e-5, Iy=2e-5, Iz=8e-5)
    s.add_joint('a', 0, 0, 0)
    s.add_joint('b', 3, 4, 5)
    s.add_frame('ab', 'a', 'b', 'steel', 'section', axial=True,
                torsional=True, bending_y=True, bending_z=True)
    s.add_support('a', r_ux=True, r_uy=True, r_uz=True, r_rx=True,
                  r_ry=True, r_rz=True)
    s.add_load_pattern('tip')
    s.load_patterns['tip'].add_joint_point_load('b', fx=1, fy=-2, fz=3)
    s.run_analysis()

    return s


@pytest.fixture
def plane_truss():
    """Create and analyse a triangular plane truss (axial-only frames)."""
    s = Structure('plane_truss')
    s.add_material('steel', E=200e6)
    s.add_section('section', A=0.01)
    s.add_joint('a', 0, 0)
    s.add_joint('b', 4, 0)
    s.add_joint('c', 2, 3)
    for name, j, k in (('ab', 'a', 'b'), ('bc', 'b', 'c'), ('ca', 'c', 'a')):
        s.add_frame(name, j, k, 'steel', 'section', axial=True)
    s.add_support('a', r_ux=True, r_uy=True)
    s.add_support('b', r_uy=True)
    s.add_load_pattern('load')
    s.load_patterns['load'].add_joint_point_load('c', fx=5, fy=-10)
    s.run_analysis()

    return s


def _ends(structure, load_pattern, name, scale):
    """Return the deformed positions expected at both ends of a frame."""
    frame = structure.frames[name]
    ends = []
    for joint in (frame.joint_j, frame.joint_k):
        d = structure.results[load_pattern].displacements[joint]\
            .displacement_vector()
        ends.append(structure.joints[joint].position() + scale * d[:3])

    return ends


class TestPlanar:
    """Tests for the choice between a 2D and a 3D view."""

    @pytest.mark.parametrize('type, planar', [
        ('bar', True), ('beam', True), ('plane_truss', True),
        ('plane_frame', True), ('grid', False), ('space_truss', False),
        ('space_frame', False),
    ])
    def test_structure_types(self, type, planar):
        """Verify the view of a structure in the XY plane by its type."""
        s = Structure(type)
        s.add_joint('a', 0, 0)
        s.add_joint('b', 1, 1)
        assert plotting.is_planar(s) is planar

    def test_joint_out_of_plane(self):
        """Verify a plane frame with a joint out of the XY plane is 3D."""
        s = Structure('plane_frame')
        s.add_joint('a', 0, 0, 0)
        s.add_joint('b', 1, 0, 1)
        assert not plotting.is_planar(s)


class TestDeformedPositions:
    """Tests for the deformed points of a frame."""

    @pytest.mark.parametrize('model, load_pattern', [
        ('portal', 'dead'), ('space_cantilever', 'tip'),
        ('plane_truss', 'load'),
    ])
    def test_ends_follow_the_joints(self, model, load_pattern, request):
        """Verify the deformed ends of every frame are the moved joints.

        For axial-only frames the transverse displacement is interpolated
        between the joints.
        """
        structure = request.getfixturevalue(model)
        for name, frame in structure.frames.items():
            points = plotting.deformed_positions(structure, load_pattern,
                                                 frame, 100)
            j, k = _ends(structure, load_pattern, name, 100)
            np.testing.assert_allclose(points[0], j, atol=1e-9)
            np.testing.assert_allclose(points[-1], k, atol=1e-9)

    def test_truss_frames_stay_straight(self, plane_truss):
        """Verify an axial-only frame is drawn straight (a rigid motion)."""
        frame = plane_truss.frames['bc']
        points = plotting.deformed_positions(plane_truss, 'load', frame, 100)
        j, k = points[0], points[-1]
        distance = np.linalg.norm(np.cross(points - j, k - j), axis=1)
        assert distance.max() == pytest.approx(0, abs=1e-9)


class TestPlotDeformed:
    """Tests for Structure.plot_deformed()."""

    def test_planar_structure_is_2d(self, portal):
        """Verify a plane frame is drawn in a 2D axes."""
        ax = portal.results['dead'].plot_deformed()
        assert ax.name != '3d'

    def test_space_structure_is_3d(self, space_cantilever):
        """Verify a space frame is drawn in a 3D axes."""
        ax = space_cantilever.results['tip'].plot_deformed()
        assert ax.name == '3d'

    def test_lines(self, portal):
        """Verify an undeformed and a deformed line per frame.

        The deformed line of each frame ends at its moved far joint.
        """
        ax = portal.results['dead'].plot_deformed(scale=50)
        lines = ax.get_lines()
        assert len(lines) == 2 * len(portal.frames)

        for name, deformed in zip(portal.frames, lines[1::2]):
            _, k = _ends(portal, 'dead', name, 50)
            np.testing.assert_allclose(
                [deformed.get_xdata()[-1], deformed.get_ydata()[-1]], k[:2],
                atol=1e-9)

    def test_auto_scale(self, portal):
        """Verify the largest displacement is 10% of the structure size.

        The displacement of a point is the magnitude of its translation.
        """
        scale = plotting._auto_scale(portal, 'dead')
        largest = max(
            np.linalg.norm(
                plotting.deformed_positions(portal, 'dead', f, scale) -
                plotting.frame_positions(f, 101), axis=1).max()
            for f in portal.frames.values())
        assert largest == pytest.approx(0.1 * np.hypot(6, 4))

    def test_largest_displacement_of_truss_is_at_a_joint(self, plane_truss):
        """Verify the largest displacement of a truss is that of a joint.

        Truss frames move rigidly between their joints, so no internal point
        moves more than the joint that moves the most.
        """
        largest, frame, i = plotting.largest_displacement(plane_truss, 'load')
        joints = max(
            np.linalg.norm(d.displacement_vector()[:3])
            for d in plane_truss.results['load'].displacements.values())
        assert largest == pytest.approx(joints)
        assert i in (0, 100)

    def test_largest_displacement_of_beam(self):
        """Verify the largest displacement of a beam is its deflection."""
        s = Structure('beam')
        s.add_material('m', E=200e6)
        s.add_section('s', Iz=1e-4)
        s.add_joint('a', x=0)
        s.add_joint('b', x=4)
        s.add_frame('ab', 'a', 'b', 'm', 's', bending_z=True)
        s.add_support('a', r_uy=True)
        s.add_support('b', r_uy=True)
        s.add_load_pattern('dead')
        s.load_patterns['dead'].add_distributed_load('ab', fy=-10)
        s.run_analysis()

        largest, _, i = plotting.largest_displacement(s, 'dead')
        uy = s.results['dead'].frames['ab'].internal_displacements.uy
        assert largest == pytest.approx(np.abs(uy).max())
        assert i == 50

    def test_no_axes_nor_labels(self, portal):
        """Verify the axes are hidden and, by default, nothing labelled."""
        ax = portal.results['dead'].plot_deformed(scale=50)
        assert not ax.axison
        assert len(ax.texts) == 0
        assert 'displacements ×50' in ax.get_title()

    def test_labels_per_frame(self, portal):
        """Verify the largest deflection of each frame is labelled.

        The label shows the real local uy, with its sign, not the magnified
        one.
        """
        frames = portal.results['dead'].frames
        expected = []
        for name in portal.frames:
            uy = np.array(frames[name].internal_displacements.uy)
            largest = uy[np.abs(uy).argmax()]
            expected.append(plotting._format(largest))
        ax = portal.results['dead'].plot_deformed(scale=50, labels=True)
        assert sorted(t.get_text() for t in ax.texts) == sorted(expected)

    def test_frames_without_bending_not_labelled(self, plane_truss):
        """Verify the bars of a truss, which do not bend, get no label."""
        ax = plane_truss.results['load'].plot_deformed(labels=True)
        assert len(ax.texts) == 0

    def test_given_axes(self, portal):
        """Verify the plot is drawn on the given axes."""
        fig, ax = plt.subplots()
        assert portal.results['dead'].plot_deformed(ax=ax) is ax

    def test_axes_first(self, portal):
        """Verify the axes are the first optional argument of the plots of
        the whole structure.
        """
        results = portal.results['dead']
        for plot in (results.plot_deformed,
                     lambda ax: results.plot_diagram('mz', ax)):
            _, ax = plt.subplots()
            assert plot(ax) is ax

    def test_without_analysis_no_results(self):
        """Verify there are no results to plot before run_analysis()."""
        s = Structure('beam')
        s.add_load_pattern('dead')
        assert 'dead' not in s.results

    def test_unknown_load_pattern_raises(self, portal):
        """Verify plotting an unknown load pattern raises ValueError."""
        with pytest.raises(ValueError, match='live'):
            plotting.plot_deformed(portal, 'live')


class TestSupports:
    """Tests for the support symbols of the plots."""

    @pytest.mark.parametrize('type, restraints, kind', [
        ('plane_frame', {'r_ux': True, 'r_uy': True, 'r_rz': True}, 'fixed'),
        ('plane_frame', {'r_ux': True, 'r_uy': True}, 'pinned'),
        ('plane_frame', {'r_uy': True}, 'roller'),
        ('plane_truss', {'r_ux': True, 'r_uy': True}, 'pinned'),
        ('plane_truss', {'r_uy': True}, 'roller'),
        ('beam', {'r_uy': True}, 'roller'),
        ('beam', {'r_ux': True, 'r_uy': True}, 'pinned'),
        ('beam', {'r_uy': True, 'r_rz': True}, 'fixed'),
        ('beam', {'r_ux': True}, None),
        ('space_truss', {'r_ux': True, 'r_uy': True, 'r_uz': True}, 'pinned'),
        ('grid', {'r_uz': True, 'r_rx': True, 'r_ry': True}, 'fixed'),
    ])
    def test_support_kind(self, type, restraints, kind):
        """Verify the kind of a support from its restraints.

        In a beam, which does not model ux, a written r_ux tells a pinned
        support from a roller; r_ux alone restrains no active DoF, so the
        support is not drawn.
        """
        s = Structure(type)
        s.add_joint('a', 0, 0)
        support = s.add_support('a', **restraints)
        assert plotting.support_kind(s, support) == kind

    def test_symbols_in_2d(self, plane_truss):
        """Verify the symbols of a pinned support and a roller.

        The pinned support is a triangle and the roller a circle, both on a
        line with four hatches under it.
        """
        ax = plane_truss.results['load'].plot_deformed()
        collections = {type(c).__name__: c for c in ax.collections}
        assert len(collections['PolyCollection'].get_paths()) == 1
        assert len(collections['PatchCollection'].get_paths()) == 1
        assert len(collections['LineCollection'].get_segments()) == \
            2 * (1 + 4)

    def test_symbols_in_3d(self, space_cantilever):
        """Verify a fixed support is drawn as a square marker in 3D."""
        ax = space_cantilever.results['tip'].plot_diagram('my')
        markers = [c for c in ax.collections
                   if type(c).__name__ == 'Path3DCollection']
        assert len(markers) == 1


def _curve(ax, n):
    """Return the curve of an element plot at its n equally spaced points.

    The curve is drawn as a positive and a negative line, with the zero
    crossings added, after the zero line.
    """
    positive, negative = ax.get_lines()[1:3]
    x, y = (np.where(np.isnan(p), n, p) for p, n in (
        (positive.get_xdata(), negative.get_xdata()),
        (positive.get_ydata(), negative.get_ydata())))

    return np.interp(np.linspace(x[0], x[-1], n), x, y)


def _symbol(title):
    """Return the component of the title of an element plot, e.g. 'uy'."""
    return title.split(' of frame')[0].split()[-1]


class TestPlotFrameSeries:
    """Tests for plot_internal_forces() and plot_internal_displacements()."""

    def test_internal_forces_of_active_flags(self, portal):
        """Verify one axes per modelled force, each with its own title,
        plotting its values.
        """
        fig = portal.results['dead'].frames['bc'].plot_internal_forces()
        assert [ax.get_title() for ax in fig.axes] == [
            "Axial force fx of frame 'bc': dead",
            "Shear force fy of frame 'bc': dead",
            "Bending moment mz of frame 'bc': dead"]

        mz = portal.results['dead'].frames['bc'].internal_forces.mz
        np.testing.assert_allclose(_curve(fig.axes[2], len(mz)), mz)

    def test_internal_displacements_translations(self, space_cantilever):
        """Verify only the translations are plotted by default."""
        results = space_cantilever.results['tip'].frames['ab']
        fig = results.plot_internal_displacements()
        titles = [ax.get_title() for ax in fig.axes]
        assert [_symbol(title) for title in titles] == ['ux', 'uy', 'uz']

        uz = results.internal_displacements.uz
        np.testing.assert_allclose(_curve(fig.axes[2], len(uz)), uz)

    def test_internal_displacements_components(self, space_cantilever):
        """Verify the given components are plotted, in their usual order."""
        results = space_cantilever.results['tip'].frames['ab']
        fig = results.plot_internal_displacements(components=('rz', 'uy'))
        titles = [ax.get_title() for ax in fig.axes]
        assert [_symbol(title) for title in titles] == ['uy', 'rz']

    def test_rotations_of_a_torsional_bar(self):
        """Verify a frame without translations plots its rotations."""
        s = Structure('torsional_bar')
        s.add_joint('a', x=0)
        s.add_joint('b', x=2)
        s.add_material('m', E=1, G=1)
        s.add_section('s', J=1)
        s.add_frame('ab', 'a', 'b', 'm', 's', torsional=True)
        s.add_support('a', r_rx=True)
        s.add_joint_point_load('b', mx=1)
        s.run_analysis()

        fig = s.plot_internal_displacements('ab')
        assert [_symbol(ax.get_title()) for ax in fig.axes] == ['rx']

    def test_inactive_component_raises(self, portal):
        """Verify asking for a displacement the frame does not model
        raises ValueError.
        """
        with pytest.raises(ValueError, match='uz'):
            portal.results['dead'].frames['bc'].plot_internal_displacements(
                components=('uz',))

    def test_length_axis(self, portal):
        """Verify the x data spans the length of the frame."""
        fig = portal.results['dead'].frames['ab'].plot_internal_forces()
        x = fig.axes[0].get_lines()[-1].get_xdata()
        assert x[0] == 0
        assert x[-1] == pytest.approx(4)

    def test_frame_without_flags_raises(self):
        """Verify a frame that models nothing raises ValueError."""
        s = Structure('beam')
        s.add_material('steel', E=200e6)
        s.add_section('section', Iz=1e-4)
        s.add_joint('a', x=0)
        s.add_joint('b', x=1)
        s.add_joint('c', x=2)
        s.add_frame('ab', 'a', 'b', 'steel', 'section', bending_z=True)
        s.add_frame('bc', 'b', 'c', 'steel', 'section')
        s.add_support('a', r_uy=True, r_rz=True)
        s.add_support('c', r_uy=True, r_rz=True)
        s.add_load_pattern('dead')
        s.load_patterns['dead'].add_joint_point_load('b', fy=-1)
        s.run_analysis()
        with pytest.raises(ValueError, match="'bc'"):
            s.results['dead'].frames['bc'].plot_internal_forces()


class TestSignParts:
    """Tests for _sign_parts()."""

    def test_parts_join_at_the_crossing(self):
        """Verify both parts hold the crossing, and only their own sign."""
        points = np.array([[0, 1], [1, 1], [2, -3]], dtype=float)
        positive, negative = plotting._sign_parts(points, [1, 1, -3])
        np.testing.assert_allclose(positive[:3], [[0, 1], [1, 1], [1.25, 0]])
        assert np.isnan(positive[3]).all()
        assert np.isnan(negative[:2]).all()
        np.testing.assert_allclose(negative[2:], [[1.25, 0], [2, -3]])


class TestExtremeIndices:
    """Tests for extreme_indices()."""

    def test_largest_and_most_negative(self):
        """Verify the largest positive and most negative values."""
        assert plotting.extreme_indices([1, 3, -2, 0]) == [1, 2]

    def test_value_reached_at_separate_places(self):
        """Verify a value reached at both ends of a fixed beam gives both."""
        assert plotting.extreme_indices([-100, 0, 50, 0, -100]) == [0, 2, 4]

    def test_run_of_equal_values_gives_its_first(self):
        """Verify consecutive equal extremes give one index."""
        assert plotting.extreme_indices([0, 5, 5, 0]) == [1]

    def test_no_positive_values(self):
        """Verify the largest value is left out if not positive."""
        assert plotting.extreme_indices([-1, -3, -2]) == [1]


class TestZeroCrossings:
    """Tests for zero_crossings()."""

    def test_between_samples(self):
        """Verify a crossing between two samples is interpolated."""
        assert plotting.zero_crossings([0, 1, 2], [1, 1, -3]) == \
            [pytest.approx(1.25)]

    def test_on_a_sample(self):
        """Verify a crossing that falls on a sample is found once."""
        assert plotting.zero_crossings([0, 1, 2], [1, 0, -1]) == [1]

    def test_ends_and_touches_are_not_crossings(self):
        """Verify zeros at the ends, and touching zero, are not crossings."""
        x = np.linspace(0, 4, 5)
        assert plotting.zero_crossings(x, [0, 1, 0, 1, 0]) == []


class TestFrameSeriesLabels:
    """Tests for the labels of the element plots."""

    @pytest.fixture
    def beam(self):
        """Create and analyse a 10 m simple beam under 12 kN/m."""
        s = Structure('beam')
        s.add_joint('a', x=0)
        s.add_joint('b', x=10)
        s.add_material('m', E=1)
        s.add_section('s', Iz=1)
        s.add_frame('ab', 'a', 'b', 'm', 's', bending_z=True)
        s.add_support('a', r_uy=True)
        s.add_support('b', r_uy=True)
        s.add_distributed_load('ab', fy=-12)
        s.run_analysis()

        return s

    def test_no_labels_by_default(self, beam):
        """Verify the element plots have no labels by default."""
        fig = beam.plot_internal_forces('ab')
        assert all(len(ax.texts) == 0 for ax in fig.axes)

    def test_labels_and_zero_crossing(self, beam):
        """Verify the largest values and the zero crossings are labelled
        with their coordinates.

        The shear goes from 60 to -60 and is zero at midspan; the moment is
        150 at midspan and zero only at the ends.
        """
        shear, moment = beam.plot_internal_forces('ab', labels=True).axes
        assert sorted(t.get_text() for t in shear.texts) == \
            ['(0, 60)', '(10, -60)', '(5, 0)']
        assert [t.get_text() for t in moment.texts] == ['(5, 150)']

    def test_moments_positive_downwards(self, beam):
        """Verify the moment axis is inverted, and the shear axis is not."""
        shear, moment = beam.plot_internal_forces('ab').axes
        assert moment.yaxis_inverted()
        assert not shear.yaxis_inverted()


class TestPlotDiagram:
    """Tests for Structure.plot_diagram()."""

    def test_moment_on_tension_side(self, portal):
        """Verify the sagging moment of the beam is drawn below it.

        The beam bc runs along +x, so its local y-axis is +Y; a positive
        (sagging) mz has its tension side at -Y.
        """
        frame = portal.frames['bc']
        base, tip, values = plotting.diagram_positions(portal, 'dead', frame,
                                                       'mz', 0.01)
        i = int(np.argmax(values))
        assert values[i] > 0
        assert tip[i][1] == pytest.approx(base[i][1] - 0.01 * values[i])

    def test_force_along_positive_axis(self, portal):
        """Verify a shear force is drawn along the positive local y-axis."""
        frame = portal.frames['bc']
        base, tip, values = plotting.diagram_positions(portal, 'dead', frame,
                                                       'fy', 0.01)
        np.testing.assert_allclose(tip[:, 1] - base[:, 1], 0.01 * values)

    def test_scale(self, portal):
        """Verify the largest value is drawn as 15% of the structure size."""
        ax = portal.results['dead'].plot_diagram('mz')
        frames = portal.results['dead'].frames
        largest = max(np.abs(frames[f].internal_forces.mz).max()
                      for f in portal.frames)
        # each frame line is followed by the positive and negative parts of
        # its diagram, which also hold the zero crossings
        size = np.hypot(6, 4)
        lines = ax.get_lines()
        for k, name in enumerate(portal.frames):
            frame = portal.frames[name]
            _, tip, _ = plotting.diagram_positions(
                portal, 'dead', frame, 'mz', 0.15 * size / largest)
            positive, negative = lines[3 * k + 1:3 * k + 3]
            x = np.where(np.isnan(positive.get_xdata()),
                         negative.get_xdata(), positive.get_xdata())
            assert set(np.round(tip[:, 0], 9)) <= set(np.round(x, 9))

    def test_no_axes(self, portal):
        """Verify the diagram is drawn without axes."""
        assert not portal.results['dead'].plot_diagram('mz').axison

    def test_labels_per_frame(self, portal):
        """Verify the largest positive and most negative value of each
        frame, and those at its ends, are labelled.

        At a corner each frame labels its own end, as its diagram is drawn
        apart from the other; the column ab, with negative moments only,
        gets its two ends.
        """
        frames = portal.results['dead'].frames
        expected = set()
        for name in portal.frames:
            mz = np.array(frames[name].internal_forces.mz)
            values = [mz[0], mz[-1]]
            values += [mz.max()] if mz.max() > 0 else []
            values += [mz.min()] if mz.min() < 0 else []
            expected |= {plotting._format(v) for v in values if v != 0}
        ax = portal.results['dead'].plot_diagram('mz', labels=True)
        assert {t.get_text() for t in ax.texts} == expected

    def test_moments_over_supports_labelled(self):
        """Verify the moment over a support is labelled even if it is not
        the largest of its frames: over the middle support of four equal
        spans under a uniform load, -wL^2/14, smaller than -3wL^2/28 over
        the next supports.
        """
        s = Structure('beam')
        for name, x in zip('abcde', range(5)):
            s.add_joint(name, x=x)
            s.add_support(name, r_uy=True)
        s.add_material('m', E=1)
        s.add_section('s', Iz=1)
        for j, k in ('ab', 'bc', 'cd', 'de'):
            s.add_frame(j + k, j, k, 'm', 's', bending_z=True)
            s.add_distributed_load(j + k, fy=-1)
        s.run_analysis()
        texts = [t.get_text() for t in s.plot_diagram('mz', labels=True).texts]
        assert plotting._format(-1 / 14) in texts
        assert texts.count(plotting._format(-3 / 28)) == 2

    def test_constant_diagram_labelled_once(self, plane_truss):
        """Verify a constant diagram, as the axial force of a bar, gets one
        label per frame, not one more at each end.
        """
        ax = plane_truss.results['load'].plot_diagram('fx', labels=True)
        assert len(ax.texts) <= len(plane_truss.frames)

    def test_shared_label_once(self):
        """Verify a value shared by two frames at their joint is labelled
        once.

        The moment over the middle support of a two-span beam is the most
        negative value of both spans.
        """
        s = Structure('beam')
        for name, x in (('a', 0), ('b', 4), ('c', 8)):
            s.add_joint(name, x=x)
        s.add_material('m', E=1)
        s.add_section('s', Iz=1)
        for name, j, k in (('ab', 'a', 'b'), ('bc', 'b', 'c')):
            s.add_frame(name, j, k, 'm', 's', bending_z=True)
            s.add_distributed_load(name, fy=-10)
        for joint in 'abc':
            s.add_support(joint, r_uy=True)
        s.run_analysis()

        texts = [t.get_text() for t in s.plot_diagram('mz', labels=True).texts]
        assert texts.count('-20') == 1
        assert texts.count('11.25') == 2

    def test_without_labels(self, portal):
        """Verify the diagram has no labels by default."""
        ax = portal.results['dead'].plot_diagram('mz')
        assert len(ax.texts) == 0

    @pytest.fixture
    def continuous_beam(self):
        """Create and analyse a beam of spans of 7 and 8 m and a 2 m
        overhang, under 42.5 kN/m (problem 4.1 of Segura).
        """
        s = Structure('beam')
        for name, x in (('1', 0), ('2', 7), ('3', 15), ('4', 17)):
            s.add_joint(name, x=x)
        s.add_material('m', E=1)
        s.add_section('s', Iz=1)
        for j, k in ('12', '23', '34'):
            s.add_frame(j + k, j, k, 'm', 's', bending_z=True)
            s.add_distributed_load(j + k, fy=-42.5)
        s.add_support('1', r_uy=True)
        s.add_support('2', r_uy=True)
        s.add_support('3', r_uy=True)
        s.run_analysis()

        return s

    def test_no_crossing_dimensions_by_default(self, continuous_beam):
        """Verify the zero crossings are not dimensioned by default."""
        assert len(continuous_beam.plot_diagram('fy').texts) == 0

    def test_crossing_dimensions(self, continuous_beam):
        """Verify the zero crossings of each span are dimensioned from its
        ends.

        The shear crosses zero once in each span; the moment once in the
        first span and twice in the second. The overhang only reaches zero
        at its tip, which is not a crossing.
        """
        shear = continuous_beam.plot_diagram('fy', dimensions=True)
        assert [t.get_text() for t in shear.texts] == \
            ['2.56', '4.44', '4.57', '3.43']
        moment = continuous_beam.plot_diagram('mz', dimensions=True)
        assert [t.get_text() for t in moment.texts] == \
            ['5.12', '1.88', '1.79', '5.56', '0.64']

    def test_deflection_dimensions(self, continuous_beam):
        """Verify the position of the largest deflection of each span is
        dimensioned from its ends, only if asked.

        The largest deflection of the overhang is at its tip, which is not
        dimensioned.
        """
        assert len(continuous_beam.plot_deformed().texts) == 0
        ax = continuous_beam.plot_deformed(dimensions=True)
        texts = [float(t.get_text()) for t in ax.texts]
        assert len(texts) == 4
        assert sum(texts[:2]) == pytest.approx(7)
        assert sum(texts[2:]) == pytest.approx(8)

    def test_crossing_dimensions_of_a_column(self, portal):
        """Verify a vertical frame gets a vertical chain of dimensions.

        The moment of the column cd crosses zero; its dimensions add up to
        its height.
        """
        ax = portal.results['dead'].plot_diagram('mz', dimensions=True)
        vertical = [float(t.get_text()) for t in ax.texts
                    if t.get_rotation() == 90]
        assert sum(vertical) == pytest.approx(4, abs=0.01)

    def test_space_structure_is_3d(self, space_cantilever):
        """Verify a space frame diagram is drawn in a 3D axes."""
        ax = space_cantilever.results['tip'].plot_diagram('my')
        assert ax.name == '3d'

    def test_frames_without_the_force(self, plane_truss):
        """Verify frames that do not model the force get no diagram."""
        ax = plane_truss.results['load'].plot_diagram('mz')
        assert len(ax.get_lines()) == len(plane_truss.frames)

    def test_invalid_force_raises(self, portal):
        """Verify an unknown force raises ValueError."""
        with pytest.raises(ValueError, match='ux'):
            portal.results['dead'].plot_diagram('ux')


class TestPlotFrameSeriesOnGivenAxes:
    """Tests for the element plots drawn on given axes."""

    def test_given_axes(self, portal):
        """Verify the forces are drawn on the given axes, in order.

        The frames of the portal model fx, fy and mz; the figure of the
        axes is returned and keeps no title of its own.
        """
        fig, axes = plt.subplots(3, 1)
        results = portal.results['dead'].frames['bc']
        assert results.plot_internal_forces(axes=axes) is fig
        mz = results.internal_forces.mz
        np.testing.assert_allclose(_curve(axes[2], len(mz)), mz)
        assert fig.get_suptitle() == ''
        assert 'frame' in axes[0].get_title()

    def test_wrong_number_of_axes_raises(self, portal):
        """Verify the number of axes must match the active components."""
        fig, axes = plt.subplots(2, 1)
        with pytest.raises(ValueError, match='needs 3 axes, not 2'):
            portal.results['dead'].frames['bc'].plot_internal_forces(axes=axes)

    def test_shared_x_keeps_tick_labels(self, portal):
        """Verify the last given axes shows its x tick labels.

        With sharex, matplotlib hides them on all but the bottom axes, which
        may be another plot.
        """
        fig, axes = plt.subplots(3, 1, sharex=True)
        portal.results['dead'].frames['bc'].plot_internal_displacements(
            axes=axes[:2])
        assert axes[1].xaxis.get_tick_params()['labelbottom']

    def test_only_lowest_shows_x_axis(self, portal):
        """Verify that, of two element plots sharing x, only the lower one
        shows the x label and tick labels, whatever the drawing order.
        """
        frame = portal.results['dead'].frames['bc']
        for order in (1, -1):
            fig, axes = plt.subplots(5, 1, sharex=True)
            plots = [lambda: frame.plot_internal_forces(axes=axes[:3]),
                     lambda: frame.plot_internal_displacements(
                         axes=axes[3:])][::order]
            for plot in plots:
                plot()
            assert axes[2].get_xlabel() == ''
            assert not axes[2].xaxis.get_tick_params()['labelbottom']
            assert axes[4].get_xlabel() == "x from joint 'b'"
            assert axes[4].xaxis.get_tick_params()['labelbottom']
            plt.close(fig)


class TestPlotLoads:
    """Tests for Structure.plot_loads()."""

    def test_load_arrows(self, portal):
        """Verify the arrows of point and distributed loads.

        The lateral load is one arrow at joint b; the uniform load of the
        beam bc is a row of eleven arrows along it (ten intervals, about a
        twelfth of the 7.2 m diagonal each), in global axes.
        """
        arrows = plotting.load_arrows(portal, 'dead')
        forces = [a for a in arrows if a[0] == 'force']
        distributed = [a for a in arrows if a[0] == 'distributed']

        assert len(forces) == 1
        _, point, vector, label = forces[0]
        np.testing.assert_allclose(point, [0, 4, 0])
        np.testing.assert_allclose(vector, [20, 0, 0])
        assert label == '20'

        assert len(distributed) == 1
        _, points, vector, label = distributed[0]
        assert points.shape == (11, 3)
        np.testing.assert_allclose(points[:, 1], 4)
        np.testing.assert_allclose(vector, [0, -10, 0], atol=1e-12)
        assert label == 'q = 10'

    def test_frame_point_load_in_global_axes(self):
        """Verify a frame point load acts at its distance, in global axes.

        The column runs along +Y, so its local y-axis is -X.
        """
        s = Structure('plane_frame')
        s.add_joint('a', 0, 0)
        s.add_joint('b', 0, 4)
        s.add_material('m', E=1)
        s.add_section('s', A=1, Iz=1)
        s.add_frame('ab', 'a', 'b', 'm', 's', axial=True, bending_z=True)
        s.add_load_pattern('wind')
        s.load_patterns['wind'].add_frame_point_load('ab', 1, fy=5)

        (_, point, vector, _), = plotting.load_arrows(s, 'wind')
        np.testing.assert_allclose(point, [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(vector, [-5, 0, 0], atol=1e-12)

    def test_reaction_labels(self, portal):
        """Verify a reaction is labelled with its symbol, global axis and
        joint, and its value.
        """
        reaction = portal.results['dead'].reactions['a']
        labels = [label for _, _, _, label in
                  plotting.reaction_arrows(portal, 'dead')]
        assert f'$R_{{\\mathrm{{a}}}}^{{y}}$ = ' \
            f'{plotting._format(abs(reaction.fy))}' in labels
        assert f'$M_{{\\mathrm{{a}}}}^{{z}}$ = ' \
            f'{plotting._format(abs(reaction.mz))}' in labels

    def test_reaction_label_escapes_the_joint(self):
        """Verify the mathtext characters of a joint name are escaped."""
        assert plotting._reaction_label('force', 0, 'a_1', -2) == \
            '$R_{\\mathrm{a\\_1}}^{x}$ = 2'

    def test_wall_moment_goes_behind_the_wall(self):
        """Verify the moment reaction of a fixed support against a wall
        turns round a point behind it, clear of the frame.
        """
        s = Structure('beam')
        s.add_joint('A', x=0)
        s.add_joint('B', x=5)
        s.add_material('m', E=1)
        s.add_section('s', Iz=1)
        s.add_frame('AB', 'A', 'B', 'm', 's', bending_z=True)
        s.add_support('A', r_uy=True, r_rz=True)
        s.add_distributed_load('AB', fy=-12)
        s.run_analysis()

        ax = s.plot_loads()
        label, = [t for t in ax.texts if t.get_text().startswith('$M')]
        assert label.xy[0] < 0
        assert label.get_ha() == 'right'

    def test_reaction_arrows(self, portal):
        """Verify one arrow per nonzero reaction component, in equilibrium.

        The fixed bases have a horizontal force, a vertical force and a
        moment each; the forces balance the loads.
        """
        arrows = plotting.reaction_arrows(portal, 'dead')
        assert sorted(a[0] for a in arrows) == ['force'] * 4 + ['moment'] * 2

        total = sum(a[2] for a in arrows if a[0] == 'force')
        np.testing.assert_allclose(total, [-20, 6 * 10, 0], atol=1e-9)

    def test_reactions_drawn_beyond_supports(self, portal):
        """Verify the offsets move the forces, not the moments."""
        offset = np.array([0, -1, 0])
        arrows = plotting.reaction_arrows(portal, 'dead', {'a': offset})
        for kind, point, _, _ in arrows:
            if np.allclose(point[0], 0):
                expected = [0, -1, 0] if kind == 'force' else [0, 0, 0]
                np.testing.assert_allclose(point, expected)

    def test_without_analysis_only_loads(self):
        """Verify the loads can be plotted before run_analysis()."""
        s = Structure('beam')
        s.add_joint('a', x=0)
        s.add_joint('b', x=4)
        s.add_material('m', E=1)
        s.add_section('s', Iz=1)
        s.add_frame('ab', 'a', 'b', 'm', 's', bending_z=True)
        s.add_support('a', r_uy=True)
        s.add_load_pattern('dead')
        s.load_patterns['dead'].add_joint_point_load('b', fy=-10)

        ax = s.load_patterns['dead'].plot_loads()
        assert ax.get_title() == 'Loads: dead'
        assert not ax.axison

    def test_reactions_in_title(self, portal):
        """Verify the title says the reactions are drawn, unless not asked."""
        results = portal.results['dead']
        assert results.plot_loads().get_title() == 'Loads and reactions: dead'
        assert results.plot_loads(reactions=False).get_title() == 'Loads: dead'

    def test_no_dimensions_by_default(self, portal):
        """Verify the structure is not dimensioned by default."""
        ax = portal.results['dead'].plot_loads(reactions=False)
        assert '6' not in [t.get_text() for t in ax.texts]

    def test_dimensions_of_a_frame(self, portal):
        """Verify the width and height of the portal are dimensioned."""
        ax = portal.results['dead'].plot_loads(reactions=False,
                                               dimensions=True)
        dimensions = [t for t in ax.texts if t.get_text() in ('6', '4')]
        assert sorted(t.get_text() for t in dimensions) == ['4', '6']
        assert [t.get_rotation() for t in dimensions
                if t.get_text() == '4'] == [90]

    def test_dimensions_of_a_continuous_beam(self):
        """Verify the spans of a beam are dimensioned, without a total."""
        s = Structure('beam')
        for name, x in (('a', 0), ('b', 7), ('c', 15), ('d', 17)):
            s.add_joint(name, x=x)
        s.add_material('m', E=1)
        s.add_section('s', Iz=1)
        for j, k in ('ab', 'bc', 'cd'):
            s.add_frame(j + k, j, k, 'm', 's', bending_z=True)
        s.add_load_pattern('dead')
        ax = s.load_patterns['dead'].plot_loads(dimensions=True)
        assert sorted(t.get_text() for t in ax.texts) == ['2', '7', '8']

    def test_support_names(self, portal):
        """Verify the supports are named after their joints, only if
        asked.
        """
        results = portal.results['dead']
        texts = [t.get_text() for t in results.plot_loads().texts]
        assert 'a' not in texts and 'd' not in texts
        texts = [t.get_text() for t in
                 results.plot_loads(support_names=True).texts]
        assert 'a' in texts and 'd' in texts

    def test_support_names_on_the_left(self, portal):
        """Verify a support name is on the left of its joint."""
        ax = portal.load_patterns['dead'].plot_loads(support_names=True)
        name, = [t for t in ax.texts if t.get_text() == 'a']
        assert name.xy[0] < portal.joints['a'].x

    def test_wall_reactions(self):
        """Verify the vertical reaction of a wall acts below it, on its
        line, and its moment goes round a point behind it.
        """
        s = Structure('beam')
        s.add_joint('a', x=0)
        s.add_joint('b', x=5)
        s.add_material('m', E=1)
        s.add_section('s', Iz=1)
        s.add_frame('ab', 'a', 'b', 'm', 's', bending_z=True)
        s.add_support('a', r_uy=True, r_rz=True)
        s.add_distributed_load('ab', fy=-12)
        s.run_analysis()
        ax = s.plot_loads()
        force, = [t for t in ax.texts if t.get_text().startswith('$R')]
        moment, = [t for t in ax.texts if t.get_text().startswith('$M')]
        assert force.xy[0] == 0 and force.xy[1] < 0
        assert moment.xy[0] < 0

    def test_axes_first(self, portal):
        """Verify the axes are the first argument of every plot_loads."""
        import matplotlib.pyplot as plt

        for owner in (portal.load_patterns['dead'], portal.results['dead']):
            _, ax = plt.subplots()
            assert owner.plot_loads(ax) is ax

    def test_frame_names(self, portal):
        """Verify the frames are named, only if asked."""
        results = portal.results['dead']
        texts = [t.get_text() for t in results.plot_loads().texts]
        assert not {'ab', 'bc', 'cd'} & set(texts)
        texts = [t.get_text() for t in
                 results.plot_loads(frame_names=True).texts]
        assert {'ab', 'bc', 'cd'} <= set(texts)

    def test_frame_names_beside_their_middle(self, portal):
        """Verify a frame name is at the middle of its frame, below a beam
        and on the left of a column.
        """
        ax = portal.load_patterns['dead'].plot_loads(frame_names=True)
        names = {t.get_text(): t for t in ax.texts}
        beam, column = names['bc'], names['ab']
        assert tuple(beam.xy) == (3, 4) and beam.xyann[1] < 0
        assert tuple(column.xy) == (0, 2) and column.xyann[0] < 0

    def test_frame_names_in_circles(self, portal):
        """Verify the frame names are in circles, and the support names
        are not.
        """
        ax = portal.load_patterns['dead'].plot_loads(support_names=True,
                                                     frame_names=True)
        names = {t.get_text(): t for t in ax.texts}
        for frame in ('ab', 'bc', 'cd'):
            patch = names[frame].get_bbox_patch()
            assert type(patch.get_boxstyle()).__name__ == 'Circle'
        for joint in ('a', 'd'):
            assert names[joint].get_bbox_patch() is None

    def test_frame_names_in_3d(self, space_cantilever):
        """Verify the frames of a space structure are named."""
        ax = space_cantilever.results['tip'].plot_loads(frame_names=True)
        texts = [t.get_text().strip() for t in ax.texts]
        assert set(space_cantilever.frames) <= set(texts)

    def test_space_structure_has_no_dimensions(self, space_cantilever):
        """Verify a space structure is not dimensioned."""
        ax = space_cantilever.results['tip'].plot_loads(dimensions=True)
        assert ax.name == '3d'

    def test_space_structure_is_3d(self, space_cantilever):
        """Verify the loads of a space frame are drawn in a 3D axes."""
        assert space_cantilever.results['tip'].plot_loads().name == '3d'

    def test_load_pattern_plots_loads_only(self, portal):
        """Verify the load pattern plots its loads, without reactions."""
        ax = portal.load_patterns['dead'].plot_loads()
        assert ax.get_title() == 'Loads: dead'

    def test_unknown_load_pattern_raises(self, portal):
        """Verify plotting an unknown load pattern raises ValueError."""
        with pytest.raises(ValueError, match='live'):
            plotting.plot_loads(portal, 'live')


class TestSamePolygons:
    """Tests for the split of a diagram by sign."""

    def test_split_at_zero(self):
        """Verify the polygons end at the zero of a linear diagram."""
        base = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        values = np.array([-1.0, 0.5, 2.0])
        tip = base + values[:, None] * [0, 1, 0]
        polygons = plotting._same_sign_polygons(base, tip, values)

        assert [sign for _, sign in polygons] == [-1, 1]
        # each polygon runs along the base and back along the tip, so the
        # zero (-1 + 1.5 x = 0) closes the first base run and opens the next
        zero = [2 / 3, 0, 0]
        first, second = polygons[0][0], polygons[1][0]
        np.testing.assert_allclose(first, [base[0], zero, zero, tip[0]])
        np.testing.assert_allclose(second[0], zero)
        np.testing.assert_allclose(second[-1], zero)

    def test_split_on_a_zero_point(self):
        """Verify the polygons also end at a zero that falls on a point.

        The shear of a simple beam under a uniform load is zero exactly at
        midspan, which is one of the internal points.
        """
        base = np.array([[x, 0, 0] for x in range(5)], dtype=float)
        values = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        tip = base + values[:, None] * [0, 1, 0]
        polygons = plotting._same_sign_polygons(base, tip, values)

        assert [sign for _, sign in polygons] == [1, -1]
        np.testing.assert_allclose(polygons[0][0][2], base[2])
        np.testing.assert_allclose(polygons[1][0][0], base[2])

    def test_zero_ends_give_no_empty_polygon(self):
        """Verify a diagram that is zero at both ends gives one polygon."""
        base = np.array([[x, 0, 0] for x in range(3)], dtype=float)
        values = np.array([0.0, 1.0, 0.0])
        tip = base + values[:, None] * [0, 1, 0]
        polygons = plotting._same_sign_polygons(base, tip, values)

        assert [sign for _, sign in polygons] == [1]
