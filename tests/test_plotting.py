"""
Tests for the plots of the analysis results.

The plots require matplotlib, an optional dependency (`pip install
pymas[plot]`); these tests are skipped when it is not installed. They check
the data drawn, not the look of the figures.

Methods of Structure:
    - plot_deformed(load_pattern, scale=None, ax=None)
    - plot_internal_forces(load_pattern, frame)
    - plot_internal_displacements(load_pattern, frame)
    - plot_diagram(load_pattern, force, scale=None, ax=None)

Typical usage:
    structure.run_analysis()
    ax = structure.plot_deformed('dead')
    ax.figure.savefig('deformed.png')
"""

import sys

import numpy as np
import pytest

from pymas import Structure, plotting

matplotlib = pytest.importorskip('matplotlib')
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
    s.add_distributed_load('dead', 'bc', fy=-10)
    s.add_joint_point_load('dead', 'b', fx=20)
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
    s.add_joint_point_load('tip', 'b', fx=1, fy=-2, fz=3)
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
    s.add_joint_point_load('load', 'c', fx=5, fy=-10)
    s.run_analysis()

    return s


def _ends(structure, load_pattern, name, scale):
    """Return the deformed positions expected at both ends of a frame."""
    frame = structure.frames[name]
    ends = []
    for joint in (frame.joint_j, frame.joint_k):
        d = structure.displacements[load_pattern][joint].displacement_vector()
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
        ax = portal.plot_deformed('dead')
        assert ax.name != '3d'

    def test_space_structure_is_3d(self, space_cantilever):
        """Verify a space frame is drawn in a 3D axes."""
        ax = space_cantilever.plot_deformed('tip')
        assert ax.name == '3d'

    def test_lines(self, portal):
        """Verify an undeformed and a deformed line per frame.

        The deformed line of each frame ends at its moved far joint.
        """
        ax = portal.plot_deformed('dead', scale=50)
        lines = ax.get_lines()
        assert len(lines) == 2 * len(portal.frames)

        for name, deformed in zip(portal.frames, lines[1::2]):
            _, k = _ends(portal, 'dead', name, 50)
            np.testing.assert_allclose(
                [deformed.get_xdata()[-1], deformed.get_ydata()[-1]], k[:2],
                atol=1e-9)

    def test_auto_scale(self, portal):
        """Verify the largest displacement is 10% of the structure size."""
        scale = plotting._auto_scale(portal, 'dead')
        largest = max(
            np.abs(plotting.deformed_positions(portal, 'dead', f, scale) -
                   plotting.frame_positions(f, 101)).max()
            for f in portal.frames.values())
        assert largest == pytest.approx(0.1 * np.hypot(6, 4))

    def test_given_axes(self, portal):
        """Verify the plot is drawn on the given axes."""
        fig, ax = plt.subplots()
        assert portal.plot_deformed('dead', ax=ax) is ax

    def test_without_analysis_raises(self):
        """Verify plotting before run_analysis() raises ValueError."""
        s = Structure('beam')
        s.add_load_pattern('dead')
        with pytest.raises(ValueError, match='run_analysis'):
            s.plot_deformed('dead')

    def test_unknown_load_pattern_raises(self, portal):
        """Verify an unknown load pattern raises ValueError."""
        with pytest.raises(ValueError, match='live'):
            portal.plot_deformed('live')


class TestPlotFrameSeries:
    """Tests for plot_internal_forces() and plot_internal_displacements()."""

    def test_internal_forces_of_active_flags(self, portal):
        """Verify one axes per modelled force, plotting its values."""
        fig = portal.plot_internal_forces('dead', 'bc')
        labels = [ax.get_ylabel() for ax in fig.axes]
        assert labels == ['Axial force fx', 'Shear force fy',
                          'Bending moment mz']

        mz = fig.axes[2].get_lines()[-1].get_ydata()
        np.testing.assert_allclose(
            mz, portal.internal_forces['dead']['bc'].mz)

    def test_internal_displacements_of_active_flags(self, space_cantilever):
        """Verify the six displacements of a full space frame."""
        fig = space_cantilever.plot_internal_displacements('tip', 'ab')
        assert len(fig.axes) == 6

        uz = fig.axes[2].get_lines()[-1].get_ydata()
        np.testing.assert_allclose(
            uz, space_cantilever.internal_displacements['tip']['ab'].uz)

    def test_length_axis(self, portal):
        """Verify the x data spans the length of the frame."""
        fig = portal.plot_internal_forces('dead', 'ab')
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
        s.add_joint_point_load('dead', 'b', fy=-1)
        s.run_analysis()
        with pytest.raises(ValueError, match="'bc'"):
            s.plot_internal_forces('dead', 'bc')


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
        ax = portal.plot_diagram('dead', 'mz')
        largest = max(np.abs(portal.internal_forces['dead'][f].mz).max()
                      for f in portal.frames)
        # the diagram lines are drawn after each frame line
        size = np.hypot(6, 4)
        for name, tip_line in zip(portal.frames, ax.get_lines()[1::2]):
            frame = portal.frames[name]
            _, tip, _ = plotting.diagram_positions(
                portal, 'dead', frame, 'mz', 0.15 * size / largest)
            np.testing.assert_allclose(tip_line.get_xdata(), tip[:, 0])

    def test_space_structure_is_3d(self, space_cantilever):
        """Verify a space frame diagram is drawn in a 3D axes."""
        ax = space_cantilever.plot_diagram('tip', 'my')
        assert ax.name == '3d'

    def test_frames_without_the_force(self, plane_truss):
        """Verify frames that do not model the force get no diagram."""
        ax = plane_truss.plot_diagram('load', 'mz')
        assert len(ax.get_lines()) == len(plane_truss.frames)

    def test_invalid_force_raises(self, portal):
        """Verify an unknown force raises ValueError."""
        with pytest.raises(ValueError, match='ux'):
            portal.plot_diagram('dead', 'ux')


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


def test_missing_matplotlib(portal, monkeypatch):
    """Verify a helpful ImportError when matplotlib is not installed."""
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', None)
    with pytest.raises(ImportError, match=r'pymas\[plot\]'):
        portal.plot_deformed('dead')
