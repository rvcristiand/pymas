"""
Tests for the Frame class.

A Frame is a 2-node element whose stiffness terms are switched on by its
flags (rigid connections). It has 6 degrees of freedom per joint in 3D:
    - axial: axial deformation (local x)
    - torsional: torsion (about local x)
    - bending_y, bending_z: bending (about local y, z)

Key methods:
    - length(), direction_cosines_vector()
    - rotation_matrix(), rotation_transformation_matrix()
    - local_stiffness_matrix(), global_stiffness_matrix()

Other methods:
    - get_internal_forces(load_pattern): Axial, shear, moments along element
    - get_internal_displacements(load_pattern): Displacements along element

Typical usage:
    structure = Structure('space_frame')
    structure.add_material('steel', E=200e9)
    structure.add_rectangular_section('rect', dy=0.5, dz=0.3)
    structure.add_joint('N1', x=0)
    structure.add_joint('N2', x=6)
    structure.add_frame('F1', 'N1', 'N2', 'steel', 'rect', bending_z=True)
    frame = structure.frames['F1']
    k_local = frame.local_stiffness_matrix()
"""

import pytest
import numpy as np


class TestFrame:
    """Tests for Frame class functionality.

    A Frame connects two joints with full stiffness (axial + shear + bending).
    """

    def test_add_frame(self, simple_frame_structure):
        """Verify that a frame can be added with correct properties.

        Creates frame connecting N1 to N2 with material and section.
        """
        frame = simple_frame_structure.frames['F1']
        assert frame.name == 'F1'
        assert frame.joint_j == 'N1'
        assert frame.joint_k == 'N2'
        assert frame.material == 'steel'
        assert frame.section == 'rect'

    def test_frame_length(self, simple_frame_structure):
        """Verify frame length is calculated correctly.

        Between N1(0,0,0) and N2(5,0,0): length = 5.0
        """
        frame = simple_frame_structure.frames['F1']
        assert frame.length() == pytest.approx(5.0, rel=1e-10)

    def test_frame_direction_cosines(self, simple_frame_structure):
        """Verify direction cosines vector.

        For frame along x-axis: direction = [1, 0, 0]
        """
        frame = simple_frame_structure.frames['F1']
        expected = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(frame.direction_cosines_vector(), expected)

    def test_frame_local_stiffness_matrix(self, simple_frame_structure):
        """Verify each component of the 12x12 local stiffness matrix.

        This test explicitly calculates and verifies the theoretical stiffness values
        for all degrees of freedom of the frame element, ensuring that each of the
        individual stiffness components is correctly formulated:
        - Axial stiffness (degrees of freedom 0 and 6, along local x-axis)
        - Torsional stiffness (degrees of freedom 3 and 9, about local x-axis)
        - Bending and shear stiffness in the x-y plane (degrees of freedom 1, 5, 7, 11, about local z-axis)
        - Bending and shear stiffness in the x-z plane (degrees of freedom 2, 4, 8, 10, about local y-axis)
        """
        frame = simple_frame_structure.frames['F1']
        k_local = frame.local_stiffness_matrix()

        material = simple_frame_structure.materials[frame.material]
        section = simple_frame_structure.sections[frame.section]

        E = material.E
        G = material.G
        A = section.A
        Iy = section.Iy
        Iz = section.Iz
        J = section.J
        L = frame.length()

        k_expected = np.zeros((12, 12))

        # 1. Axial stiffness terms (local x-direction)
        k_axial = E * A / L
        k_expected[0, 0] = k_expected[6, 6] = k_axial
        k_expected[0, 6] = k_expected[6, 0] = -k_axial

        # 2. Torsional stiffness terms (about local x-axis)
        k_torsion = G * J / L
        k_expected[3, 3] = k_expected[9, 9] = k_torsion
        k_expected[3, 9] = k_expected[9, 3] = -k_torsion

        # 3. Bending and shear stiffness in x-y plane (about local z-axis)
        k_expected[1, 1] = 12 * E * Iz / L**3
        k_expected[1, 5] = 6 * E * Iz / L**2
        k_expected[1, 7] = -12 * E * Iz / L**3
        k_expected[1, 11] = 6 * E * Iz / L**2

        k_expected[5, 1] = 6 * E * Iz / L**2
        k_expected[5, 5] = 4 * E * Iz / L
        k_expected[5, 7] = -6 * E * Iz / L**2
        k_expected[5, 11] = 2 * E * Iz / L

        k_expected[7, 1] = -12 * E * Iz / L**3
        k_expected[7, 5] = -6 * E * Iz / L**2
        k_expected[7, 7] = 12 * E * Iz / L**3
        k_expected[7, 11] = -6 * E * Iz / L**2

        k_expected[11, 1] = 6 * E * Iz / L**2
        k_expected[11, 5] = 2 * E * Iz / L
        k_expected[11, 7] = -6 * E * Iz / L**2
        k_expected[11, 11] = 4 * E * Iz / L

        # 4. Bending and shear stiffness in x-z plane (about local y-axis)
        k_expected[2, 2] = 12 * E * Iy / L**3
        k_expected[2, 4] = -6 * E * Iy / L**2
        k_expected[2, 8] = -12 * E * Iy / L**3
        k_expected[2, 10] = -6 * E * Iy / L**2

        k_expected[4, 2] = -6 * E * Iy / L**2
        k_expected[4, 4] = 4 * E * Iy / L
        k_expected[4, 8] = 6 * E * Iy / L**2
        k_expected[4, 10] = 2 * E * Iy / L

        k_expected[8, 2] = -12 * E * Iy / L**3
        k_expected[8, 4] = 6 * E * Iy / L**2
        k_expected[8, 8] = 12 * E * Iy / L**3
        k_expected[8, 10] = 6 * E * Iy / L**2

        k_expected[10, 2] = -6 * E * Iy / L**2
        k_expected[10, 4] = 2 * E * Iy / L
        k_expected[10, 8] = 6 * E * Iy / L**2
        k_expected[10, 10] = 4 * E * Iy / L

        # Perform comprehensive assertion of the entire local stiffness matrix
        np.testing.assert_array_almost_equal(k_local, k_expected, decimal=5)

    def test_frame_axial_stiffness_contribution(self, simple_frame_structure):
        """Verify axial stiffness terms exist.

        Axial terms are at (0,0), (0,6), (6,0), (6,6).
        """
        frame = simple_frame_structure.frames['F1']
        k_local = frame.local_stiffness_matrix()
        # Axial stiffness positive
        assert k_local[0, 0] > 0
        assert k_local[6, 6] > 0
        # Off-diagonal negative (consistency)
        assert k_local[0, 6] < 0
        assert k_local[6, 0] < 0

    def test_frame_global_stiffness_matrix(self, simple_frame_structure):
        """Verify global stiffness matrix can be computed.

        Transforms local stiffness to global coordinate system.
        """
        simple_frame_structure.set_joint_indices()
        frame = simple_frame_structure.frames['F1']
        k_global = frame.global_stiffness_matrix()
        assert k_global.shape == (12, 12)

    def test_frame_rotation_matrix(self, simple_frame_structure):
        """Verify rotation matrix is 3x3.

        Transforms 3D vectors between local and global coords.
        """
        frame = simple_frame_structure.frames['F1']
        rotation = frame.rotation_matrix()
        assert rotation.shape == (3, 3)
        np.testing.assert_array_almost_equal(rotation, np.eye(3))

    def test_frame_rotation_transformation_matrix(self, simple_frame_structure):
        """Verify rotation transformation matrix is 12x12.

        Transforms 12-DOF displacement/force vectors.
        """
        frame = simple_frame_structure.frames['F1']
        t = frame.rotation_transformation_matrix()
        assert t.shape == (12, 12)


def _frame(direction, roll=None, Iy=2e-4, Iz=8e-4):
    """Create a space frame from the origin to `direction`.

    Args:
        direction (tuple): Coordinates of the far joint.
        roll (float, optional): Roll of the frame, in radians.
        Iy (float, optional): Inertia about the local y-axis.
        Iz (float, optional): Inertia about the local z-axis.

    Returns:
        Frame: Frame object.
    """
    from pymas import Structure

    s = Structure('space_frame')
    s.add_material('steel', E=200e9, G=77e9)
    s.add_section('section', A=0.01, J=1e-4, Iy=Iy, Iz=Iz)
    s.add_joint('j', x=0, y=0, z=0)
    s.add_joint('k', *direction)
    return s.add_frame('f', 'j', 'k', 'steel', 'section', axial=True,
                       torsional=True, bending_y=True, bending_z=True,
                       roll=roll)


DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1),
              (0, 0, -1), (3, 4, 5), (-2, 1, 7)]


class TestFrameRoll:
    """Tests for the roll of the frame cross section.

    The roll rotates the cross section about the local x-axis (right-hand
    rule, thumb from joint j to joint k), in radians.
    """

    # rotation matrices before roll was added
    REFERENCE = {
        (1, 0, 0): [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        (-1, 0, 0): [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        (0, 1, 0): [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        (0, -1, 0): [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        (0, 0, 1): [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        (0, 0, -1): [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        (3, 4, 5): [
            [0.42426406871192857, -0.5656854249492381, -0.7071067811865477],
            [0.5656854249492381, 0.7753225633997771, -0.2808467957502788],
            [0.7071067811865477, -0.2808467957502788, 0.6489415053121517]],
        (-2, 1, 7): [
            [-0.27216552697590884, -0.1360827634879543, -0.9525793444156804],
            [0.1360827634879543, 0.9745566894604818, -0.1781031737766272],
            [0.9525793444156804, -0.1781031737766272, -0.24672221643639064]],
    }

    @pytest.mark.parametrize('direction', DIRECTIONS)
    @pytest.mark.parametrize('roll', [None, 0])
    def test_no_roll_unchanged(self, direction, roll):
        """Verify roll None or 0 gives the rotation matrix used before."""
        np.testing.assert_allclose(_frame(direction, roll).rotation_matrix(),
                                   self.REFERENCE[direction], atol=1e-15)

    @pytest.mark.parametrize('direction, y, z', [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    ])
    def test_reference_orientation(self, direction, y, z):
        """Verify the local y- and z-axes at roll 0 (README table)."""
        r = _frame(direction).rotation_matrix()
        np.testing.assert_allclose(r[:, 1], y, atol=1e-15)
        np.testing.assert_allclose(r[:, 2], z, atol=1e-15)

    @pytest.mark.parametrize('direction', DIRECTIONS)
    @pytest.mark.parametrize('roll', [0.3, np.pi / 2, -2.0, np.pi])
    def test_rotation_matrix_is_proper(self, direction, roll):
        """Verify the local x-axis follows the frame and R is a rotation."""
        frame = _frame(direction, roll)
        r = frame.rotation_matrix()
        np.testing.assert_allclose(r @ [1, 0, 0],
                                   frame.direction_cosines_vector(),
                                   atol=1e-15)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-15)
        assert np.linalg.det(r) == pytest.approx(1)

    @pytest.mark.parametrize('direction', DIRECTIONS)
    @pytest.mark.parametrize('roll', [0.3, np.pi / 2, -2.0])
    def test_roll_rotates_local_y(self, direction, roll):
        """Verify the local y-axis turns by roll about the frame axis.

        Rodrigues' formula with the right-hand rule about the local x-axis.
        """
        x = _frame(direction).direction_cosines_vector()
        y0 = _frame(direction).rotation_matrix()[:, 1]
        expected = y0 * np.cos(roll) + np.cross(x, y0) * np.sin(roll) + \
            x * np.dot(x, y0) * (1 - np.cos(roll))
        np.testing.assert_allclose(_frame(direction, roll).rotation_matrix()
                                   [:, 1], expected, atol=1e-15)

    @pytest.mark.parametrize('direction', [(5, 0, 0), (0, 3, 0), (0, 0, 3),
                                           (3, 4, 5), (-2, 1, 7)])
    def test_quarter_roll_swaps_inertias(self, direction):
        """Verify roll pi/2 with (Iy, Iz) equals roll 0 with (Iz, Iy)."""
        k_roll = _frame(direction, np.pi / 2, Iy=2e-4, Iz=8e-4) \
            .global_stiffness_matrix()
        k_swap = _frame(direction, 0, Iy=8e-4, Iz=2e-4) \
            .global_stiffness_matrix()
        np.testing.assert_allclose(k_roll, k_swap, rtol=1e-9,
                                   atol=1e-9 * np.abs(k_swap).max())

    def test_roll_exported(self, tmp_path):
        """Verify the roll is exported, and omitted when None."""
        import json

        frame = _frame((3, 4, 5), 0.5)
        frame._parent.add_frame('g', 'j', 'k', 'steel', 'section')
        filename = tmp_path / 'roll.json'
        frame._parent.export(str(filename))
        with open(filename) as f:
            frames = json.load(f)['frames']
        assert frames['f']['roll'] == 0.5
        assert 'roll' not in frames['g']


class TestRollToAlign:
    """Tests for Frame.roll_to_align()."""

    def test_reference_case(self):
        """Verify the roll that points local z up for a (3, 4, 5) frame."""
        roll = _frame((3, 4, 5)).roll_to_align((0, 0, 1), axis='z')
        assert np.degrees(roll) == pytest.approx(23.4, abs=0.05)

    @pytest.mark.parametrize('direction, wanted', [
        (d, w) for d in DIRECTIONS for w in [(0, 0, 1), (1, 2, -3), (0, 1, 0)]
        if np.linalg.norm(np.cross(d, w)) > 0])
    @pytest.mark.parametrize('axis, column', [('y', 1), ('z', 2)])
    def test_aligned_axis(self, direction, wanted, axis, column):
        """Verify the aligned axis matches the projected wanted direction."""
        frame = _frame(direction)
        x = frame.direction_cosines_vector()
        w = np.array(wanted, dtype=float)
        projection = w - np.dot(w, x) * x

        frame.roll = frame.roll_to_align(wanted, axis=axis)
        np.testing.assert_allclose(frame.rotation_matrix()[:, column],
                                   projection / np.linalg.norm(projection),
                                   atol=1e-12)

    def test_parallel_raises(self):
        """Verify a wanted direction parallel to the frame raises."""
        with pytest.raises(ValueError):
            _frame((3, 4, 5)).roll_to_align((-6, -8, -10))

    def test_invalid_axis_raises(self):
        """Verify an axis other than 'y' or 'z' raises."""
        with pytest.raises(ValueError):
            _frame((3, 4, 5)).roll_to_align((0, 0, 1), axis='x')
