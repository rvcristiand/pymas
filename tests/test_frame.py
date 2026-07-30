"""
Tests for the Frame class.

A Frame is a 2-node element with axial, shear, and bending stiffness
(rigid connections). It has 6 degrees of freedom per joint in 3D.

The Frame class extends Truss with bending stiffness:
    - Axial deformation (local x)
    - Shear deformation (local y, z)
    - Bending (about local y, z)

Key methods (inherited from Truss):
    - length(), direction_cosines_vector()
    - rotation_matrix(), rotation_transformation_matrix()
    - local_stiffness_matrix(), global_stiffness_matrix()

Frame-specific methods:
    - get_internal_forces(load_pattern): Axial, shear, moments along element
    - get_internal_displacements(load_pattern): Displacements along element

Typical usage:
    structure = Structure()
    structure.add_material('steel', modulus_elasticity=200e9)
    structure.add_rectangular_section('rect', 0.3, 0.5)
    structure.add_joint('N1', x=0)
    structure.add_joint('N2', x=6)
    structure.add_frame('F1', 'N1', 'N2', 'steel', 'rect')
    frame = structure.elements['F1']
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
        frame = simple_frame_structure.elements['F1']
        assert frame.length() == pytest.approx(5.0, rel=1e-10)

    def test_frame_direction_cosines(self, simple_frame_structure):
        """Verify direction cosines vector.

        For frame along x-axis: direction = [1, 0, 0]
        """
        simple_frame_structure.set_degrees_freedom()
        frame = simple_frame_structure.elements['F1']
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
        simple_frame_structure.set_degrees_freedom()
        frame = simple_frame_structure.elements['F1']
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
        simple_frame_structure.set_degrees_freedom()
        frame = simple_frame_structure.elements['F1']
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
        simple_frame_structure.set_degrees_freedom()
        simple_frame_structure.set_joint_indices()
        frame = simple_frame_structure.elements['F1']
        k_global = frame.global_stiffness_matrix()
        assert k_global.shape == (12, 12)

    def test_frame_rotation_matrix(self, simple_frame_structure):
        """Verify rotation matrix is 3x3.

        Transforms 3D vectors between local and global coords.
        """
        simple_frame_structure.set_degrees_freedom()
        frame = simple_frame_structure.elements['F1']
        rotation = frame.rotation_matrix()
        assert rotation.shape == (3, 3)
        np.testing.assert_array_almost_equal(rotation, np.eye(3))

    def test_frame_rotation_transformation_matrix(self, simple_frame_structure):
        """Verify rotation transformation matrix is 12x12.

        Transforms 12-DOF displacement/force vectors.
        """
        simple_frame_structure.set_degrees_freedom()
        frame = simple_frame_structure.elements['F1']
        t = frame.rotation_transformation_matrix()
        assert t.shape == (12, 12)
