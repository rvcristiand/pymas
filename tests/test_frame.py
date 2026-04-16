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

    def test_add_frame(self, simple_structure):
        """Verify that a frame can be added with correct properties.

        Creates frame connecting N1 to N2 with material and section.
        """
        frame = simple_structure.elements['F1']
        assert frame.name == 'F1'
        assert frame.joint_j == 'N1'
        assert frame.joint_k == 'N2'
        assert frame.material == 'steel'
        assert frame.section == 'rect'

    def test_frame_length(self, simple_structure):
        """Verify frame length is calculated correctly.

        Between N1(0,0,0) and N2(5,0,0): length = 5.0
        """
        frame = simple_structure.elements['F1']
        assert frame.length() == pytest.approx(5.0, rel=1e-10)

    def test_frame_direction_cosines(self, simple_structure):
        """Verify direction cosines vector.

        For frame along x-axis: direction = [1, 0, 0]
        """
        simple_structure.set_degrees_freedom()
        frame = simple_structure.elements['F1']
        expected = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(frame.direction_cosines_vector(), expected)

    def test_frame_local_stiffness_matrix(self, simple_structure):
        """Verify local stiffness matrix includes bending terms.

        Local stiffness is 12x12 with non-zero terms for:
        - Axial (indices 0,6)
        - Torsion (index 3)
        - Shear/bending (indices 1,2,4,5,7,8,10,11)
        """
        simple_structure.set_degrees_freedom()
        frame = simple_structure.elements['F1']
        k_local = frame.local_stiffness_matrix()
        assert k_local.shape == (12, 12)
        # Torsion stiffness at diagonal
        assert k_local[3, 3] > 0
        # Bending stiffness at diagonals
        assert k_local[4, 4] > 0
        assert k_local[10, 10] > 0

    def test_frame_axial_stiffness_contribution(self, simple_structure):
        """Verify axial stiffness terms exist.

        Axial terms are at (0,0), (0,6), (6,0), (6,6).
        """
        simple_structure.set_degrees_freedom()
        frame = simple_structure.elements['F1']
        k_local = frame.local_stiffness_matrix()
        # Axial stiffness positive
        assert k_local[0, 0] > 0
        assert k_local[6, 6] > 0
        # Off-diagonal negative (consistency)
        assert k_local[0, 6] < 0
        assert k_local[6, 0] < 0

    def test_frame_global_stiffness_matrix(self, simple_structure):
        """Verify global stiffness matrix can be computed.

        Transforms local stiffness to global coordinate system.
        """
        simple_structure.set_degrees_freedom()
        simple_structure.set_joint_indices()
        frame = simple_structure.elements['F1']
        k_global = frame.global_stiffness_matrix()
        assert k_global.shape == (12, 12)

    def test_frame_rotation_matrix(self, simple_structure):
        """Verify rotation matrix is 3x3.

        Transforms 3D vectors between local and global coords.
        """
        simple_structure.set_degrees_freedom()
        frame = simple_structure.elements['F1']
        rotation = frame.rotation_matrix()
        assert rotation.shape == (3, 3)
        np.testing.assert_array_almost_equal(rotation, np.eye(3))

    def test_frame_rotation_transformation_matrix(self, simple_structure):
        """Verify rotation transformation matrix is 12x12.

        Transforms 12-DOF displacement/force vectors.
        """
        simple_structure.set_degrees_freedom()
        frame = simple_structure.elements['F1']
        t = frame.rotation_transformation_matrix()
        assert t.shape == (12, 12)