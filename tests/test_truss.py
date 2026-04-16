"""
Tests for the Truss class.

A Truss is a linear element that transmits axial force only (hinged connections at both ends).
It has 3 degrees of freedom per joint in 3D: ux, uy, uz.

Key methods:
    - length(): Element length between joints
    - direction_cosines_vector(): Unit vector along truss axis
    - rotation_matrix(): 3x3 rotation from local to global
    - rotation_transformation_matrix(): 12x12 transformation matrix
    - local_stiffness_matrix(): 12x12 stiffness in local coords
    - global_stiffness_matrix(): Stiffness in global coords

Typical usage:
    structure = Structure()
    structure.add_material('steel', modulus_elasticity=200e9)
    structure.add_section('circle', area=0.01)
    structure.add_joint('N1', x=0)
    structure.add_joint('N2', x=5)
    structure.add_truss('T1', 'N1', 'N2', 'steel', 'circle')
    truss = structure.elements['T1']
    length = truss.length()
"""

import pytest
import numpy as np


class TestTruss:
    """Tests for Truss class functionality.

    A Truss connects two joints with axial stiffness only.
    """

    def test_add_truss(self, simple_structure):
        """Verify that a truss can be added with correct properties.

        Creates truss connecting N1 to N2 with material 'steel' and section 'circle'.
        """
        truss = simple_structure.elements['T1']
        assert truss.name == 'T1'
        assert truss.joint_j == 'N1'
        assert truss.joint_k == 'N2'
        assert truss.material == 'steel'
        assert truss.section == 'circle'

    def test_truss_length(self, simple_structure):
        """Verify that truss length is calculated correctly.

        Length = distance between joints N1(0,0,0) and N2(5,0,0) = 5.0
        """
        truss = simple_structure.elements['T1']
        assert truss.length() == pytest.approx(5.0, rel=1e-10)

    def test_truss_direction_cosines(self, simple_structure):
        """Verify direction cosines vector is a unit vector along truss axis.

        For truss along x-axis from (0,0,0) to (5,0,0), direction is [1,0,0].
        """
        simple_structure.type = '3D'
        simple_structure.set_degrees_freedom()
        truss = simple_structure.elements['T1']
        expected = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(truss.direction_cosines_vector(), expected)

    def test_truss_rotation_matrix(self, simple_structure):
        """Verify rotation matrix transforms local to global coords.

        For horizontal truss, rotation matrix equals identity ( aligned with x-axis).
        """
        simple_structure.type = '3D'
        simple_structure.set_degrees_freedom()
        truss = simple_structure.elements['T1']
        rotation = truss.rotation_matrix()
        assert rotation.shape == (3, 3)
        np.testing.assert_array_almost_equal(rotation, np.eye(3))

    def test_truss_local_stiffness_matrix(self, simple_structure):
        """Verify local stiffness matrix has correct shape.

        Local stiffness matrix is 12x12 for a 2-node truss element.
        """
        simple_structure.set_degrees_freedom()
        truss = simple_structure.elements['T1']
        k_local = truss.local_stiffness_matrix()
        assert k_local.shape == (12, 12)

    def test_truss_global_stiffness_matrix(self, simple_structure):
        """Verify global stiffness matrix can be computed.

        Uses degrees of freedom to filter the stiffness matrix.
        """
        simple_structure.set_degrees_freedom()
        simple_structure.set_joint_indices()
        truss = simple_structure.elements['T1']
        k_global = truss.global_stiffness_matrix()
        assert k_global.shape == (12, 12)

    def test_truss_rotation_transformation_matrix(self, simple_structure):
        """Verify rotation transformation matrix for 12 DOF.

        This matrix transforms displacement/force vectors between
        local and global coordinate systems.
        """
        simple_structure.set_degrees_freedom()
        truss = simple_structure.elements['T1']
        t = truss.rotation_transformation_matrix()
        assert t.shape == (12, 12)