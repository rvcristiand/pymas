"""
Tests for truss members.

A truss member is a Frame with only the `axial` flag set: it transmits axial
force only (hinged connections at both ends). In a 'space_truss' structure it
has 3 degrees of freedom per joint: ux, uy, uz.

Key methods:
    - length(): Element length between joints
    - direction_cosines_vector(): Unit vector along truss axis
    - rotation_matrix(): 3x3 rotation from local to global
    - rotation_transformation_matrix(): 12x12 transformation matrix
    - local_stiffness_matrix(): 12x12 stiffness in local coords
    - global_stiffness_matrix(): Stiffness in global coords

Typical usage:
    structure = Structure('space_truss')
    structure.add_material('steel', E=200e9)
    structure.add_section('circle', A=0.01)
    structure.add_joint('N1', x=0)
    structure.add_joint('N2', x=5)
    structure.add_frame('T1', 'N1', 'N2', 'steel', 'circle', axial=True)
    truss = structure.frames['T1']
    length = truss.length()
"""

import pytest
import numpy as np


class TestTruss:
    """Tests for truss members (axial-only frames).

    A truss member connects two joints with axial stiffness only.
    """

    def test_add_truss(self, simple_truss_structure):
        """Verify that a truss can be added with correct properties.

        Creates truss connecting N1 to N2 with material 'steel' and section 'circle'.
        """
        truss = simple_truss_structure.frames['T1']
        assert truss.name == 'T1'
        assert truss.joint_j == 'N1'
        assert truss.joint_k == 'N2'
        assert truss.material == 'steel'
        assert truss.section == 'circle'

    def test_truss_length(self, simple_truss_structure):
        """Verify that truss length is calculated correctly.

        Length = distance between joints N1(0,0,0) and N2(5,0,0) = 5.0
        """
        truss = simple_truss_structure.frames['T1']
        assert truss.length() == pytest.approx(5.0, rel=1e-10)

    def test_truss_direction_cosines(self, simple_truss_structure):
        """Verify direction cosines vector is a unit vector along truss axis.

        For truss along x-axis from (0,0,0) to (5,0,0), direction is [1,0,0].
        """
        truss = simple_truss_structure.frames['T1']
        expected = np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(truss.direction_cosines_vector(), expected)

    def test_truss_rotation_matrix(self, simple_truss_structure):
        """Verify rotation matrix transforms local to global coords.

        For horizontal truss, rotation matrix equals identity ( aligned with x-axis).
        """
        truss = simple_truss_structure.frames['T1']
        rotation = truss.rotation_matrix()
        assert rotation.shape == (3, 3)
        np.testing.assert_array_almost_equal(rotation, np.eye(3))

    def test_truss_local_stiffness_matrix(self, simple_truss_structure):
        """Verify local stiffness matrix has correct shape.

        Local stiffness matrix is 12x12 for a 2-node truss element.
        """
        truss = simple_truss_structure.frames['T1']
        k_local = truss.local_stiffness_matrix()
        assert k_local.shape == (12, 12)

    def test_truss_global_stiffness_matrix(self, simple_truss_structure):
        """Verify global stiffness matrix can be computed.

        The 12x12 matrix is filtered by the active degrees of freedom: a
        space truss has 3 per joint, so the result is 6x6.
        """
        simple_truss_structure.set_joint_indices()
        truss = simple_truss_structure.frames['T1']
        k_global = truss.global_stiffness_matrix()
        assert k_global.shape == (6, 6)

    def test_truss_rotation_transformation_matrix(self, simple_truss_structure):
        """Verify rotation transformation matrix for 12 DOF.

        This matrix transforms displacement/force vectors between
        local and global coordinate systems.
        """
        truss = simple_truss_structure.frames['T1']
        t = truss.rotation_transformation_matrix()
        assert t.shape == (12, 12)
