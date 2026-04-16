"""
Tests for the Support class.

A Support defines boundary conditions at a joint, specifying which degrees of freedom
are restrained. Supports prevent translation and/or rotation at a joint.

Degrees of freedom in 3D:
    - Translational: ux, uy, uz (translation along x, y, z axes)
    - Rotational: rx, ry, rz (rotation about x, y, z axes)

Methods:
    - restrain_vector(): Returns boolean array of restrained DOFs (filtered by structure type)

Typical usage:
    structure = Structure()
    structure.add_joint('N1', x=0, y=0, z=0)
    structure.add_support('N1', r_ux=True, r_uy=True, r_uz=True)
    support = structure.supports['N1']
    restrains = support.restrain_vector()  # array([True, True, True, False, False, False])
"""

import pytest
import numpy as np


class TestSupport:
    """Tests for Support class functionality.

    A Support restrains specific degrees of freedom at a joint.
    """

    def test_add_support(self, simple_structure):
        """Verify that a support can be added with correct properties.

        Creates support at N1 with full restraints.
        """
        support = simple_structure.supports['N1']
        assert support.joint == 'N1'
        assert support.r_ux is True
        assert support.r_uy is True
        assert support.r_uz is True

    def test_restrain_vector_3d(self, simple_structure):
        """Verify restrain vector for 3D structure.

        For fully restrained support, all 6 DOFs should be True.
        """
        simple_structure.set_degrees_freedom()
        support = simple_structure.supports['N1']
        restrains = support.restrain_vector()
        expected = np.array([True, True, True, True, True, True])
        np.testing.assert_array_equal(restrains, expected)

    def test_restrain_vector_partial(self, simple_structure):
        """Verify restrain vector when only some DOFs are restrained.

        For support with only uy and rz restrained: [False, True, False, False, False, True]
        """
        simple_structure.set_degrees_freedom()
        support = simple_structure.supports['N2']
        restrains = support.restrain_vector()
        expected = np.array([False, True, False, False, False, True])
        np.testing.assert_array_equal(restrains, expected)

    def test_support_default_restrains(self, structure):
        """Verify that restraints default to None (unrestrained).

        When no restraints specified, all should be None (treated as False).
        """
        structure.add_joint('N1', x=0, y=0, z=0)
        support = structure.add_support('N1')
        assert support.r_ux is None
        assert support.r_uy is None

    def test_support_restrain_vector_defaults(self, simple_structure):
        """Verify None restraints are treated as False.

        When restrain value is None, it should become False in vector.
        """
        simple_structure.set_degrees_freedom()
        simple_structure.add_support('N1', r_uy=True)
        support = simple_structure.supports['N1']
        restrains = support.restrain_vector()
        expected = np.array([False, True, False, False, False, False])
        np.testing.assert_array_equal(restrains, expected)