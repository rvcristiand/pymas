"""
Tests for the Joint class.

A Joint represents a node in the structural model with spatial coordinates (x, y, z).
Joints define the endpoints of structural elements (trusses, frames) and can have
supports and loads applied to them.

Typical usage:
    structure = Structure()
    joint = structure.add_joint('N1', x=0, y=0, z=0)
    coordinates = joint.coordinate_vector()  # array([0, 0, 0])
"""

import pytest
import numpy as np


class TestJoint:
    """Tests for Joint class functionality.

    A Joint stores:
        - name: Joint identifier
        - x, y, z: Coordinates in 3D space (None if not specified)
    """

    def test_add_joint(self, structure):
        """Verify that a joint can be added with all coordinates.

        Creates joint 'N1' at origin (0,0,0).
        """
        joint = structure.add_joint('N1', x=0, y=0, z=0)
        assert joint.name == 'N1'
        assert joint.x == 0
        assert joint.y == 0
        assert joint.z == 0
        assert 'N1' in structure.joints

    def test_joint_default_coordinates(self, structure):
        """Verify that joint coordinates default to None when not specified.

        When no coordinates provided, x, y, z should all be None.
        """
        joint = structure.add_joint('N1')
        assert joint.x is None
        assert joint.y is None
        assert joint.z is None

    def test_coordinate_vector(self, structure):
        """Verify that coordinate vector is returned correctly.

        coordinate_vector() returns numpy array [x, y, z].
        """
        structure.add_joint('N1', x=1, y=2, z=3)
        joint = structure.joints['N1']
        expected = np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(joint.coordinate_vector(), expected)

    def test_coordinate_vector_with_none(self, structure):
        """Verify that None coordinates default to 0 in coordinate vector.

        When coordinate is None, it should default to 0 in the vector.
        """
        structure.add_joint('N1')
        joint = structure.joints['N1']
        expected = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(joint.coordinate_vector(), expected)

    def test_coordinate_vector_partial_coordinates(self, structure):
        """Verify that partial coordinates work correctly.

        Only specified coordinates should have their values;
        unspecified should default to 0.
        """
        structure.add_joint('N1', x=1, z=3)
        joint = structure.joints['N1']
        expected = np.array([1, 0, 3])
        np.testing.assert_array_almost_equal(joint.coordinate_vector(), expected)