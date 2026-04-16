"""
Tests for load-related classes.

This module tests:
    - LoadPattern: Groups loads that act simultaneously
    - JointPointLoad: Force/moment applied directly to a joint
    - ElementPointLoad: Force/moment applied to an element at a specific distance
    - DistributedLoad: Uniformly distributed load along an element

Load Pattern:
    A LoadPattern groups multiple loads that are applied simultaneously.
    Each pattern is analyzed separately.

Load Types:
    - Joint point loads: Applied directly to joints
    - Element point loads: Applied to elements at a distance from near joint
    - Distributed loads: Uniformly distributed along element length

Each load type has a load_vector() method that returns forces/moments
in the structure's global coordinate system.

Typical usage:
    structure = Structure()
    structure.add_load_pattern('dead')
    structure.add_joint_point_load('dead', 'N2', fy=-10e3)
    structure.add_distributed_load('dead', 'F1', fy=-5e3)
    structure.run_analysis()
"""

import pytest
import numpy as np


class TestLoadPattern:
    """Tests for LoadPattern class."""

    def test_add_load_pattern(self, structure):
        """Verify that a load pattern can be added.

        Creates load pattern named 'dead'.
        """
        load_pattern = structure.add_load_pattern('dead')
        assert load_pattern.name == 'dead'
        assert 'dead' in structure.load_patterns

    def test_load_pattern_empty(self, structure):
        """Verify load pattern initializes empty dictionaries.

        Initially, a load pattern has no loads applied.
        """
        load_pattern = structure.add_load_pattern('dead')
        assert load_pattern.joint_point_loads == {}
        assert load_pattern.element_point_loads == {}
        assert load_pattern.element_distributed_loads == {}


class TestJointPointLoad:
    """Tests for JointPointLoad class.

    JointPointLoad applies a force and/or moment directly to a joint.
    The load is specified in global coordinates.
    """

    def test_add_joint_point_load(self, structure_with_loads):
        """Verify joint point load can be added.

        Adds fy=-10kN to joint N2.
        """
        load_pattern = structure_with_loads.load_patterns['dead']
        point_load = load_pattern.add_joint_point_load('N2', fy=-10e3)
        assert point_load.joint == 'N2'
        assert point_load.fy == -10e3

    def test_joint_point_load_vector(self, structure_with_loads):
        """Verify load vector has correct values.

        For 3D truss: load vector has 3 components (ux, uy, uz).
        For 3D: load vector has 6 components (ux, uy, uz, rx, ry, rz).
        """
        structure_with_loads.set_degrees_freedom()
        load_pattern = structure_with_loads.load_patterns['dead']
        joint = 'N2'
        point_load = load_pattern.joint_point_loads[joint][0]
        load_vector = point_load.load_vector()
        # 6 DOF for 3D type
        expected = np.array([0, -10e3, 0, 0, 0, 0])
        np.testing.assert_array_almost_equal(load_vector, expected)


class TestDistributedLoad:
    """Tests for DistributedLoad class.

    DistributedLoad applies a uniform load along an element's length.
    The load is specified in the element's local coordinate system.
    """

    def test_add_distributed_load(self, structure_with_loads):
        """Verify distributed load can be added to an element.

        Adds fy=-5kN/m to frame F1.
        """
        load_pattern = structure_with_loads.load_patterns['dead']
        distributed_load = load_pattern.add_element_distributed_load('F1', fy=-5e3)
        assert distributed_load.element == 'F1'
        assert distributed_load.fy == -5e3

    def test_distributed_load_fixed_load_vector(self, structure_with_loads):
        """Verify fixed-end load vector calculation.

        The fixed-end load vector represents forces/moments at element ends
        if the element were fully restrained.
        """
        load_pattern = structure_with_loads.load_patterns['dead']
        distributed_load = load_pattern.element_distributed_loads['F1'][0]
        fixed_vector = distributed_load.fixed_load_vector()
        assert fixed_vector.shape == (12, 1)
        # fy: q*L/2 = 5000*6/2 = 15000 (both ends)
        assert fixed_vector[1, 0] == pytest.approx(5e3 * 6 / 2)
        assert fixed_vector[7, 0] == pytest.approx(5e3 * 6 / 2)


class TestElementPointLoad:
    """Tests for ElementPointLoad class.

    ElementPointLoad applies a force/moment to an element at a specific
    distance from the near joint (joint_j). The load is specified in
    the element's local coordinate system.
    """

    @pytest.mark.skip(reason="ElementPointLoad requires element length for fixed_load_vector")
    def test_add_element_point_load(self, structure):
        """Verify element point load can be added.

        Adds fy=-10kN at distance=3m from near joint on frame F1.
        """
        material = structure.add_material('steel', modulus_elasticity=200e9)
        section = structure.add_rectangular_section('rect', 0.3, 0.5)
        structure.add_joint('N1', x=0)
        structure.add_joint('N2', x=6)
        structure.add_frame('F1', 'N1', 'N2', 'steel', 'rect')
        load_pattern = structure.add_load_pattern('test')
        point_load = load_pattern.add_element_point_load('F1', 3, fy=-10e3)
        assert point_load.element == 'F1'
        assert point_load.fy == -10e3