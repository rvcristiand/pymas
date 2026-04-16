"""
Pytest configuration and fixtures for pymas tests.

This module provides reusable fixtures for testing the pymas structural analysis library.
Each fixture creates a typical structure scenario for testing different functionalities.

Fixtures:
    structure: Empty structure ready to be populated.
    simple_structure: A basic 3D structure with joints, truss, frame, and supports.
    structure_with_loads: A structure with multiple elements and applied loads.
    structure_and_load: A minimal structure for element point load testing.

Typical usage:
    def test_example(simple_structure):
        # simple_structure already has materials, sections, joints, elements, supports
        simple_structure.run_analysis()
        assert 'dead' in simple_structure.displacements
"""

import pytest
import numpy as np
from pymas import Structure


@pytest.fixture
def structure():
    """Create an empty Structure object.

    Returns:
        Structure: An empty structure ready to be populated with
            materials, sections, joints, elements, supports, and loads.
    """
    return Structure()


@pytest.fixture
def simple_structure():
    """Create a simple tested 3D structure.

    The structure contains:
        - Material 'steel' with E=200e9 GPa, G=77e9 GPa
        - Section 'circle' with area=0.01, J=0.0001, Iy=0.0001, Iz=0.0001
        - RectangularSection 'rect' with base=0.3, height=0.5
        - Joints N1(0,0,0) and N2(5,0,0)
        - Truss T1 connecting N1-N2
        - Frame F1 connecting N1-N2
        - Support at N1 (fully restrained: ux,uy,uz,rx,ry,rz)
        - Support at N2 (partially restrained: uy, rz)

    Returns:
        Structure: A configured 3D structure for testing.
    """
    s = Structure()

    s.add_material('steel', modulus_elasticity=200e9, modulus_elasticity_shear=77e9)
    s.add_section('circle', area=0.01, torsion_constant=0.0001,
                 inertia_y=0.0001, inertia_z=0.0001)
    s.add_rectangular_section('rect', 0.3, 0.5)
    s.add_joint('N1', x=0, y=0, z=0)
    s.add_joint('N2', x=5, y=0, z=0)
    s.add_truss('T1', 'N1', 'N2', 'steel', 'circle')
    s.add_frame('F1', 'N1', 'N2', 'steel', 'rect')
    s.add_support('N1', r_ux=True, r_uy=True, r_uz=True,
                 r_rx=True, r_ry=True, r_rz=True)
    s.add_support('N2', r_uy=True, r_rz=True)

    return s


@pytest.fixture
def structure_with_loads():
    """Create a tested structure with loads applied.

    The structure contains:
        - Materials 'concrete' (E=30e9) and 'steel' (E=200e9)
        - RectangularSection 'rect' with base=0.3, height=0.5
        - Joints N1(0,0), N2(6,0), N3(12,0) (2D beam type)
        - Frames F1 (N1-N2) and F2 (N2-N3)
        - Support at N1 (fully restrained)
        - Support at N3 (partially restrained: uy, rz)
        - LoadPattern 'dead' with:
            - Point load at N2: fy=-10kN
            - Distributed load on F1: fy=-5kN/m
            - Distributed load on F2: fy=-5kN/m

    Note: This fixture has known issues with singular stiffness matrix.

    Returns:
        Structure: A structure configured with loads.
    """
    s = Structure()

    s.add_material('concrete', modulus_elasticity=30e9)
    s.add_material('steel', modulus_elasticity=200e9)
    s.add_rectangular_section('rect', 0.3, 0.5)
    s.add_joint('N1', x=0, y=0)
    s.add_joint('N2', x=6, y=0)
    s.add_joint('N3', x=12, y=0)
    s.add_frame('F1', 'N1', 'N2', 'steel', 'rect')
    s.add_frame('F2', 'N2', 'N3', 'steel', 'rect')
    s.add_support('N1', r_ux=True, r_uy=True, r_uz=True, r_rx=True, r_ry=True, r_rz=True)
    s.add_support('N3', r_uy=True, r_rz=True)
    s.add_load_pattern('dead')
    s.add_joint_point_load('dead', 'N2', fy=-10e3)
    s.add_distributed_load('dead', 'F1', fy=-5e3)
    s.add_distributed_load('dead', 'F2', fy=-5e3)

    return s


@pytest.fixture
def structure_and_load():
    """Create a minimal structure for element point load testing.

    The structure contains:
        - Material 'steel' with E=200e9
        - RectangularSection 'rect' with base=0.3, height=0.5
        - Joints N1(0,0) and N2(6,0)
        - Frame F1 connecting N1-N2

    Returns:
        Structure: Minimal structure without loads applied yet.
    """
    s = Structure()
    s.add_material('steel', modulus_elasticity=200e9)
    s.add_rectangular_section('rect', 0.3, 0.5)
    s.add_joint('N1', x=0)
    s.add_joint('N2', x=6)
    s.add_frame('F1', 'N1', 'N2', 'steel', 'rect')
    return s