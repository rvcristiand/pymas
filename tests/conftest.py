"""
Pytest configuration and fixtures for pymas tests.

This module provides reusable fixtures for testing the pymas structural analysis library.
Each fixture creates a typical structure scenario for testing different functionalities.

Fixtures:
    structure: Empty structure ready to be populated.
    simple_truss_structure: Simple 3D truss structure (N1 ---- N2).
    simple_frame_structure: Simple 3D frame structure (N1 ---- N2).
    simple_beam_structure: Simple beam structure for beam-type analysis.
    structure_with_loads: A multi-span structure with loads applied.
    structure_and_load: A minimal structure for element point load testing.

Typical usage:
    def test_example(simple_truss_structure):
        # simple_truss_structure already has materials, sections, joints, elements, supports
        simple_truss_structure.run_analysis()
        assert 'dead' in simple_truss_structure.displacements
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
def simple_truss_structure():
    """Create a simple 3D truss structure.

    A single truss element connecting two joints with proper supports.
    This fixture is for testing Truss-specific functionality.

    Structure:
        N1(0,0,0) ---- Truss T1 ---- N2(5,0,0)

    Contains:
        - Material 'steel': E=200e9 GPa (no shear needed for truss)
        - Section 'circle': A=0.01, J=0.0001, Iy=0.0001, Iz=0.0001
        - Joint N1 at origin (0,0,0)
        - Joint N2 at (5,0,0)
        - Truss T1 connecting N1-N2
        - Support at N1: fully restrained (all 6 DOF)
        - Support at N2: partially restrained (uy, rz - for stability)

    Returns:
        Structure: A simple truss structure for testing.
    """
    s = Structure()

    s.add_material('steel', modulus_elasticity=200e9)
    s.add_section('circle', area=0.01, torsion_constant=0.0001,
                 inertia_y=0.0001, inertia_z=0.0001)
    s.add_joint('N1', x=0, y=0, z=0)
    s.add_joint('N2', x=5, y=0, z=0)
    s.add_truss('T1', 'N1', 'N2', 'steel', 'circle')
    s.add_support('N1', r_ux=True, r_uy=True, r_uz=True,
                 r_rx=True, r_ry=True, r_rz=True)
    s.add_support('N2', r_uy=True, r_rz=True)

    return s


@pytest.fixture
def simple_frame_structure():
    """Create a simple 3D frame structure.

    A single frame element connecting two joints with proper supports.
    This fixture is for testing Frame-specific functionality.

    Structure:
        N1(0,0,0) ---- Frame F1 ---- N2(5,0,0)

    Contains:
        - Material 'steel': E=200e9 GPa, G=77e9 GPa
        - RectangularSection 'rect': base=0.3, height=0.5
        - Joint N1 at origin (0,0,0)
        - Joint N2 at (5,0,0)
        - Frame F1 connecting N1-N2
        - Support at N1: fully restrained (all 6 DOF)
        - Support at N2: partially restrained (uy, rz)

    Returns:
        Structure: A simple frame structure for testing.
    """
    s = Structure()

    s.add_material('steel', modulus_elasticity=200e9, modulus_elasticity_shear=77e9)
    s.add_rectangular_section('rect', 0.3, 0.5)
    s.add_joint('N1', x=0, y=0, z=0)
    s.add_joint('N2', x=5, y=0, z=0)
    s.add_frame('F1', 'N1', 'N2', 'steel', 'rect')
    s.add_support('N1', r_ux=True, r_uy=True, r_uz=True,
                 r_rx=True, r_ry=True, r_rz=True)
    s.add_support('N2', r_uy=True, r_rz=True)

    return s


@pytest.fixture
def simple_beam_structure():
    """Create a simple beam-type structure.

    A single frame element with beam-type DOFs (2 per joint: uy, rz).
    This fixture is for testing beam-specific functionality.

    Structure:
        N1(0) ---- Frame F1 ---- N2(6)

    Contains:
        - Material 'steel': E=200e9 GPa
        - RectangularSection 'rect': base=0.3, height=0.5
        - Joints N1(0), N2(6)
        - Frame F1 connecting N1-N2
        - Support at N1: uy restrained
        - Support at N2: uy restrained

    Returns:
        Structure: A simple beam structure for testing.
    """
    s = Structure(type='beam')

    s.add_material('steel', modulus_elasticity=200e9)
    s.add_rectangular_section('rect', 0.3, 0.5)
    s.add_joint('N1', x=0)
    s.add_joint('N2', x=6)
    s.add_frame('F1', 'N1', 'N2', 'steel', 'rect')
    s.add_support('N1', r_uy=True)
    s.add_support('N2', r_uy=True)

    return s


@pytest.fixture
def structure_with_loads():
    """Create a multi-span structure with loads applied.

    A two-span beam with distributed loads on both spans.
    Properly configured to avoid singular matrix issues.

    Structure:
        N1 ---- F1 ---- N2 ---- F2 ---- N3
        |________________|___________|

    Contains:
        - Material 'steel': E=200e9
        - RectangularSection 'rect': base=0.3, height=0.5
        - Joints N1(0), N2(6), N3(12)
        - Frames F1 (N1-N2) and F2 (N2-N3)
        - Support at N1: uy restrained ( pinned)
        - Support at N2: uy restrained (roller - for stability)
        - Support at N3: uy restrained (pinned)
        - LoadPattern 'dead' with distributed loads

    Returns:
        Structure: A properly configured multi-span structure.
    """
    s = Structure(type='beam')

    s.add_material('steel', modulus_elasticity=200e9)
    s.add_rectangular_section('rect', 0.3, 0.5)
    s.add_joint('N1', x=0)
    s.add_joint('N2', x=6)
    s.add_joint('N3', x=12)
    s.add_frame('F1', 'N1', 'N2', 'steel', 'rect')
    s.add_frame('F2', 'N2', 'N3', 'steel', 'rect')
    s.add_support('N1', r_uy=True)
    s.add_support('N2', r_uy=True)
    s.add_support('N3', r_uy=True)
    s.add_load_pattern('dead')
    s.add_distributed_load('dead', 'F1', fy=-5e3)
    s.add_distributed_load('dead', 'F2', fy=-5e3)

    return s


@pytest.fixture
def structure_and_load():
    """Create a minimal structure for element point load testing.

    A single frame without loads, ready to receive point loads.

    Contains:
        - Material 'steel': E=200e9
        - RectangularSection 'rect': base=0.3, height=0.5
        - Joints N1(0), N2(6)
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


# DEPRECATED: Keep for backward compatibility but warn
@pytest.fixture
def simple_structure():
    """DEPRECATED: Use simple_truss_structure or simple_frame_structure instead.

    This fixture is kept for backward compatibility with existing tests.
    It contains both a truss and frame element, which is physically unusual.
    """
    import warnings
    warnings.warn(
        "simple_structure fixture is deprecated. "
        "Use simple_truss_structure or simple_frame_structure instead.",
        DeprecationWarning,
        stacklevel=2
    )

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