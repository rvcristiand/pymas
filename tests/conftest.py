"""
Pytest configuration and fixtures for pymas tests.

This module provides reusable fixtures for testing the pymas structural
analysis library. Each fixture creates a typical structure scenario for testing
different functionalities.

Fixtures:
    structure: Empty space frame structure ready to be populated.
    simple_truss_structure: Simple space truss structure (N1 ---- N2).
    simple_frame_structure: Simple space frame structure (N1 ---- N2).
    simple_beam_structure: Simple beam structure for beam-type analysis.
    structure_with_loads: A multi-span beam with loads applied.
    structure_and_load: A minimal beam for frame point load testing.

Typical usage:
    def test_example(simple_frame_structure):
        # simple_frame_structure already has materials, sections, joints,
        # frames and supports
        simple_frame_structure.add_load_pattern('dead')
        simple_frame_structure.run_analysis()
        assert 'dead' in simple_frame_structure.displacements
"""

import pytest
from pymas import Structure


@pytest.fixture
def structure():
    """Create an empty space frame Structure object.

    Returns:
        Structure: An empty structure with 6 DoF per joint, ready to be
            populated with materials, sections, joints, frames, supports, and
            loads.
    """
    return Structure('space_frame')


@pytest.fixture
def simple_truss_structure():
    """Create a simple space truss structure.

    A single axial-only frame connecting two joints with proper supports.
    This fixture is for testing truss-like frames.

    Structure:
        N1(0,0,0) ---- Frame T1 ---- N2(5,0,0)

    Contains:
        - Material 'steel': E=200e9 (no shear needed for a truss)
        - Section 'circle': A=0.01, J=0.0001, Iy=0.0001, Iz=0.0001
        - Joint N1 at origin (0,0,0)
        - Joint N2 at (5,0,0)
        - Frame T1 connecting N1-N2, axial only
        - Support at N1: fully restrained
        - Support at N2: restrained in uy and uz

    Returns:
        Structure: A simple truss structure for testing.
    """
    s = Structure('space_truss')

    s.add_material("steel", E=200e9)
    s.add_section("circle", A=0.01, J=0.0001, Iy=0.0001, Iz=0.0001)
    s.add_joint("N1", x=0, y=0, z=0)
    s.add_joint("N2", x=5, y=0, z=0)
    s.add_frame("T1", "N1", "N2", "steel", "circle", axial=True)
    s.add_support("N1", r_ux=True, r_uy=True, r_uz=True)
    s.add_support("N2", r_uy=True, r_uz=True)

    return s


@pytest.fixture
def simple_frame_structure():
    """Create a simple space frame structure.

    A single frame connecting two joints with proper supports.
    This fixture is for testing Frame-specific functionality.

    Structure:
        N1(0,0,0) ---- Frame F1 ---- N2(5,0,0)

    Contains:
        - Material 'steel': E=200e9, G=77e9
        - RectangularSection 'rect': base=0.3, height=0.5
        - Joint N1 at origin (0,0,0)
        - Joint N2 at (5,0,0)
        - Frame F1 connecting N1-N2 (axial, torsion and both bendings)
        - Support at N1: fully restrained (all 6 DoF)
        - Support at N2: partially restrained (uy, rz)

    Returns:
        Structure: A simple frame structure for testing.
    """
    s = Structure('space_frame')

    s.add_material("steel", E=200e9, G=77e9)
    s.add_rectangular_section("rect", 0.3, 0.5)
    s.add_joint("N1", x=0, y=0, z=0)
    s.add_joint("N2", x=5, y=0, z=0)
    s.add_frame("F1", "N1", "N2", "steel", "rect", axial=True,
                torsional=True, bending_y=True, bending_z=True)
    s.add_support(
        "N1", r_ux=True, r_uy=True, r_uz=True, r_rx=True, r_ry=True, r_rz=True
    )
    s.add_support("N2", r_uy=True, r_rz=True)

    return s


@pytest.fixture
def simple_beam_structure():
    """Create a simple beam-type structure.

    A single frame with beam-type DoF (2 per joint: uy, rz).
    This fixture is for testing beam-specific functionality.

    Structure:
        N1(0) ---- Frame F1 ---- N2(6)

    Contains:
        - Material 'steel': E=200e9
        - RectangularSection 'rect': base=0.3, height=0.5
        - Joints N1(0), N2(6)
        - Frame F1 connecting N1-N2 (bending about local z)
        - Support at N1: uy restrained
        - Support at N2: uy restrained

    Returns:
        Structure: A simple beam structure for testing.
    """
    s = Structure('beam')

    s.add_material("steel", E=200e9)
    s.add_rectangular_section("rect", 0.3, 0.5)
    s.add_joint("N1", x=0)
    s.add_joint("N2", x=6)
    s.add_frame("F1", "N1", "N2", "steel", "rect", bending_z=True)
    s.add_support("N1", r_uy=True)
    s.add_support("N2", r_uy=True)

    return s


@pytest.fixture
def structure_with_loads():
    """Create a multi-span beam with loads applied.

    A two-span beam with distributed loads on both spans.

    Structure:
        N1 ---- F1 ---- N2 ---- F2 ---- N3
        ^               ^               ^

    Contains:
        - Material 'steel': E=200e9
        - RectangularSection 'rect': base=0.3, height=0.5
        - Joints N1(0), N2(6), N3(12)
        - Frames F1 (N1-N2) and F2 (N2-N3), bending about local z
        - Supports at N1, N2 and N3: uy restrained
        - LoadPattern 'dead' with distributed loads

    Returns:
        Structure: A multi-span beam with loads.
    """
    s = Structure('beam')

    s.add_material("steel", E=200e9)
    s.add_rectangular_section("rect", 0.3, 0.5)
    s.add_joint("N1", x=0)
    s.add_joint("N2", x=6)
    s.add_joint("N3", x=12)
    s.add_frame("F1", "N1", "N2", "steel", "rect", bending_z=True)
    s.add_frame("F2", "N2", "N3", "steel", "rect", bending_z=True)
    s.add_support("N1", r_uy=True)
    s.add_support("N2", r_uy=True)
    s.add_support("N3", r_uy=True)
    s.add_load_pattern("dead")
    s.add_distributed_load("dead", "F1", fy=-5e3)
    s.add_distributed_load("dead", "F2", fy=-5e3)

    return s


@pytest.fixture
def structure_and_load():
    """Create a minimal beam for frame point load testing.

    A single frame without loads, ready to receive point loads.

    Contains:
        - Material 'steel': E=200e9
        - RectangularSection 'rect': base=0.3, height=0.5
        - Joints N1(0), N2(6)
        - Frame F1 connecting N1-N2, bending about local z

    Returns:
        Structure: Minimal beam without loads applied yet.
    """
    s = Structure('beam')
    s.add_material("steel", E=200e9)
    s.add_rectangular_section("rect", 0.3, 0.5)
    s.add_joint("N1", x=0)
    s.add_joint("N2", x=6)
    s.add_frame("F1", "N1", "N2", "steel", "rect", bending_z=True)
    return s
