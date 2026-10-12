"""
Tests for Section and RectangularSection classes.

The Section class defines geometric properties of a cross-section:
    - A: Area
    - J: Torsion constant
    - Iy, Iz: Moments of inertia about local axes

The RectangularSection class automatically calculates these properties
from its dimensions along the local y- and z-axes (dy, dz) using standard
formulas.

Typical usage:
    structure = Structure('space_frame')
    section = structure.add_section('rect', A=0.01, Iy=0.0001)
    rect = structure.add_rectangular_section('beam', dy=0.5, dz=0.3)
"""

import pytest


class TestSection:
    """Tests for generic Section class.

    A generic section with manually specified properties.
    """

    def test_add_section(self, structure):
        """Verify that a section can be added with specified properties.

        Creates a section with area, torsion constant, and moments of inertia.
        """
        section = structure.add_section('rect', A=0.01, J=0.0001,
                                        Iy=0.0001, Iz=0.0001)
        assert section.name == 'rect'
        assert section.A == 0.01
        assert section.J == 0.0001
        assert 'rect' in structure.sections

    def test_section_default_values(self, structure):
        """Verify that section properties default to None when not provided.

        When no properties are specified, all should be None.
        """
        section = structure.add_section('rect')
        assert section.A is None
        assert section.J is None
        assert section.Iy is None
        assert section.Iz is None


class TestRectangularSection:
    """Tests for RectangularSection class.

    Automatically calculates:
        - Area: A = dy * dz
        - Inertia y: Iy = (1/12) * dy * dz^3
        - Inertia z: Iz = (1/12) * dz * dy^3
        - Torsion: J per Bredt formula
    """

    def test_add_rectangular_section(self, structure):
        """Verify that a rectangular section can be created with dy and dz.

        Creates a rectangular section with dy=0.5 and dz=0.3.
        """
        section = structure.add_rectangular_section('rect', dy=0.5, dz=0.3)
        assert section.name == 'rect'
        assert section.dy == 0.5
        assert section.dz == 0.3

    def test_rectangular_section_area(self, structure):
        """Verify that section area is calculated correctly.

        Area = dy * dz = 0.5 * 0.3 = 0.15
        """
        section = structure.add_rectangular_section('rect', dy=0.5, dz=0.3)
        assert section.A == pytest.approx(0.3 * 0.5, rel=1e-10)

    def test_rectangular_section_inertia_y(self, structure):
        """Verify that moment of inertia about y-axis is calculated.

        Iy = (1/12) * dy * dz^3
           = (1/12) * 0.5 * 0.3^3
        """
        section = structure.add_rectangular_section('rect', dy=0.5, dz=0.3)
        assert section.Iy == pytest.approx((1/12) * 0.5 * 0.3**3, rel=1e-10)

    def test_rectangular_section_inertia_z(self, structure):
        """Verify that moment of inertia about z-axis is calculated.

        Iz = (1/12) * dz * dy^3
           = (1/12) * 0.3 * 0.5^3
        """
        section = structure.add_rectangular_section('rect', dy=0.5, dz=0.3)
        assert section.Iz == pytest.approx((1/12) * 0.3 * 0.5**3, rel=1e-10)

    def test_rectangular_section_torsion_constant(self, structure):
        """Verify that torsion constant is calculated using Bredt formula.

        J = (1/3 - 0.21*(a/b)*(1 - (1/12)*(a/b)^4)) * b * a^3
        where a = min(dy, dz), b = max(dy, dz)
        """
        section = structure.add_rectangular_section('rect', dy=0.5, dz=0.3)
        a, b = sorted((0.3, 0.5))
        expected_J = (1/3 - 0.21 * (a/b) * (1 - (1/12) * (a/b)**4)) * b * a**3
        assert section.J == pytest.approx(expected_J, rel=1e-10)
    def test_rectangular_section_positional_raises(self, structure):
        """Verify dy and dz must be given by name.

        An old positional call (name, base, height) must fail instead of
        silently swapping Iy and Iz.
        """
        with pytest.raises(TypeError):
            structure.add_rectangular_section('rect', 0.3, 0.5)
