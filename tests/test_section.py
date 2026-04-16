"""
Tests for Section and RectangularSection classes.

The Section class defines geometric properties of a cross-section:
    - A: Area
    - J: Torsion constant
    - Iy, Iz: Moments of inertia about local axes

The RectangularSection class automatically calculates these properties
from base and height dimensions using standard formulas.

Typical usage:
    structure = Structure()
    section = structure.add_section('rect', area=0.01, inertia_y=0.0001)
    rect = structure.add_rectangular_section('beam', 0.3, 0.5)
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
        section = structure.add_section('rect', area=0.01, torsion_constant=0.0001,
                                        inertia_y=0.0001, inertia_z=0.0001)
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
        - Area: A = base * height
        - Inertia y: Iy = (1/12) * height * base^3
        - Inertia z: Iz = (1/12) * base * height^3
        - Torsion: J per Bredt formula
    """

    def test_add_rectangular_section(self, structure):
        """Verify that a rectangular section can be created with base and height.

        Creates a rectangular section with base=0.3 and height=0.5.
        """
        section = structure.add_rectangular_section('rect', 0.3, 0.5)
        assert section.name == 'rect'
        assert section.base == 0.3
        assert section.height == 0.5

    def test_rectangular_section_area(self, structure):
        """Verify that section area is calculated correctly.

        Area = base * height = 0.3 * 0.5 = 0.15
        """
        section = structure.add_rectangular_section('rect', 0.3, 0.5)
        assert section.A == pytest.approx(0.3 * 0.5, rel=1e-10)

    def test_rectangular_section_inertia_y(self, structure):
        """Verify that moment of inertia about y-axis is calculated.

        Iy = (1/12) * height * base^3
           = (1/12) * 0.5 * 0.3^3
        """
        section = structure.add_rectangular_section('rect', 0.3, 0.5)
        assert section.Iy == pytest.approx((1/12) * 0.5 * 0.3**3, rel=1e-10)

    def test_rectangular_section_inertia_z(self, structure):
        """Verify that moment of inertia about z-axis is calculated.

        Iz = (1/12) * base * height^3
           = (1/12) * 0.3 * 0.5^3
        """
        section = structure.add_rectangular_section('rect', 0.3, 0.5)
        assert section.Iz == pytest.approx((1/12) * 0.3 * 0.5**3, rel=1e-10)

    def test_rectangular_section_torsion_constant(self, structure):
        """Verify that torsion constant is calculated using Bredt formula.

        J = (1/3 - 0.21*(a/b)*(1 - (1/12)*(a/b)^4)) * b * a^3
        where a = min(base, height), b = max(base, height)
        """
        section = structure.add_rectangular_section('rect', 0.3, 0.5)
        a, b = sorted((0.3, 0.5))
        expected_J = (1/3 - 0.21 * (a/b) * (1 - (1/12) * (a/b)**4)) * b * a**3
        assert section.J == pytest.approx(expected_J, rel=1e-10)