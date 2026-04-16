"""
Tests for the Material class.

The Material class represents a linear elastic material with elastic modulus (E)
and shear modulus (G) properties. It is used to define the material properties
of structural elements like trusses and frames.

Typical usage:
    structure = Structure()
    material = structure.add_material('steel', modulus_elasticity=200e9)
    print(material.E)  # 200000000000.0
"""

import pytest


class TestMaterial:
    """Tests for Material class functionality.

    The Material class stores:
        - name: Material identifier
        - E: Modulus of elasticity (Young's modulus)
        - G: Shear modulus (modulus of rigidity)
    """

    def test_add_material(self, structure):
        """Verify that a material can be added to the structure with correct properties.

        Creates a material 'steel' with E=200e9 and checks that:
        - The material name is correct
        - The elastic modulus is stored
        - The material is added to the structure's materials dictionary
        """
        material = structure.add_material('steel', modulus_elasticity=200e9)
        assert material.name == 'steel'
        assert material.E == 200e9
        assert 'steel' in structure.materials

    def test_add_material_with_shear_modulus(self, structure):
        """Verify that a material can be created with both elastic and shear modulus.

        Creates a material with both E and G to ensure both properties
        are properly stored.
        """
        material = structure.add_material('steel', modulus_elasticity=200e9, modulus_elasticity_shear=77e9)
        assert material.G == 77e9

    def test_material_default_values(self, structure):
        """Verify that material properties default to None when not provided.

        When no modulus values are specified, the material should have
        None values for both E and G.
        """
        material = structure.add_material('steel')
        assert material.E is None
        assert material.G is None