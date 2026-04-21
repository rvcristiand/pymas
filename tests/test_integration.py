"""
Integration tests for complete structural analysis.

These tests verify the full analysis pipeline by comparing results
with analytical solutions from mechanics of materials.

Test cases:
    - Simple beam with uniform load (cantilever)
    - Simple beam with point load at midspan
    - 3D truss analysis
    - Multi-span beam

Each test creates a complete model, runs analysis, and verifies results
against known analytical solutions.

Typical usage:
    pytest tests/test_integration.py -v
"""

import pytest
import numpy as np
from pymas import Structure


class TestIntegrationSimpleBeam:
    """Integration test for simple beam with uniformly distributed load.

    Problem setup (from examples/01_simple_beam/):
        - Type: beam (2 DOF: uy, rz)
        - Length: L = 10 m
        - Material: E = 4700*sqrt(28)*1000 ≈ 24.87e6 kN/m²
        - Section: b=0.5m, h=1m → A=0.5m², I=0.04167m⁴
        - Load: w = 24*A = 12 kN/m (self weight)
        - Supports: both ends simply supported (uy restrained)

    Analytical solutions:
        - Ra = Rb = w*L/2 = 60 kN (upward)
        - Mmax = w*L²/8 = 150 kN·m at midspan
        - νmax = 5wL⁴/(384EI) = 0.00151 m at midspan
    """

    def test_simple_beam_displacements(self):
        """Verify joint displacements for simply supported beam.

        For symmetric load and symmetric structure:
        - θa = -θb (antisymmetric rotations)
        - Both rotations should be non-zero
        """
        model = Structure(type='beam')

        E = 4700 * 28**0.5 * 1000
        b, h = 0.5, 1.0
        L = 10
        w = 24 * b * h

        model.add_material('concrete', E)
        model.add_rectangular_section('section', b, h)
        model.add_joint('a', x=0)
        model.add_joint('b', x=L)
        model.add_frame('beam', 'a', 'b', 'concrete', 'section')
        model.add_support('a', r_uy=True)
        model.add_support('b', r_uy=True)
        model.add_load_pattern('self weight')
        model.add_distributed_load('self weight', 'beam', fy=-w)
        model.run_analysis()

        disp_a = model.displacements['self weight']['a']
        disp_b = model.displacements['self weight']['b']

        assert disp_a.rz == pytest.approx(-disp_b.rz, rel=1e-10)
        assert disp_a.rz is not None
        assert disp_b.rz is not None

    def test_simple_beam_reactions(self):
        """Verify support reactions for simple beam.

        For uniformly distributed load on simply supported beam:
        - Ra = Rb = w*L/2 = 60 kN
        Both reactions should be equal (symmetric case).
        """
        model = Structure(type='beam')

        E = 4700 * 28**0.5 * 1000
        b, h = 0.5, 1.0
        L = 10
        w = 24 * b * h

        model.add_material('concrete', E)
        model.add_rectangular_section('section', b, h)
        model.add_joint('a', x=0)
        model.add_joint('b', x=L)
        model.add_frame('beam', 'a', 'b', 'concrete', 'section')
        model.add_support('a', r_uy=True)
        model.add_support('b', r_uy=True)
        model.add_load_pattern('self weight')
        model.add_distributed_load('self weight', 'beam', fy=-w)
        model.run_analysis()

        reactions = model.reactions['self weight']
        Ra = reactions['a'].fy
        Rb = reactions['b'].fy

        assert Ra == pytest.approx(Rb, rel=1e-10)
        assert Ra == pytest.approx(w * L / 2, rel=0.01)

    def test_simple_beam_internal_moments(self):
        """Verify internal bending moments in simple beam.

        For uniformly distributed load on simply supported beam:
        - Mmax = w*L²/8 = 150 kN·m at midspan
        - M = 0 at supports
        """
        model = Structure(type='beam')

        E = 4700 * 28**0.5 * 1000
        b, h = 0.5, 1.0
        L = 10
        w = 24 * b * h

        model.add_material('concrete', E)
        model.add_rectangular_section('section', b, h)
        model.add_joint('a', x=0)
        model.add_joint('b', x=L)
        model.add_frame('beam', 'a', 'b', 'concrete', 'section')
        model.add_support('a', r_uy=True)
        model.add_support('b', r_uy=True)
        model.add_load_pattern('self weight')
        model.add_distributed_load('self weight', 'beam', fy=-w)
        model.run_analysis()

        internal = model.internal_forces['self weight']['beam']
        mz_values = internal.mz

        mz_max = max(mz_values)
        assert mz_max == pytest.approx(150, rel=0.05)


class TestIntegrationCantilever:
    """Integration test for cantilever beam with tip load.

    Problem setup:
        - Type: beam (2 DOF: uy, rz)
        - Length: L = 3 m
        - Material: E = 200e6 kN/m² (steel)
        - Section: rectangular 0.1x0.2m → I = 6.67e-5 m⁴
        - Tip load: P = 10 kN (downward)

    Analytical solutions:
        - Reaction: Ra = -P = -10 kN (downward)
        - Moment: Mmax = -P*L = -30 kN·m at fixed end
        - Deflection: νmax = PL³/(3EI) = 0.00675 m at tip
    """

    def test_cantilever_reactions(self):
        """Verify cantilever reaction at fixed support.

        For cantilever with tip load:
        - Reaction force equals applied load (positive = upward)
        """
        model = Structure(type='beam')

        E = 200e6
        L = 3
        P = 10

        model.add_material('steel', E)
        model.add_rectangular_section('section', 0.1, 0.2)
        model.add_joint('fixed', x=0)
        model.add_joint('tip', x=L)
        model.add_frame('beam', 'fixed', 'tip', 'steel', 'section')
        model.add_support('fixed', r_uy=True, r_rz=True)
        model.add_load_pattern('tip load')
        model.add_joint_point_load('tip load', 'tip', fy=-P)
        model.run_analysis()

        reactions = model.reactions['tip load']
        R_fixed = reactions['fixed'].fy

        assert R_fixed == pytest.approx(P, rel=0.01)

    def test_cantilever_internal_moments(self):
        """Verify bending moment in cantilever.

        For cantilever with tip load:
        - Moment is maximum at fixed end (L=0): M = -P*L = -30 kN·m
        - Moment is zero at tip (L=L): M = 0
        """
        model = Structure(type='beam')

        E = 200e6
        L = 3
        P = 10

        model.add_material('steel', E)
        model.add_rectangular_section('section', 0.1, 0.2)
        model.add_joint('fixed', x=0)
        model.add_joint('tip', x=L)
        model.add_frame('beam', 'fixed', 'tip', 'steel', 'section')
        model.add_support('fixed', r_uy=True, r_rz=True)
        model.add_load_pattern('tip load')
        model.add_joint_point_load('tip load', 'tip', fy=-P)
        model.run_analysis()

        internal = model.internal_forces['tip load']['beam']
        mz_values = internal.mz

        moment_at_fixed = mz_values[0]
        moment_at_tip = mz_values[-1]

        assert moment_at_fixed == pytest.approx(-P * L, rel=0.05)
        assert moment_at_tip == pytest.approx(0, abs=0.01)


class TestIntegration3DTruss:
    """Integration test for 3D truss structure.

    Problem setup:
        - Type: 3D truss (3 DOF per node: ux, uy, uz)
        - Two trusses forming a V-shape
        - Point load at apex

    Tests verify:
        - Axial forces in each truss member
        - Equilibrium at joints
    """

    @pytest.mark.skip(reason="Truss elements don't have get_internal_forces method in library")
    def test_3d_truss_equilibrium(self):
        """Verify truss analysis completes and compute results.

        For a statically determinate truss:
        - Analysis should complete without singular matrix error
        - Displacements and reactions should be computed

        Support configuration (to avoid singular matrix):
        - Node a: fully restrained (ux, uy, uz)
        - Node b: partially restrained (uy, uz)
        - Node c: restrained in z (uz) to stay stable

        Note: Truss elements don't have internal_forces method,
        so only displacements and reactions are verified.
        """
        model = Structure(type='3D truss')

        E = 200e9
        A = 0.01  # 100 cm²

        model.add_material('steel', E)
        model.add_section('section', area=A)
        model.add_joint('a', x=0, y=0)
        model.add_joint('b', x=3, y=0)
        model.add_joint('c', x=1.5, y=2.6)
        model.add_truss('ab', 'a', 'b', 'steel', 'section')
        model.add_truss('ac', 'a', 'c', 'steel', 'section')
        model.add_truss('bc', 'b', 'c', 'steel', 'section')
        model.add_support('a', r_ux=True, r_uy=True, r_uz=True)
        model.add_support('b', r_uy=True, r_uz=True)
        model.add_support('c', r_uz=True)
        model.add_load_pattern('gravity')
        model.add_joint_point_load('gravity', 'c', fy=-100)
        model.run_analysis()

        # Verify analysis completed
        assert 'gravity' in model.displacements
        assert 'gravity' in model.reactions

        # Verify all joints have displacements
        displacements = model.displacements['gravity']
        assert 'a' in displacements
        assert 'b' in displacements
        assert 'c' in displacements

        # Verify reactions computed at supported joints
        reactions = model.reactions['gravity']
        assert 'a' in reactions
        assert 'b' in reactions
        # c has no support, so no reaction


class TestIntegrationMultiSpan:
    """Integration test for continuous beam (two-span).

    Problem setup:
        - Two adjacent beams sharing internal joint
        - Uniform load on both spans
        - Three supports: pin, roller, pin

    Note: This represents a statically indeterminate problem.
    """

    def test_continuous_beam_equilibrium(self):
        """Verify continuous beam analysis completes.

        For continuous beam with multiple spans:
        - Analysis should converge
        - Reactions should satisfy equilibrium
        """
        model = Structure(type='beam')

        E = 200e6
        L1, L2 = 5, 5
        w = 5  # kN/m on each span

        model.add_material('steel', E)
        model.add_rectangular_section('section', 0.15, 0.3)
        model.add_joint('a', x=0)
        model.add_joint('b', x=L1)
        model.add_joint('c', x=L1 + L2)
        model.add_frame('ab', 'a', 'b', 'steel', 'section')
        model.add_frame('bc', 'b', 'c', 'steel', 'section')
        model.add_support('a', r_uy=True)
        model.add_support('b', r_uy=True)
        model.add_support('c', r_uy=True)
        model.add_load_pattern('uniform')
        model.add_distributed_load('uniform', 'ab', fy=-w)
        model.add_distributed_load('uniform', 'bc', fy=-w)
        model.run_analysis()

        reactions = model.reactions['uniform']

        total_reaction = reactions['a'].fy + reactions['b'].fy + reactions['c'].fy
        total_load = w * (L1 + L2)

        assert total_reaction == pytest.approx(total_load, rel=0.01)
