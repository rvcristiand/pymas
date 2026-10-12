"""
Integration tests for complete structural analysis.

These tests verify the full analysis pipeline by comparing results
with analytical solutions from mechanics of materials.

Test cases:
    - Simple beam with uniform load (cantilever)
    - Simple beam with point load at midspan
    - Plane truss analysis
    - Multi-span beam

Each test creates a complete model, runs analysis, and verifies results
against known analytical solutions.

Typical usage:
    pytest tests/test_integration.py -v
"""

import pytest
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
        model = Structure('beam')

        E = 4700 * 28**0.5 * 1000
        b, h = 0.5, 1.0
        L = 10
        w = 24 * b * h

        model.add_material('concrete', E)
        model.add_rectangular_section('section', dy=h, dz=b)
        model.add_joint('a', x=0)
        model.add_joint('b', x=L)
        model.add_frame('beam', 'a', 'b', 'concrete', 'section', bending_z=True)
        model.add_support('a', r_ux=True, r_uy=True)
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
        model = Structure('beam')

        E = 4700 * 28**0.5 * 1000
        b, h = 0.5, 1.0
        L = 10
        w = 24 * b * h

        model.add_material('concrete', E)
        model.add_rectangular_section('section', dy=h, dz=b)
        model.add_joint('a', x=0)
        model.add_joint('b', x=L)
        model.add_frame('beam', 'a', 'b', 'concrete', 'section', bending_z=True)
        model.add_support('a', r_ux=True, r_uy=True)
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
        model = Structure('beam')

        E = 4700 * 28**0.5 * 1000
        b, h = 0.5, 1.0
        L = 10
        w = 24 * b * h

        model.add_material('concrete', E)
        model.add_rectangular_section('section', dy=h, dz=b)
        model.add_joint('a', x=0)
        model.add_joint('b', x=L)
        model.add_frame('beam', 'a', 'b', 'concrete', 'section', bending_z=True)
        model.add_support('a', r_ux=True, r_uy=True)
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
        model = Structure('beam')

        E = 200e6
        L = 3
        P = 10

        model.add_material('steel', E)
        model.add_rectangular_section('section', dy=0.2, dz=0.1)
        model.add_joint('fixed', x=0)
        model.add_joint('tip', x=L)
        model.add_frame('beam', 'fixed', 'tip', 'steel', 'section', bending_z=True)
        model.add_support('fixed', r_ux=True, r_uy=True, r_rz=True)
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
        model = Structure('beam')

        E = 200e6
        L = 3
        P = 10

        model.add_material('steel', E)
        model.add_rectangular_section('section', dy=0.2, dz=0.1)
        model.add_joint('fixed', x=0)
        model.add_joint('tip', x=L)
        model.add_frame('beam', 'fixed', 'tip', 'steel', 'section', bending_z=True)
        model.add_support('fixed', r_ux=True, r_uy=True, r_rz=True)
        model.add_load_pattern('tip load')
        model.add_joint_point_load('tip load', 'tip', fy=-P)
        model.run_analysis()

        internal = model.internal_forces['tip load']['beam']
        mz_values = internal.mz

        moment_at_fixed = mz_values[0]
        moment_at_tip = mz_values[-1]

        assert moment_at_fixed == pytest.approx(-P * L, rel=0.05)
        assert moment_at_tip == pytest.approx(0, abs=0.01)


class TestIntegrationPlaneTruss:
    """Integration test for plane truss structure.

    Problem setup:
        - Type: plane truss (2 DOF per node: ux, uy)
        - Three axial-only frames forming a triangle
        - Point load at apex

    Tests verify:
        - Displacements and reactions are computed
        - Global equilibrium of the reactions
    """

    def test_plane_truss_equilibrium(self):
        """Verify truss analysis completes and reactions balance the load.

        For a statically determinate truss:
        - Analysis should complete without singular matrix error
        - Displacements and reactions should be computed
        - Reactions should balance the applied load

        Support configuration (r_uz is inactive in a plane truss):
        - Node a: pinned (ux, uy)
        - Node b: roller (uy)
        - Node c: only r_uz, so effectively free
        """
        model = Structure('plane_truss')

        E = 200e9
        A = 0.01  # 100 cm²

        model.add_material('steel', E)
        model.add_section('section', A=A)
        model.add_joint('a', x=0, y=0)
        model.add_joint('b', x=3, y=0)
        model.add_joint('c', x=1.5, y=2.6)
        model.add_frame('ab', 'a', 'b', 'steel', 'section', axial=True)
        model.add_frame('ac', 'a', 'c', 'steel', 'section', axial=True)
        model.add_frame('bc', 'b', 'c', 'steel', 'section', axial=True)
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

        # global equilibrium
        assert reactions['a'].fx == pytest.approx(0, abs=1e-6)
        assert reactions['a'].fy + reactions['b'].fy == pytest.approx(100)


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
        model = Structure('beam')

        E = 200e6
        L1, L2 = 5, 5
        w = 5  # kN/m on each span

        model.add_material('steel', E)
        model.add_rectangular_section('section', dy=0.3, dz=0.15)
        model.add_joint('a', x=0)
        model.add_joint('b', x=L1)
        model.add_joint('c', x=L1 + L2)
        model.add_frame('ab', 'a', 'b', 'steel', 'section', bending_z=True)
        model.add_frame('bc', 'b', 'c', 'steel', 'section', bending_z=True)
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


class TestIntegrationFramePointLoad:
    """Integration tests for frame point loads.

    Problem setup:
        - Simple beam with a point load at midspan (analytical solution)
        - Space frame cantilever with a point load of six components, compared
          with the same model split at the load into two frames carrying an
          equivalent joint load

    Analytical solutions (simple beam):
        - Ra = Rb = P/2
        - Mmax = P*L/4 at midspan
        - vmax = P*L^3/(48*E*I) at midspan
    """

    def test_simple_beam_midspan_point_load(self):
        """Verify reactions, moment and deflection of a simple beam."""
        model = Structure('beam')

        E = 200e6
        b, h = 0.2, 0.4
        I = b * h**3 / 12
        L = 6
        P = 10

        model.add_material('steel', E)
        model.add_rectangular_section('section', dy=h, dz=b)
        model.add_joint('a', x=0)
        model.add_joint('b', x=L)
        model.add_frame('beam', 'a', 'b', 'steel', 'section', bending_z=True)
        model.add_support('a', r_uy=True)
        model.add_support('b', r_uy=True)
        model.add_load_pattern('point')
        model.add_frame_point_load('point', 'beam', L / 2, fy=-P)
        model.run_analysis()

        reactions = model.reactions['point']
        assert reactions['a'].fy == pytest.approx(P / 2)
        assert reactions['b'].fy == pytest.approx(P / 2)

        mz = model.internal_forces['point']['beam'].mz
        assert max(mz) == pytest.approx(P * L / 4)
        assert mz[0] == pytest.approx(0, abs=1e-9)
        assert mz[-1] == pytest.approx(0, abs=1e-9)

        uy = model.internal_displacements['point']['beam'].uy
        assert min(uy) == pytest.approx(-P * L**3 / (48 * E * I))
        assert uy[-1] == pytest.approx(0, abs=1e-12)

    @staticmethod
    def _cantilever(split, direction, dist, load):
        """Build and analyse a space frame cantilever along `direction`.

        Args:
            split (bool): Whether to split the frame at the load, applying the
                load to the new joint instead of to the frame.
            direction (tuple): Vector from the fixed to the free end.
            dist (float): Distance of the load from the fixed end.
            load (dict): Local components of the load (fx, ..., mz).

        Returns:
            Structure: Analysed model.
        """
        import numpy as np

        model = Structure('space_frame')
        model.add_material('steel', E=200e6, G=80e6)
        model.add_section('section', A=0.01, J=1e-5, Iy=2e-5, Iz=8e-5)

        u = np.array(direction, dtype=float)
        L = np.linalg.norm(u)
        u /= L
        flags = dict(axial=True, torsional=True, bending_y=True,
                     bending_z=True)

        model.add_joint('a', *(0 * u))
        model.add_joint('b', *(L * u))
        model.add_support('a', r_ux=True, r_uy=True, r_uz=True, r_rx=True,
                          r_ry=True, r_rz=True)
        model.add_load_pattern('point')

        if not split:
            model.add_frame('ab', 'a', 'b', 'steel', 'section', **flags)
            model.add_frame_point_load('point', 'ab', dist, **load)
        else:
            model.add_joint('m', *(dist * u))
            model.add_frame('am', 'a', 'm', 'steel', 'section', **flags)
            model.add_frame('mb', 'm', 'b', 'steel', 'section', **flags)

            # local load components to global
            r = model.frames['am'].rotation_matrix()
            f = r @ [load.get(k, 0) for k in ('fx', 'fy', 'fz')]
            m = r @ [load.get(k, 0) for k in ('mx', 'my', 'mz')]
            model.add_joint_point_load('point', 'm', *f, *m)

        model.run_analysis()

        return model

    @pytest.mark.parametrize('direction', [(5, 0, 0), (3, 4, 5)])
    @pytest.mark.parametrize('load', [
        {'fx': 7}, {'fy': -10}, {'fz': 4}, {'mx': 3}, {'my': -6}, {'mz': 5},
        {'fx': 7, 'fy': -10, 'fz': 4, 'mx': 3, 'my': -6, 'mz': 5},
    ])
    def test_point_load_equals_split_frame(self, direction, load):
        """Verify a frame point load matches the equivalent split model.

        The reactions, the displacements of the free end, and the internal
        forces and displacements at both ends and at the load must agree.
        """
        import numpy as np

        L = np.linalg.norm(direction)
        dist = 0.3 * L
        whole = self._cantilever(False, direction, dist, load)
        split = self._cantilever(True, direction, dist, load)

        components = ('fx', 'fy', 'fz', 'mx', 'my', 'mz')
        for c in components:
            assert getattr(whole.reactions['point']['a'], c) == \
                pytest.approx(getattr(split.reactions['point']['a'], c),
                              abs=1e-9)

        for c in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz'):
            assert getattr(whole.displacements['point']['b'], c) == \
                pytest.approx(getattr(split.displacements['point']['b'], c),
                              abs=1e-12)

        # index 30 is the section at the load, on the side of the fixed end
        f_whole = whole.internal_forces['point']['ab']
        f_am = split.internal_forces['point']['am']
        f_mb = split.internal_forces['point']['mb']
        for c in components:
            values = getattr(f_whole, c)
            assert values[0] == pytest.approx(getattr(f_am, c)[0], abs=1e-9)
            assert values[30] == pytest.approx(getattr(f_am, c)[-1],
                                               abs=1e-9)
            assert values[-1] == pytest.approx(getattr(f_mb, c)[-1],
                                               abs=1e-9)

        d_whole = whole.internal_displacements['point']['ab']
        d_am = split.internal_displacements['point']['am']
        d_mb = split.internal_displacements['point']['mb']
        for c in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz'):
            values = getattr(d_whole, c)
            assert values[30] == pytest.approx(getattr(d_am, c)[-1],
                                               abs=1e-12)
            assert values[-1] == pytest.approx(getattr(d_mb, c)[-1],
                                               abs=1e-12)

        # internal displacements at the free end match the joint, in local
        # axes
        r = whole.frames['ab'].rotation_matrix()
        d_b = whole.displacements['point']['b'].displacement_vector()
        local = np.concatenate((r.T @ d_b[:3], r.T @ d_b[3:]))
        for value, c in zip(local, ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')):
            assert getattr(d_whole, c)[-1] == pytest.approx(value, abs=1e-12)
