"""
Tests for the Structure class.

The Structure class is the main class in pymas for modeling and analyzing
structural systems using the Direct Stiffness Method.

Structure types (the type is fixed when the Structure is created):
    - 'space_frame': 6 DoF per joint (ux, uy, uz, rx, ry, rz)
    - 'space_truss': 3 DoF per joint (ux, uy, uz)
    - 'beam': 2 DoF per joint (uy, rz)
    - any other name: DoF given by the keyword flags ux, uy, uz, rx, ry, rz

Key methods:
    - add_material(), add_section(), add_joint()
    - add_frame()
    - add_support(), add_load_pattern()
    - get_degrees_freedom(), set_joint_indices()
    - set_stiffness_matrix()
    - run_analysis()
    - export(filename)

Analysis process (run by run_analysis()):
    1. set_joint_indices(): Assign global indices to each DoF
    2. set_stiffness_matrix(): Assemble global stiffness matrix
    3. analyse_load_pattern(): Solve for each load pattern

Typical usage:
    structure = Structure('space_frame')
    structure.add_material('steel', E=200e9)
    structure.add_rectangular_section('rect', dy=0.5, dz=0.3)
    structure.add_joint('N1', x=0)
    structure.add_joint('N2', x=6)
    structure.add_frame('F1', 'N1', 'N2', 'steel', 'rect', bending_z=True)
    structure.add_support('N1', r_ux=True, r_uy=True, r_rz=True)
    structure.add_load_pattern('dead')
    structure.add_joint_point_load('dead', 'N2', fy=-10e3)
    structure.run_analysis()
    displacements = structure.displacements['dead']
"""

import pytest
import numpy as np
from pymas import Structure


class TestStructure:
    """Tests for basic Structure class functionality."""

    def test_type_is_required(self):
        """Verify the structure type must be given.

        There is no default type: Structure() raises TypeError.
        """
        with pytest.raises(TypeError):
            Structure()

    def test_type(self, structure):
        """Verify the structure type is stored."""
        assert structure.type == 'space_frame'

    def test_custom_type(self):
        """Verify a type not in _DOF_MAP takes its DoF from the flags.

        A custom type with ux and uy active behaves like a plane truss.
        """
        structure = Structure('custom', ux=True, uy=True)
        assert structure.type == 'custom'
        np.testing.assert_array_equal(
            structure.get_degrees_freedom(),
            np.array([True, True, False, False, False, False]))

    @pytest.mark.parametrize('flags', [{}, {'ux': False, 'rz': False}])
    def test_custom_type_without_active_dof_raises(self, flags):
        """Verify a custom type without any active DoF raises ValueError.

        Omitted flags are False, so a misspelled type such as 'plane_frme'
        (or the former 'axle') is reported instead of giving a structure
        with no DoF.
        """
        with pytest.raises(ValueError, match='torsional_bar'):
            Structure('custom', **flags)

    def test_custom_type_omitted_flags_are_false(self):
        """Verify the omitted flags of a custom type are False."""
        structure = Structure('custom', rx=True)
        np.testing.assert_array_equal(
            structure.get_degrees_freedom(),
            np.array([False, False, False, True, False, False]))

    def test_flags_ignored_for_known_type(self):
        """Verify the flags do not override a type in _DOF_MAP."""
        structure = Structure('beam', ux=True)
        np.testing.assert_array_equal(
            structure.get_degrees_freedom(),
            np.array([False, True, False, False, False, True]))


class TestDegreesOfFreedom:
    """Tests for degrees of freedom configuration.

    Degrees of freedom (DoF) define which displacements/rotations
    are considered in the analysis. They depend on the structure type.
    """

    @pytest.mark.parametrize('type, expected', [
        ('bar',            [True, False, False, False, False, False]),
        ('torsional_bar',  [False, False, False, True, False, False]),
        ('beam',           [False, True, False, False, False, True]),
        ('plane_truss',    [True, True, False, False, False, False]),
        ('plane_frame',    [True, True, False, False, False, True]),
        ('grid',           [False, True, True, True, False, False]),
        ('space_truss',    [True, True, True, False, False, False]),
        ('space_frame',    [True, True, True, True, True, True]),
    ])
    def test_degrees_freedom(self, type, expected):
        """Verify the active DoF of every predefined structure type."""
        structure = Structure(type)
        np.testing.assert_array_equal(
            structure.get_degrees_freedom(), np.array(expected))

    def test_number_active_dof_space_frame(self, structure):
        """Verify a space frame has 6 active DoF.

        number_active_degrees_freedom() returns count of True values.
        """
        assert structure.number_active_degrees_freedom() == 6

    def test_number_active_dof_space_truss(self):
        """Verify a space truss has 3 active DoF."""
        structure = Structure('space_truss')
        assert structure.number_active_degrees_freedom() == 3

    def test_number_active_dof_beam(self):
        """Verify a beam has 2 active DoF."""
        structure = Structure('beam')
        assert structure.number_active_degrees_freedom() == 2


class TestJointIndices:
    """Tests for joint indices assignment.

    Joint indices map each joint's DOFs to global matrix positions.
    Each joint gets n_dof consecutive indices starting from 0.
    """

    def test_set_joint_indices(self, simple_frame_structure):
        """Verify joint indices can be assigned to all joints.

        Creates indices for N1 and N2.
        """
        simple_frame_structure.set_joint_indices()
        indices = simple_frame_structure.get_joint_indices()
        assert 'N1' in indices
        assert 'N2' in indices

    def test_joint_indices_shape(self, simple_frame_structure):
        """Verify indices array has correct shape.

        Each joint gets n_dof indices (6 for a space frame).
        """
        simple_frame_structure.set_joint_indices()
        indices = simple_frame_structure.get_joint_indices()
        assert indices['N1'].shape == (6,)


class TestStiffnessMatrix:
    """Tests for global stiffness matrix assembly.

    The global stiffness matrix [K] is assembled from all element stiffness matrices.
    It's a sparse matrix stored as dense numpy array.
    """

    def test_set_stiffness_matrix(self, simple_frame_structure):
        """Verify stiffness matrix can be assembled.

        Matrix size = n_joints * n_dof per joint.
        For 2 joints in a space frame: 2 * 6 = 12
        """
        simple_frame_structure.set_joint_indices()
        simple_frame_structure.set_stiffness_matrix()
        k = simple_frame_structure.get_stiffness_matrix()
        assert k.shape == (12, 12)


class TestAnalysis:
    """Tests for structural analysis.

    run_analysis() solves the structure for all load patterns.
    It computes:
        - displacements: Joint displacements
        - reactions: Support reactions
        - end_actions: Forces at element ends
        - internal_forces: Forces along elements
        - internal_displacements: Displacements along elements
    """

    def test_run_analysis_simple(self, simple_frame_structure):
        """Verify simple analysis can run.

        Run analysis and check displacements are computed.
        """
        simple_frame_structure.add_load_pattern('dead')
        simple_frame_structure.add_joint_point_load('dead', 'N2', fy=-10e3)
        simple_frame_structure.run_analysis()
        assert 'dead' in simple_frame_structure.displacements

    def test_displacements_results(self, structure_with_loads):
        """Verify displacements are computed for all joints."""
        structure_with_loads.run_analysis()
        displacements = structure_with_loads.displacements['dead']
        for joint in ['N1', 'N2', 'N3']:
            assert joint in displacements

    def test_reactions_results(self, structure_with_loads):
        """Verify reactions are computed at supported joints."""
        structure_with_loads.run_analysis()
        reactions = structure_with_loads.reactions['dead']
        assert 'N1' in reactions
        assert 'N3' in reactions


class TestExport:
    """Tests for JSON export functionality.

    Export saves the structure model and analysis results to JSON format.
    Includes: materials, sections, joints, frames, supports, loads, results.
    """

    def test_export_json(self, simple_frame_structure, tmp_path):
        """Verify structure can be exported to JSON file.

        Export creates a valid JSON file with structure data.
        """
        simple_frame_structure.set_joint_indices()
        filename = tmp_path / "test_export.json"
        simple_frame_structure.export(str(filename))
        assert filename.exists()
    def test_export_frame_loads(self, structure_and_load, tmp_path):
        """Verify frame point and distributed loads are exported.

        Both kinds of load are listed under the frame, with their class name
        as type.
        """
        import json

        structure_and_load.add_load_pattern('live')
        structure_and_load.add_frame_point_load('live', 'F1', 2, fy=-10e3)
        structure_and_load.add_distributed_load('live', 'F1', fy=-5e3)
        filename = tmp_path / "test_export.json"
        structure_and_load.export(str(filename))

        with open(filename) as f:
            loads = json.load(f)['load_patterns']['live']['frames']['F1']

        assert [load['type'] for load in loads] == \
            ['FramePointLoad', 'DistributedLoad']
        assert loads[0]['dist'] == 2
        assert loads[0]['fy'] == -10e3
