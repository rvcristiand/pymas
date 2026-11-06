import json

import numpy as np
from scipy.sparse import coo_matrix

from .primitives import CircularSection, Displacement, EndActions, Frame, \
    FrameResults, InternalDisplacements, InternalForces, Joint, LoadPattern, \
    Material, RectangularSection, Reaction, Results, Section, Support


class Structure:
    """Analyzes linear elastic framed structures subjected to static loads.

    This class provides functionality to define, analyze, and export different
    types of framed structures. Users can add materials, cross sections,
    joints, frames, supports, and load patterns to build a complete structural
    model.

    The type of structure and its global degrees of freedom (DoF) are defined
    during instantiation. The class utilizes an internal `_DOF_MAP` to
    preconfigure DoF for common structure types. These DoF are the independent
    movements (translations and rotations) of each joint; every frame is
    computed in 3D and then reduced to the active DoF.

    The preconfigured structure types and their active degrees of freedom
    are:

    - 'bar': Axial bar (ux).
    - 'torsional_bar': Torsional bar (rx).
    - 'beam': Beam (uy, rz).
    - 'plane_truss': Plane truss (ux, uy).
    - 'plane_frame': Plane frame (ux, uy, rz).
    - 'grid': Grid in the xy plane, loaded along z (uz, rx, ry).
    - 'space_truss': Space truss (ux, uy, uz).
    - 'space_frame': Space frame (ux, uy, uz, rx, ry, rz).

    Alternatively, users can explicitly define the active translations (ux, uy,
    uz) and rotations (rx, ry, rz) for a custom structural model if none of the
    predefined types fit their specific needs. These flags are ignored for a
    predefined type.

    `run_analysis()` stores the results of each load pattern in `results`. The
    results of the 'default' load pattern can also be reached through
    shortcuts: `displacements`, `reactions`, `end_actions`, `internal_forces`,
    `internal_displacements` and the `plot_*` methods.

    Attributes:
        type (str): Type of structure.
        materials (dict): Materials of the model.
        sections (dict): Cross sections of the model.
        joints (dict): Joints of the model.
        frames (dict): Frames of the model.
        supports (dict): Joint supports of the model.
        load_patterns (dict): Load patterns of the model.
        results (dict): Results of the analysis.
    """

    _DOF_MAP = {
        'bar':           {'ux': True},
        'torsional_bar': {'rx': True},
        'beam':          {'uy': True, 'rz': True},
        'plane_truss':   {'ux': True, 'uy': True},
        'plane_frame':   {'ux': True, 'uy': True, 'rz': True},
        'space_truss':   {'ux': True, 'uy': True, 'uz': True},
        'grid':          {'uz': True, 'rx': True, 'ry': True},
        'space_frame':   {'ux': True, 'uy': True, 'uz': True,
                          'rx': True, 'ry': True, 'rz': True},
    }

    def __init__(self, type, *, ux=False, uy=False, uz=False, rx=False,
                 ry=False, rz=False):
        """Instantiates a Structure object.

        Initializes a new structural model and sets up internal variables and
        empty dictionaries for various structural components and analysis
        results.

        Args:
            type (str): Type of the structure (e.g., 'plane_frame',
                'space_truss'). This determines the active degrees of freedom
                if present in `_DOF_MAP`; otherwise they are taken from the
                flags below, which are ignored for a predefined type.
            ux (bool, optional): Defines if translation in the x-direction is
                an active degree of freedom.
            uy (bool, optional): Defines if translation in the y-direction is
                an active degree of freedom.
            uz (bool, optional): Defines if translation in the z-direction is
                an active degree of freedom.
            rx (bool, optional): Defines if rotation about the x-axis is an
                active degree of freedom.
            ry (bool, optional): Defines if rotation about the y-axis is an
                active degree of freedom.
            rz (bool, optional): Defines if rotation about the z-axis is an
                active degree of freedom.

        Raises:
            ValueError: If `type` is not in `_DOF_MAP` and no degree of
                freedom is set to True.
        """
        # type of structure and degrees of freedom
        self.type = type

        # flags as given by the user
        self._ux, self._uy, self._uz = ux, uy, uz
        self._rx, self._ry, self._rz = rx, ry, rz

        # the degrees of freedom come from _DOF_MAP, or from the flags for a
        # custom type; a degree of freedom missing from the map is inactive
        flags = {'ux': ux, 'uy': uy, 'uz': uz, 'rx': rx, 'ry': ry, 'rz': rz}
        dofs = self._DOF_MAP.get(type, flags)

        # set the degrees of freedom of the model
        self._dof = np.array([dofs.get(dof, False) for dof in flags])

        # a structure without degrees of freedom cannot be analysed
        if not self._dof.any():
            types = ', '.join(repr(t) for t in self._DOF_MAP)
            raise ValueError(
                f"{type!r} is not a predefined structure type and no degree "
                f"of freedom is active. Use one of {types}, or set at least "
                f"one of ux, uy, uz, rx, ry, rz to True."
            )

        # dictionary of joint indices
        self._j_i = {}

        # matrix stiffness of the structure
        self._k = np.array([])

        # material and cross section dictionaries
        self.materials = {}
        self.sections = {}

        # joint and frame dictionaries
        self.joints = {}
        self.frames = {}

        # joint support and load pattern dictionaries
        self.supports = {}
        self.load_patterns = {}

        # results of the analysis, by load pattern
        self.results = {}

    def add_material(self, name, E=None, G=None):
        """Add a material to the model.

        Adds a new material object to the model's material dictionary.

        Args:
            name (str): Name of the material.
            E (float, optional): Elastic modulus of the material.
            G (float, optional): Shear modulus of the material.

        Returns:
            Material: Material object.
        """
        # add a material object to the dictionary of materials
        material = self.materials[name] = Material(self, name, E, G)

        return material

    def add_section(self, name, A=None, J=None, Iy=None, Iz=None):
        """Add a cross section to the model.

        Creates and adds a new generic cross section object to the model's
        section dictionary.

        Args:
            name (str): Name of the cross section.
            A (float, optional): Area of the cross section.
            J (float, optional): Torsion constant of the cross section.
            Iy (float, optional): Inertia of the cross section with respect to
                the local y-axis.
            Iz (float, optional): Inertia of the cross section with respect to
                the local z-axis.

        Returns:
            Section: Cross section.
        """
        # add a section object to the dictionary of cross sections
        section = self.sections[name] = Section(self, name, A, J, Iy, Iz)

        return section

    def add_circular_section(self, name, diameter):
        """Add a circular cross section to the model.

        Creates and adds a new circular cross section object to the model's
        section dictionary, calculating its area, torsion constant, and moments
        of inertia based on its diameter.

        Args:
            name (str): Name of the circular cross section.
            diameter (float): Diameter of the circular cross section.

        Returns:
            CircularSection: Circular cross section.
        """
        # add a circular cross section object to the dict of cross sections
        circ_sect = self.sections[name] = CircularSection(self, name, diameter)

        return circ_sect

    def add_rectangular_section(self, name, *, dy, dz):
        """Add a rectangular cross section to the model.

        Creates and adds a new rectangular cross section object to the model's
        section dictionary, calculating its area, torsion constant, and moments
        of inertia based on its dimensions along the local y- and z-axes.

        Args:
            name (str): Name of the rectangular cross section.
            dy (float): Dimension of the rectangular cross section along the
                local y-axis. Bending about the local z-axis uses it as depth.
            dz (float): Dimension of the rectangular cross section along the
                local z-axis. Bending about the local y-axis uses it as depth.

        Returns:
            RectangularSection: Rectangular cross section.
        """
        # create a rectangular cross section object
        rect_sect = RectangularSection(self, name, dy=dy, dz=dz)

        # add the rect cross section object to the dict of cross sections
        self.sections[name] = rect_sect

        return rect_sect

    def add_joint(self, name, x=None, y=None, z=None):
        """Add a joint to the model.

        Creates and adds a new joint object to the model's joint dictionary.

        Args:
            name (str): Name of the joint.
            x (float, optional): Coordinate X of the joint.
            y (float, optional): Coordinate Y of the joint.
            z (float, optional): Coordinate Z of the joint.

        Returns:
            Joint: Joint object.
        """
        # add a joint object to the dictionary of joints
        joint = self.joints[name] = Joint(self, name, x, y, z)

        return joint

    def add_frame(self, name, joint_j, joint_k, material, section,
                  axial=None, torsional=None, bending_y=None, bending_z=None,
                  roll=None):
        """Add a frame to the model.

        Creates and adds a new frame object to the model's frame dictionary,
        connecting two specified joints with a given material and
        cross section.

        Args:
            name (str): Name of the frame.
            joint_j (str): Name of the near joint of the frame.
            joint_k (str): Name of the far joint of the frame.
            material (str): Name of the material of the frame.
            section (str): Name of the section of the frame.
            axial (bool, optional): Consideration of axial deformation of the
                frame. Defaults to None.
            torsional (bool, optional): Consideration of torsional deformation
                of the frame. Defaults to None.
            bending_y (bool, optional): Consideration of bending around the
                local y-axis of the frame. Defaults to None.
            bending_z (bool, optional): Consideration of bending around the
                local z-axis of the frame. Defaults to None.
            roll (float, optional): Rotation of the cross section about the
                local x-axis, in radians, following the right-hand rule with
                the thumb pointing from the near to the far joint. Defaults to
                None, which means 0.

        Returns:
            Frame: Frame object.
        """
        # create a frame object
        frame = Frame(self, name, joint_j, joint_k, material, section,
                      axial=axial, torsional=torsional, bending_y=bending_y,
                      bending_z=bending_z, roll=roll)

        # add the frame object to the dictionary of frames
        self.frames[name] = frame

        return frame

    def add_support(self, joint, *, r_ux=None, r_uy=None, r_uz=None,
                    r_rx=None, r_ry=None, r_rz=None):
        """Add a joint support to the model.

        Creates and adds a new joint support object to the model's support
        dictionary for a specified joint, defining which displacements are
        restrained. The restraints are keyword-only, e.g.
        `add_support('a', r_ux=True, r_uy=True)`.

        Args:
            joint (str): Name of the joint.
            r_ux (bool, optional): Whether the global x-axis translation is
                restrained.
            r_uy (bool, optional): Whether the global y-axis translation is
                restrained.
            r_uz (bool, optional): Whether the global z-axis translation is
                restrained.
            r_rx (bool, optional): Whether the global x-axis rotation is
                restrained.
            r_ry (bool, optional): Whether the global y-axis rotation is
                restrained.
            r_rz (bool, optional): Whether the global z-axis rotation is
                restrained.

        Returns:
            Support: Joint support.
        """
        # create a joint support object
        support = Support(self, joint, r_ux, r_uy, r_uz, r_rx, r_ry, r_rz)

        # add the joint support object to the dictionary of supports
        self.supports[joint] = support

        return support

    def add_load_pattern(self, name):
        """Add a load pattern to the model.

        Creates and adds a new load pattern object to the model's load pattern
        dictionary.

        Args:
            name (str): Name of the load pattern.

        Returns:
            LoadPattern: Load pattern.
        """
        # add a load pattern object to the dictionary of load patterns
        loadPattern = self.load_patterns[name] = LoadPattern(self, name)

        return loadPattern

    def _default_load_pattern(self):
        """Return the 'default' load pattern, creating it the first time."""
        if 'default' not in self.load_patterns:
            self.add_load_pattern('default')

        return self.load_patterns['default']

    def add_joint_point_load(self, joint, fx=None, fy=None, fz=None, mx=None,
                             my=None, mz=None):
        """Add a joint point load to the 'default' load pattern.

        The 'default' load pattern is created the first time it is used.
        To add it to another load pattern, use the methods of the load pattern
        returned by `add_load_pattern()`.

        Args:
            joint (str): Name of the joint.
            fx (float, optional): Intensity of the point load along the global
                x-axis.
            fy (float, optional): Intensity of the point load along the global
                y-axis.
            fz (float, optional): Intensity of the point load along the global
                z-axis.
            mx (float, optional): Intensity of the point load around the global
                x-axis.
            my (float, optional): Intensity of the point load around the global
                y-axis.
            mz (float, optional): Intensity of the point load around the global
                z-axis.

        Returns:
            PointLoad: Point load.
        """
        # get the load pattern object from the dictionary of load patterns
        lP = self._default_load_pattern()
        # add the point load to the load pattern
        pointLoad = lP.add_joint_point_load(joint, fx, fy, fz, mx, my, mz)

        return pointLoad

    def add_frame_point_load(self, frame, dist, fx=None, fy=None, fz=None,
                             mx=None, my=None, mz=None):
        """Add a frame point load to the 'default' load pattern.

        The load acts at a given distance from the near joint of the frame.
        The 'default' load pattern is created the first time it is used.
        To add it to another load pattern, use the methods of the load pattern
        returned by `add_load_pattern()`.

        Args:
            frame (str): Name of the frame.
            dist (float): Distance of the point load from the near joint.
            fx (float, optional): Intensity of the point load along the local
                x-axis.
            fy (float, optional): Intensity of the point load along the local
                y-axis.
            fz (float, optional): Intensity of the point load along the local
                z-axis.
            mx (float, optional): Intensity of the point load around the local
                x-axis.
            my (float, optional): Intensity of the point load around the local
                y-axis.
            mz (float, optional): Intensity of the point load around the local
                z-axis.

        Returns:
            FramePointLoad: Frame point load.
        """
        # get the load pattern object from the dictionary of load patterns
        lP = self._default_load_pattern()
        # add the point load to the load pattern
        pL = lP.add_frame_point_load(frame, dist, fx, fy, fz, mx, my, mz)

        return pL

    def add_distributed_load(self, frame, fx=None, fy=None, fz=None, mx=None,
                             my=None, mz=None):
        """Add a frame uniformly distributed load to the 'default' load
        pattern.

        The 'default' load pattern is created the first time it is used.
        To add it to another load pattern, use the methods of the load pattern
        returned by `add_load_pattern()`.

        Args:
            frame (str): Name of the frame.
            fx (float, optional): Intensity of the distributed load along the
                local x-axis.
            fy (float, optional): Intensity of the distributed load along the
                local y-axis.
            fz (float, optional): Intensity of the distributed load along the
                local z-axis.
            mx (float, optional): Intensity of the distributed load around the
                local x-axis.
            my (float, optional): Intensity of the distributed load around the
                local y-axis.
            mz (float, optional): Intensity of the distributed load around the
                local z-axis.

        Returns:
            DistributedLoad: Uniformly distributed load.
        """
        # get the load pattern object from the dictionary of load patterns
        lP = self._default_load_pattern()
        # add the uniformly distributed load to the load pattern
        dL = lP.add_distributed_load(frame, fx, fy, fz, mx, my, mz)

        return dL

    def get_degrees_freedom(self):
        """Return the degrees of freedom of the model.

        Returns:
            ndarray: Degrees of freedom of the model.
        """
        return self._dof

    def number_active_degrees_freedom(self):
        """Get the number of active degrees of freedom of the joints of the
        model.

        Returns:
            int: Number of active degrees of freedom of the joints of model.
        """
        return np.count_nonzero(self.get_degrees_freedom())

    def get_joint_indices(self):
        """Return the joint indices of the model.

        Returns:
            dict: Joint indices of the model.
        """
        return self._j_i

    def set_joint_indices(self):
        """Set the joint indices of the model.

        This method assigns unique, contiguous global indices to each degree of
        freedom for every joint in the model, storing them in the `_j_i`
        dictionary.
        """
        # number of joints
        n_j = len(self.joints)
        # number of active degrees of freedom
        n_dof = self.number_active_degrees_freedom()
        # create the joint indices of the structure
        joint_indices = np.arange(n_j * n_dof).reshape(n_j, n_dof)
        # set the dictionary of joint indices
        self._j_i = {j: i for j, i in zip(self.joints, joint_indices)}

    def get_stiffness_matrix(self):
        """Return the stiffness matrix of the model.

        Returns:
            ndarray: Stiffness matrix of the model.
        """
        return self._k

    def set_stiffness_matrix(self):
        """Set the stiffness matrix of the structure.

        This method calculates the local stiffness matrices for all frames,
        transforms them to the global coordinate system, and assembles them
        into the global stiffness matrix of the structure. It also applies
        modifications due to supports.
        """
        # number of joints
        n_j = len(self.joints)
        # number of frames
        n_f = len(self.frames)

        # number of active degrees of freedom per joint
        n_dof = self.number_active_degrees_freedom()
        # number of degrees of freedom per frame
        n_dof_f = 2 * n_dof
        # number of degrees of freedom of the structure
        n_dof_s = n_j * n_dof

        # joint indices
        j_i = self.get_joint_indices()

        # number of items of the stiffness matrix of the frames
        n = n_dof_f**2

        # row and column positions of the striffness matrices items
        # of the frames in the stiffness matrix of the structure
        rows = np.empty(n_f * n, dtype=int)
        cols = np.empty_like(rows)
        # data of the stiffness matrices of the frames
        data = np.empty_like(rows, dtype=float)

        # assembly the frame stiffness matrices
        for i, frame in enumerate(self.frames.values()):
            # get the joint indices of the near and far joints of the frame...
            e_i = np.concatenate((j_i[frame.joint_j], j_i[frame.joint_k]))
            # ... to create a readonly view of the joint indices, with the
            # shape of the frame stiffness matrix
            e_i = np.broadcast_to(e_i, (n_dof_f, n_dof_f))

            # get the global stiffness matrix of the frame
            k_e = frame.global_stiffness_matrix()

            # start and end indices
            start = i * n
            end = (i + 1) * n

            # collapse the replicated joint indices to get the rows
            # and columns positions of the frames of the stiffness matrix
            rows[start:end] = e_i.flatten('F')
            cols[start:end] = e_i.flatten()

            # collapse the frame matrix stiffness
            data[start:end] = k_e.flatten()

        # create the stiffness matrix of the structure
        k_s = coo_matrix((data, (rows, cols)), 2 * (n_dof_s, )).toarray()

        # modify the stiffness matrix of the structure by the supports
        for joint, support in self.supports.items():
            # joint indices of the support
            indices = j_i[joint]

            # restrains of the support
            restrains = support.restrains()

            # modify the stiffness matrix of the structure
            for index in indices[restrains]:
                k_s[index] = k_s[:, index] = np.zeros(n_dof_s)
                k_s[index, index] = 1

        # set the stiffness matrix of the structure
        self._k = k_s

    def analyse_load_pattern(self, load_pattern):
        """Analyse the model subjected to a load pattern.

        This method calculates the joint displacements, frame end actions,
        support reactions, and internal forces and displacements for a given
        load pattern.

        Args:
            load_pattern (str): Load pattern name.
        """
        # degrees of freedom of the joints
        dof_j = self.get_degrees_freedom()
        # degrees of freedom of the frames
        dof_f = np.tile(dof_j, 2)

        # number active degrees of freedom per joint
        n_dof_j = np.count_nonzero(dof_j)
        # number active degrees of freedom per frame
        n_dof_f = 2 * n_dof_j
        # number of joints
        n_j = len(self.joints)

        # joint indices of the structure
        j_i = self.get_joint_indices()

        # load pattern object
        loadPattern = self.load_patterns[load_pattern]

        load_vector = loadPattern.load_vector()
        load_vector_support = np.copy(load_vector)

        for j, support in self.supports.items():
            indices = j_i[j]
            restrains = support.restrains()

            for index in indices[restrains]:
                load_vector_support[index, 0] = 0

        # find displacements
        u = np.linalg.solve(self.get_stiffness_matrix(), load_vector_support)

        # store displacements
        l_p_d = {}

        for j in self.joints:
            indices = j_i[j]

            displacement = np.full(6, None)
            displacement[dof_j] = u[indices, 0]

            l_p_d[j] = Displacement(self, load_pattern, j, *displacement)

        # the results of the load pattern; the frames read the displacements
        # and end actions from them to get their internal results
        results = self.results[load_pattern] = Results(self, load_pattern)
        results.displacements = l_p_d

        # store frame end actions
        rows = []
        cols = []
        data = []

        l_p_e_a = {}

        for key, frame in self.frames.items():
            i_e = np.concatenate((j_i[frame.joint_j], j_i[frame.joint_k]))

            k_e = frame.local_stiffness_matrix()
            t_e = frame.rotation_transformation_matrix()

            u_e = np.zeros((12, 1))
            u_e[dof_f] = u[i_e]
            u_e = np.dot(np.transpose(t_e), u_e)

            f_fixed = np.zeros((12, 1))

            if key in loadPattern.frame_point_loads:
                for pLoad in loadPattern.frame_point_loads[key]:
                    f_fixed += pLoad.fixed_load_vector()

            if key in loadPattern.frame_distributed_loads:
                for dLoad in loadPattern.frame_distributed_loads[key]:
                    f_fixed += dLoad.fixed_load_vector()

            f_end_actions = np.full(12, None)
            f_end_actions[dof_f] = \
                np.ravel(np.dot(k_e, u_e) + f_fixed)[dof_f]

            l_p_e_a[key] = EndActions(self, load_pattern, key, *f_end_actions)
            results.frames[key] = FrameResults(self, load_pattern, key,
                                               end_actions=l_p_e_a[key])

            # reactions
            if frame.joint_j in self.supports or \
               frame.joint_k in self.supports:
                rows.extend(i_e.tolist())
                cols.extend(n_dof_f * [0])
                data.extend(np.dot(
                    t_e, l_p_e_a[key].get_end_actions()
                ).flatten()[dof_f].tolist())

        # store reactions
        load_vector += loadPattern.fixed_load_vector()
        f_end_actions = coo_matrix(
            (data, (rows, cols)), (n_j * n_dof_j, 1)).toarray()

        load_pattern_reactions = {}

        for j in self.supports:
            indices = j_i[j]
            reactions = np.full(6, None)
            reactions[dof_j] = np.ravel(
                f_end_actions[indices] - load_vector[indices])

            load_pattern_reactions[j] = Reaction(
                self, load_pattern, j, *reactions)

        results.reactions = load_pattern_reactions

        # store internal forces and displacements
        for key, frame in self.frames.items():
            frame_results = results.frames[key]
            frame_results.internal_forces = InternalForces(
                self, load_pattern, key,
                **frame.get_internal_forces(load_pattern))
            frame_results.internal_displacements = InternalDisplacements(
                self, load_pattern, key,
                **frame.get_internal_displacements(load_pattern))

    def run_analysis(self):
        """Analyse the structure subjected to all load patterns.

        This method first sets up the joint indices and the global stiffness
        matrix, then proceeds to analyze the structure for each load pattern
        sequentially.

        """
        # set joint indices
        self.set_joint_indices()

        # set the stiffness matrix of the structure
        self.set_stiffness_matrix()

        # solve the structure due to each load pattern, dropping the results
        # of a previous analysis
        self.results = {}
        for load_pattern in self.load_patterns:
            self.analyse_load_pattern(load_pattern)

    # shortcuts to the results of the 'default' load pattern, which takes the
    # loads added through the methods of the structure

    def _default_results(self):
        """Return the results of the 'default' load pattern.

        Raises:
            ValueError: If the 'default' load pattern has no results.
        """
        if 'default' not in self.results:
            raise ValueError("there are no results of the 'default' load "
                             "pattern; add loads through the structure and "
                             "call run_analysis(), or use "
                             "model.results[name] for other load patterns")

        return self.results['default']

    def _default_frame_results(self, attribute):
        """Return a result of every frame for the 'default' load pattern."""
        return {name: getattr(frame_results, attribute) for name,
                frame_results in self._default_results().frames.items()}

    @property
    def displacements(self):
        """dict: Joint displacements of the 'default' load pattern, by joint.

        A shortcut to `results['default'].displacements`.
        """
        return self._default_results().displacements

    @property
    def reactions(self):
        """dict: Support reactions of the 'default' load pattern, by joint.

        A shortcut to `results['default'].reactions`.
        """
        return self._default_results().reactions

    @property
    def end_actions(self):
        """dict: Frame end actions of the 'default' load pattern, by frame.

        A shortcut to the `end_actions` of `results['default'].frames`.
        """
        return self._default_frame_results('end_actions')

    @property
    def internal_forces(self):
        """dict: Internal forces of the 'default' load pattern, by frame.

        A shortcut to the `internal_forces` of `results['default'].frames`.
        """
        return self._default_frame_results('internal_forces')

    @property
    def internal_displacements(self):
        """dict: Internal displacements of the 'default' load pattern, by
        frame.

        A shortcut to the `internal_displacements` of
        `results['default'].frames`.
        """
        return self._default_frame_results('internal_displacements')

    def plot_loads(self, ax=None, reactions=True, dimensions=False,
                   support_names=False, frame_names=False):
        """Plot the loads and reactions of the 'default' load pattern.

        A shortcut to `results['default'].plot_loads()`, which describes the
        plot.

        Args:
            ax (Axes, optional): Matplotlib axes to draw on. By default, a
                new figure is created.
            reactions (bool, optional): Whether to draw the reactions.
            dimensions (bool, optional): Whether to dimension the
                structure. False by default.
            support_names (bool, optional): Whether to write the name of
                the joint of each support beside it. False by default.
            frame_names (bool, optional): Whether to write the name of each
                frame, in a circle, beside its middle. False by default.

        Returns:
            Axes: Matplotlib axes of the plot.
        """
        return self._default_results().plot_loads(ax, reactions,
                                                  dimensions, support_names,
                                                  frame_names)

    def plot_deformed(self, ax=None, scale=None, labels=False,
                      dimensions=False):
        """Plot the deformed shape under the 'default' load pattern.

        A shortcut to `results['default'].plot_deformed()`, which describes
        the plot.

        Args:
            ax (Axes, optional): Matplotlib axes to draw on. By default, a
                new figure is created.
            scale (float, optional): Factor applied to the displacements. By
                default, the largest displacement is drawn as 10% of the size
                of the structure.
            labels (bool, optional): Whether to label the largest deflection
                of each frame. False by default.
            dimensions (bool, optional): Whether to dimension the position
                of the largest deflection of each frame. False by default.

        Returns:
            Axes: Matplotlib axes of the plot.
        """
        return self._default_results().plot_deformed(ax, scale, labels,
                                                     dimensions)

    def plot_diagram(self, force, ax=None, scale=None, labels=False,
                     dimensions=False):
        """Plot the diagram of an internal force under the 'default' load
        pattern.

        A shortcut to `results['default'].plot_diagram()`, which describes
        the plot.

        Args:
            force (str): Internal force: 'fx', 'fy', 'fz', 'mx', 'my' or
                'mz'.
            ax (Axes, optional): Matplotlib axes to draw on. By default, a
                new figure is created.
            scale (float, optional): Length per unit of force. By default,
                the largest value is drawn as 15% of the size of the
                structure.
            labels (bool, optional): Whether to label the largest positive
                and negative values of each frame, and those at its ends.
                False by default.
            dimensions (bool, optional): Whether to dimension the zero
                crossings. False by default.

        Returns:
            Axes: Matplotlib axes of the plot.
        """
        return self._default_results().plot_diagram(force, ax, scale,
                                                    labels, dimensions)

    def plot_internal_forces(self, frame, axes=None, labels=False):
        """Plot the internal forces of a frame under the 'default' load
        pattern.

        A shortcut to `results['default'].frames[frame]
        .plot_internal_forces()`, which describes the plot.

        Args:
            frame (str): Name of the frame.
            axes (list, optional): Matplotlib axes to draw on, one per
                internal force that the frame models. By default, a new
                figure is created.
            labels (bool, optional): Whether to label the largest values
                and the zero crossings. False by default.

        Returns:
            Figure: Matplotlib figure, with one axes per internal force.
        """
        return self._default_results().frames[frame].plot_internal_forces(
            axes, labels)

    def plot_internal_displacements(self, frame, axes=None, labels=False,
                                    components=None):
        """Plot the internal displacements of a frame under the 'default'
        load pattern.

        A shortcut to `results['default'].frames[frame]
        .plot_internal_displacements()`, which describes the plot.

        Args:
            frame (str): Name of the frame.
            axes (list, optional): Matplotlib axes to draw on, one per
                plotted displacement. By default, a new figure is created.
            labels (bool, optional): Whether to label the largest values
                and the zero crossings. False by default.
            components (list, optional): Displacements to plot, among
                those that the frame models, e.g. ('uy', 'rz'). By default,
                its translations.

        Returns:
            Figure: Matplotlib figure, with one axes per plotted
                displacement.
        """
        return self._default_results().frames[frame]\
            .plot_internal_displacements(axes, labels, components)

    def export(self, filename):
        """Save the structure data to a file in JSON format.

        This method serializes the type of structure and its active degrees of
        freedom, and the model's materials, sections, joints, frames,
        supports, load patterns (including their applied loads), and analysis
        results (displacements, reactions, end actions, internal forces, and
        internal displacements) into a JSON format.

        Args:
            filename (str): Filename.
        """
        data = {}

        # save the type of structure and its active degrees of freedom
        data['type'] = self.type
        data['degrees_of_freedom'] = [
            dof for dof, active in
            zip(('ux', 'uy', 'uz', 'rx', 'ry', 'rz'), self._dof) if active]

        # save the materials
        if self.materials:
            data['materials'] = {}
            for key, material in self.materials.items():
                data['materials'][key] = \
                    {attr: value for attr, value in material.__dict__.items()
                     if not attr.startswith('_') and value is not None}

        # save the sections
        if self.sections:
            data['sections'] = {}
            for key, section in self.sections.items():
                data['sections'][key] = {'name': section.name}
                data['sections'][key]['type'] = section.__class__.__name__
                data['sections'][key].update({attr: value for attr, value in
                                              section.__dict__.items() if not
                                              attr.startswith('_') and value is
                                              not None})

        # save the joints
        if self.joints:
            data['joints'] = {}
            for key, joint in self.joints.items():
                data['joints'][key] = {attr: value for attr, value in
                                       joint.__dict__.items() if not
                                       attr.startswith('_') and value is not
                                       None}

        # save the frames
        if self.frames:
            data['frames'] = {}
            for key, frame in self.frames.items():
                data['frames'][key] = {attr: value for attr, value in
                                       frame.__dict__.items() if not
                                       attr.startswith('_') and value is not
                                       None}

        # save the supports
        if self.supports:
            data['supports'] = {}
            for key, support in self.supports.items():
                data['supports'][key] = {attr: value for attr, value in
                                         support.__dict__.items() if not
                                         attr.startswith('_') and value is not
                                         None}

        # save the load patterns
        if self.load_patterns:
            data['load_patterns'] = {}
            for key, loadPattern in self.load_patterns.items():
                data['load_patterns'][key] = {'name': loadPattern.name}

                # save loads at joints
                if loadPattern.joint_point_loads:
                    data['load_patterns'][key]['joints'] = {}
                    for _joint, point_loads in \
                            loadPattern.joint_point_loads.items():
                        data['load_patterns'][key]['joints'][_joint] = []
                        for pointLoad in point_loads:
                            data['load_patterns'][key]['joints'][_joint]\
                                .append({attr: value for attr, value in
                                         pointLoad.__dict__.items() if not
                                         attr.startswith('_') and value is not
                                         None})

                # save loads at frames
                if loadPattern.frame_point_loads or \
                        loadPattern.frame_distributed_loads:

                    data['load_patterns'][key]['frames'] = {}

                    for _frame, frame_loads in \
                            list(loadPattern.frame_point_loads.items()) + \
                            list(loadPattern.frame_distributed_loads.items()):
                        if _frame not in data['load_patterns'][key]['frames']:
                            data['load_patterns'][key]['frames'][_frame] = []
                        for frameLoad in frame_loads:
                            _data = {'type': frameLoad.__class__.__name__}
                            _data.update({attr: value for attr, value in
                                          frameLoad.__dict__.items(
                                          ) if not attr.startswith('_') and
                                          value is not None})
                            data['load_patterns'][key]['frames'][_frame].\
                                append(_data)

        # save the results, by load pattern
        if self.results:
            data['results'] = {}
            for key, results in self.results.items():
                data['results'][key] = {
                    'displacements': {joint: _public(displacement) for
                                      joint, displacement in
                                      results.displacements.items()},
                    'reactions': {joint: _public(reaction) for joint, reaction
                                  in results.reactions.items()},
                    'frames': {frame: {
                        'end_actions': _public(frame_results.end_actions),
                        'internal_forces':
                            _public(frame_results.internal_forces),
                        'internal_displacements':
                            _public(frame_results.internal_displacements),
                    } for frame, frame_results in results.frames.items()},
                }

        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def _public(obj):
    """Return the public attributes of an object that are not None."""
    return {attr: value for attr, value in obj.__dict__.items()
            if not attr.startswith('_') and value is not None}
