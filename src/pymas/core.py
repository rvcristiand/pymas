import json
import numpy as np

from scipy.sparse import coo_matrix
# TODO mejorar la importanción de este modulo
from pymas.primitives import *


class Structure:
    # The available type of structures are:

    # - '3D': 6 degrees-of-freedom per joint applicable for a general
    #         tridimensional structural analysis.
    #
    # Note: it would be as many options as type of structures (plane truss,
    #       plane frame, plane grid, space truss and space frame [See Fenves,
    #       1964]) and two types of elements: a frame element and a truss
    #       element type. To analyse the structure, restrain the displacements
    #       of the structure according to the type of structure.
    """Model and analyze a linear framed structure subjected to static loads.

    TODO describe this class in a paragraph

    Describe the structure...

    Attributes
    ----------
    type : str
        Type of structure.
    materials : dict
        Materials of the structure.
    sections : dict
        Cross sections of the structure.
    joints : dict
        Joints of the structure.
    elements : dict
        Elements of the structure.
    supports : dict
        Supports of the structure.
    load_patterns : dict
        Load patterns of the structure.
    displacements : dict
        Joint displacements of the structure.
    end_actions : dict
        Element end actions of the structure.
    reactions : dict
        Support reactions of the structure.
    internal_forces: dict
        Internal element forces of the structure.
    internal_displacements: dict
        Internal element displacements of the structure.

    Methods
    -------
    add_material(name, modulus_elasticity, modulus_elasticity_shear)
        Add a material to the structure.
    add_section(name, area, torsion_constant, inertia_yy, inertia_zz)
        Add a cross section to the structure.
    add_rectangular_section(name, base, height)
        Add a rectangular cross section to the structure.
    add_joint(name, [x, y, z])
        Add a joint to the structure.
    add_truss(name, joint_j, joint_k, material, section)
        Add a truss to the structure.
    add_frame(name, joint_j, joint_k, material, section)
        Add a frame to the structure.
    add_support(joint, [r_ux, r_uy, r_uz, r_rx, r_ry, r_rz])
        Add a support to the structure.
    add_load_pattern(name)
        Add a load pattern to the structure.
    add_point_load_at_joint(load_pattern, joint, [fx, fy, fz, mx, my, mz])
        Add a point load at joint to the structure.
    add_uniformly_distributed_load(load_pattern, element, [wx, wy, wz])
        Add a uniformly distributed load at element of the structure.
    get_flags_joint_displacements()
        Returns the array of joint displacement flags.
    set_flags_joint_displacements()
        Set the array of joint displacement flags.
    get_joint_indices()
        Returns the joint indices of the structure.
    set_indices()
        Set the joint indices of the structure.
    get_stiffness_matrix()
        Returns the stiffness matrix of the structure.
    get_stiffness_matrix_modified_by_supports()
        Returns the stiffness matrix of the structure modified by the supports.
    set_stiffness_matrix_modified_by_supports()
        Set the stiffness matrix of the structure modified by the supports.
    solve_load_pattern(load_pattern)
        Solve a load pattern of the structure.
    solve()
        Solve the structure.
    export(filename)
        Export the model of the structure.
    """

    def __init__(self, type='3D'):
        """Instantiate a Structure object.

        Parameters
        ----------
        type : str
            Type of structure.
        """
        # initialize the internal variables

        # flags of the joint displacements
        self._ux = None
        self._uy = None
        self._uz = None
        self._rx = None
        self._ry = None
        self._rz = None
        # array of the joint displacement flags
        self._flags_joint_displacements = np.array([])
        # dictionary of joint indices of the structure
        self._joint_indices = {}
        # matrix stiffness of the structure modified by the supports
        self._ksupport = np.array([])

        # type of the structure
        self.type = type

        # material and section dictionaries
        self.materials = {}
        self.sections  = {}

        # joint and element dictionaries
        self.joints   = {}
        self.elements = {}

        # support and load pattern dictionaries
        self.supports      = {}
        self.load_patterns = {}

        # joint displacements, element end actions and support reactions dicts
        self.displacements = {}
        self.end_actions   = {}
        self.reactions     = {}

        # internal element forces and displacements dictionaries
        self.internal_forces        = {}
        self.internal_displacements = {}

    def add_material(self, name, modulus_elasticity, modulus_elasticity_shear):
        """Add a material to the structure.

        Parameters
        ----------
        name : str
            Name of the material to add.
        modulus_elasticity : float
            Modulus of elasticity of the material to add.
        modulus_elasticity_shear : float
            Modulus of elasticity in shear of the material to add.

        Returns
        -------
        material : Material
            Material object.
        """
        # material properties
        E = modulus_elasticity
        G = modulus_elasticity_shear

        # add a material object to the dictionary of materials
        material = self.materials[name] = Material(self, name, E, G)

        return material

    def add_section(self, name, area, torsion_constant, inertia_yy, inertia_zz):
        """Add a cross section to the structure.

        Parameters
        ----------
        name : str
            Name of the cross section to add.
        area : float
            Area of the cross section to add.
        torsion_constant : float
            Torsion constant of the cross section to add.
        inertia_yy : float
            Moment of inertia of the cross section to add with respect to the
            local y-axis.
        inertia_zz : float
            Moment of inertia of the cross section to add with respect to the
            local z-axis.

        Returns
        -------
        section : Section
            Section object.
        """
        # cross section properties
        A = area
        J = torsion_constant
        Iyy = inertia_yy
        Izz = inertia_zz

        # add a section object to the dictionary of cross sections
        section = self.sections[name] = Section(self, name, A, J, Iyy, Izz)

        return section

    def add_rectangular_section(self, name, base, height):
        """Add a rectangular cross section to the structure.

        Parameters
        ----------
        name : str
            Name of the rectangular cross section to add.
        base : float
            Base of the rectangular cross section to add.
        height : float
            Height of the rectangular cross section to add.

        Returns
        -------
        rect_sect : RectangularSection
            RectangularSection object.
        """
        # create a rectangular cross section object
        rect_sect = RectangularSection(self, name, base, height)

        # add the rectangular cross section object to the dict of cross sections
        self.sections[name] = rect_sect

        return rect_sect

    def add_joint(self, name, x=None, y=None, z=None):
        """Add a joint to the structure.

        Parameters
        ----------
        name : str
            Name of the joint to add.
        x : float, optional
            Coordinate X of the joint to add.
        y : float, optional
            Coordinate Y of the joint to add.
        z : float, optional
            Coordinate Z of the joint to add.

        Returns
        -------
        joint : Joint
            Joint object.
        """
        # add a joint object to the dictionary of joints
        joint = self.joints[name] = Joint(self, name, x, y, z)

        return joint

    def add_truss(self, name, joint_j, joint_k, material, section):
        """Add a truss element to the structure.

        Parameters
        ----------
        name : str
            Name of the truss element to add.
        joint_j : str
            Name of the near joint of the truss element to add.
        joint_k : str
            Name of the far joint of the truss element to add.
        material : str
            Name of the material of the truss element to add.
        section : str
            Name of the section of the truss element to add.

        Returns
        -------
        truss : Truss
            Truss object.
        """
        # create a truss object
        truss = Truss(self, name, joint_j, joint_k, material, section)

        # add the truss object to the dictionary of elements
        self.elements[name] = truss

        return truss


    def add_frame(self, name, joint_j, joint_k, material, section):
        """Add a frame to the structure.

        Parameters
        ----------
        name : str
            Name of the frame to add.
        joint_j : str
            Name of the near joint of the frame to add.
        joint_k : str
            Name of the far joint of the frame to add.
        material : str
            Name of the material of the frame to add.
        section : str
            Name of the section of the frame to add.

        Returns
        -------
        frame : Frame
            Frame object.
        """
        # create a frame object
        frame = Frame(self, name, joint_j, joint_k, material, section)

        # add the frame object to the dictionary of elements
        self.elements[name] = frame

        return frame

    def add_support(self, joint, r_ux=None, r_uy=None, r_uz=None, r_rx=None,
                    r_ry=None, r_rz=None):
        """Add a support to the structure.

        Parameters
        ----------
        joint : str
            Name of the joint of the support to add.
        r_ux : bool, optional
            Flag indicating whether the translation of the joint of the
            support to add along the global x-axis is constrained.
        r_uy : bool, optional
            Flag indicating whether the translation of the joint of the
            support to add along the global y-axis is constrained.
        r_uz : bool, optional
            Flag indicating whether the translation of the joint of the
            support to add along the global z-axis is constrained.
        r_rx : bool, optional
            Flag indicating whether the rotation of the joint of the
            support to add around the global x-axis is constrained.
        r_ry : bool, optional
            Flag indicating whether the rotation of the joint of the
            support to add around the global y-axis is constrained.
        r_rz : bool, optional
            Flag indicating whether the rotation of the joint of the
            support to add around the global z-axis is constrained.

        Returns
        -------
        support : Support
            Support object.
        """
        # create a support object
        support = Support(self, joint, r_ux, r_uy, r_uz, r_rx, r_ry, r_rz)

        # add the support object to the dictionary of supports
        self.supports[joint] = support

        return support

    def add_load_pattern(self, name):
        """Add a load pattern to the structure.

        Parameters
        ----------
        name : str
            Name of the load pattern to add.

        Returns
        -------
        loadPattern : LoadPattern.
            LoadPattern object.
        """
        # add a load pattern to the dictionary of load patterns
        loadPattern = self.load_patterns[name] = LoadPattern(self, name)

        return loadPattern

    def add_point_load_at_joint(self, load_pattern, joint, fx=None, fy=None,
                                fz=None, mx=None, my=None, mz=None):
        """Add a point load at joint to the structure.

        Parameters
        ----------
        load_pattern : str
            Name of the load pattern of the point load at joint to add.
        joint : str
            Name of the joint of the point load at joint to add.
        fx : float, optional
            Force of the point load at joint to add along the global x-axis.
        fy : float
            Force of the point load at joint to add along the global y-axis.
        fz : float
            Force of the point load at joint to add along the global z-axis.
        mx : float
            Force of the point load at joint to add around the global x-axis.
        my : float
            Force of the point load at joint to add around the global y-axis.
        mz : float
            Force of the point load at joint to add around the global z-axis.

        Returns
        -------
        pointLoad : PointLoadAtJoint
            PointLoadAtJoint object.
        """
        # get the load pattern object from the dictionary of load patterns
        loadPattern = self.load_patterns[load_pattern]
        # get the add point load at joint method from the load pattern object
        add_point_load_at_joint = loadPattern.add_point_load_at_joint
        # add a point load at joint to the dictionary of point loads at joints
        pointLoad = add_point_load_at_joint(joint, fx, fy, fz, mx, my, mz)

        return pointLoad

    # add_point_load_at_frame(load_pattern, frame, [fx, fy, fz, mx, my, mz])
    #     Add a point load at frame to a load pattern of the structure.
    #
    # def add_point_load_at_frame(self, load_pattern, frame,
    #                             fx=None, fy=None, fz=None,
    #                             mx=None, my=None, mz=None):
    #     """Add a point load at frame to a load pattern of the structure.

    #     Parameters
    #     ----------
    #     load_pattern : str
    #         Name of the load pattern of the point load at joint.
    #     frame : str
    #         Name of the frame of the point load at frame.
    #     fx : tuple, optional
    #         Tuple of the force along the global x-axis andhis (value, position).
    #     fy : tuple, optional
    #         (value, position).
    #     fz : tuple, optional
    #         (value, position).
    #     mx : tuple, optional
    #         (value, position).
    #     my : tuple, optional
    #         (value, position).
    #     mz : tuple, optional
    #         (value, position).

    #     Returns
    #     -------
    #     pointLoad : PointLoadAtFrame
    #         PointLoadAtFrame object.
    #     """
    #     loadPattern = self.load_patterns[load_pattern]
    #     add_point_load_at_frame = load_pattern.add_point_load_at_frame
    #     pointLoad = add_point_load_at_frame(frame, fx, fy, fz, mx, my, mz)

    #     return pointLoad

    def add_uniformly_distributed_load(self, load_pattern, element, wx=None,
                                       wy=None, wz=None):
        """Add a uniformly distributed load at an element of the structure.

        Parameters
        ----------
        load_pattern : str
            Name of the load pattern of the uniformly distributed load at
            element to add.
        element : str
            Name of the element of the uniformly distributed load at element
            to add.
        wx : float, optional
            Force of the uniformly distributed load at element to add along the
            local x-axis.
        wy : float, optional
            Force of the uniformly distributed load at element to add along the
            local y-axis.
        wz : float, optional
            Force of the uniformly distributed load at element to add along the
            local z-axis.

        Returns
        -------
        distributedLoad : DistributedLoad
            DistributedLoad object.
        """
        # get the load pattern object from the dictionary of load patterns
        loadPattern = self.load_patterns[load_pattern]
        # get the add_uniformly_distributed_load method from the load pattern
        addUniformlyDistributedLoad = \
            loadPattern.add_uniformly_distributed_load_at_element
        # add a uniformly distributed load object
        distributedLoad = addUniformlyDistributedLoad(element, wx, wy, wz)

        return distributedLoad

    def get_flags_joint_displacements(self):
        """Returns the array of joint displacement flags.

        Returns
        -------
        ndarray
            Joint displacement flags.
        """
        return self._flags_joint_displacements

    def set_flags_joint_displacements(self):
        """Set the array of joint displacement flags."""

        # types of structures
        type_str = ['3D']

        # set joint displacement flags
        if self.type == '3D':
            self._ux = self._uy = self._uz = \
                self._rx = self._ry = self._rz = True
        else:
            raise ValueError(f"Invalid type of structure '{self.type}'. "
                             f"Choose one from a valid option ('{type_str}')")

        # set the array of joint displacement flags
        self._flags_joint_displacements = np.array(
            [self._ux, self._uy, self._uz, self._rx, self._ry, self._rz])

    def get_joint_indices(self):
        """Returns the joint indices of the structure.

        Returns
        -------
        dict
            Joint indices of the structure.
        """
        return self._joint_indices

    def set_joint_indices(self):
        """Set the joint indices of the structure."""

        # number of joint displacements
        n = 6

        # set the dictionary of joint indices of the structure
        joint_indices = np.arange(n * len(self.joints)).reshape(-1, n)

        self._joint_indices = {joint: indices for joint, indices in
                               zip(self.joints, joint_indices)}

    def get_stiffness_matrix(self):
        """
        Returns the stiffness matrix of the structure.

        Returns
        -------
        ndarray
            Stiffness matrix of the structure.
        """
        # number of joint displacements
        no_joint_displacements = 6

        # joint indices of the structure
        joint_indices = self.get_joint_indices()

        # number of joints and elements
        no_joints = len(self.joints)
        no_elements = len(self.elements)

        # number joint displacements per element
        n = 2 * no_joint_displacements
        # size of the stiffness matrix elements
        n_2 = n**2

        # row and column positions of the elements of the striffness matrices
        # in the stiffness matrix of the structure
        rows = np.empty(no_elements * n_2, dtype=int)
        cols = np.empty(no_elements * n_2, dtype=int)
        # data of the elements of the stiffness matrices
        data = np.empty(no_elements*n_2)

        # assembly the element stiffness matrices
        for i, element in enumerate(self.elements.values()):
            # get the global matrix stiffness of the element
            k_element = element.get_global_stiffness_matrix()

            # get the joint indices of the near and far element joints,...
            element_indices = np.concatenate((joint_indices[element.joint_j],
                                              joint_indices[element.joint_k]))
            # ..., to create a readonly view of the joint indices, with the
            # shape of the element stiffness matrix
            element_indices = np.broadcast_to(element_indices, (n, n))

            # collapse the replicated joint indices of the element to get the
            # rows & columns positions of the elements of the stiffness matrix
            # by columns
            rows[i*n_2:(i+1)*n_2] = element_indices.flatten('F')
            cols[i*n_2:(i+1)*n_2] = element_indices.flatten()  # and by rows
            # collapse the frame matrix stiffness
            data[i*n_2:(i+1)*n_2] = k_element.flatten()

        return coo_matrix((data, (rows, cols)),
                          2*(no_joint_displacements * no_joints,)).toarray()

    def get_stiffness_matrix_modified_by_supports(self):
        """Returns the stiffness matrix of the structure modified by the
        supports.

        Returns
        -------
        ndarray : array
            Stiffness matrix of the structure modified the by supports.
        """
        return self._ksupport

    def set_stiffness_matrix_modified_by_supports(self):
        """Set the stiffness matrix of the structure modified by the
        supports."""
        # flags joint displacements
        flags_joint_displacements = self.get_flags_joint_displacements()

        # number of joints
        no_joints = len(self.joints)

        # joint indices of the structure
        joint_indices = self.get_joint_indices()

        # stiffness matrix of the structure
        stiff_matrix = self.get_stiffness_matrix()
        # size of the stiffness matrix of the structure
        n = np.shape(stiff_matrix)[0]

        # modify the stiffness matrix of the structure by the supports
        for joint, support in self.supports.items():
            # joint indices of the support
            indices = joint_indices[joint]
            # restrains of the support
            restraints = support.get_restraints()

            # modify the stiffness matrix of the structure
            for index in indices[restraints]:
                stiff_matrix[index] = stiff_matrix[:, index] = np.zeros(n)
                stiff_matrix[index, index] = 1

        # modify the stiffness matrix of the structure by the analysis options
        flags_displacements = np.tile(flags_joint_displacements, no_joints)

        for index in (flags_displacements == False).nonzero()[0]:
            stiff_matrix[index] = stiff_matrix[:, index] = np.zeros(n)
            stiff_matrix[index, index] = 1

        # save the stiffness matrix of the structure modified by the supports
        self._ksupport = stiff_matrix

    def solve_load_pattern(self, load_pattern):
        """
        Solve the structure subjected to a load pattern.

        Parameters
        ----------
        load_pattern : str
            Load pattern name.
        """
        # number displacements per joint
        n = 6
        # flags of the joint displacements analyzed
        flags_joint_displacements = \
            self.get_flags_joint_displacements()

        # number of joints
        no_joints = len(self.joints)
        # joint indices of the structure
        joint_indices = self.get_joint_indices()

        # load pattern object
        loadPattern = self.load_patterns[load_pattern]

        f = loadPattern.get_f()
        f_support = np.copy(f)

        for joint, support in self.supports.items():
            indices = joint_indices[joint]
            restrains = support.get_restraints()

            for index in indices[restrains]:
                f_support[index, 0] = 0

        # modify the vector force by the analysis options
        flags_displacements = np.tile(flags_joint_displacements, no_joints)

        for index in (flags_displacements == False).nonzero()[0]:
            f_support[index, 0] = 0

        # find displacements
        u = np.linalg.solve(
            self.get_stiffness_matrix_modified_by_supports(), f_support)

        # store displacements
        load_pattern_displacements = {}

        for joint in self.joints:
            indices = joint_indices[joint]
            load_pattern_displacements[joint] = Displacements(
                self, load_pattern, joint, *u[indices, 0])

        self.displacements[load_pattern] = load_pattern_displacements

        # store frame end actions
        rows = []
        cols = []
        data = []

        load_pattern_end_actions = {}

        for key, element in self.elements.items():
            indices_element = np.concatenate((joint_indices[element.joint_j],
                                              joint_indices[element.joint_k]))

            t = element.get_rotation_transformation_matrix()
            k_element = element.get_local_stiffness_matrix()

            u_element = u[indices_element]
            u_element = np.dot(np.transpose(t), u_element)
            f_fixed = np.zeros((12, 1))

            if key in loadPattern.uniformly_distributed_loads_at_elements:
                for distributed_load in \
                    loadPattern.uniformly_distributed_loads_at_elements[key]:
                    f_fixed += distributed_load.get_f_fixed()

            f_end_actions = np.dot(k_element, u_element).flatten()
            f_end_actions += np.dot(np.transpose(t), f_fixed).flatten()
            load_pattern_end_actions[key] = EndActions(
                self, load_pattern, key, *f_end_actions)

            # reactions
            if element.joint_j in self.supports or element.joint_k in self.supports:
                rows.extend(indices_element)
                cols.extend(len(indices_element) * [0])
                data.extend(
                    np.dot(t, load_pattern_end_actions[key].get_end_actions()).flatten())

        self.end_actions[load_pattern] = load_pattern_end_actions

        # store reactions
        f += loadPattern.get_f_fixed()
        f_end_actions = coo_matrix(
            (data, (rows, cols)), (no_joints * n, 1)).toarray()

        load_pattern_reactions = {}

        for joint in self.supports:
            indices = joint_indices[joint]
            reactions = f_end_actions[indices] - f[indices]
            load_pattern_reactions[joint] = Reaction(
                self, load_pattern, joint, *reactions[:, 0])

        self.reactions[load_pattern] = load_pattern_reactions

        # store internal forces
        load_pattern_internal_forces = {}

        for key, element in self.elements.items():
            load_pattern_internal_forces[key] = InternalForces(
                self, load_pattern, key, **element.get_internal_forces(load_pattern))

        self.internal_forces[load_pattern] = load_pattern_internal_forces

        # store internal displacements
        load_pattern_internal_displacements = {}

        for key, element in self.elements.items():
            load_pattern_internal_displacements[key] = InternalDisplacements(
                self, load_pattern, key, **element.get_internal_displacements(load_pattern))

        self.internal_displacements[load_pattern] = load_pattern_internal_displacements

    def solve(self):
        """Solve the structure."""
        # set flags joint displacements
        self.set_flags_joint_displacements()

        # set joint indices of the structure
        self.set_joint_indices()

        # set the stiffness matrix of the structure modified by the supports
        self.set_stiffness_matrix_modified_by_supports()

        # solve the structure due to each load pattern
        for load_pattern in self.load_patterns:
            self.solve_load_pattern(load_pattern)

    def export(self, filename):
        """
        Save the structure to a file in json format.

        Parameters
        ----------
        filename : string
            Filename
        """
        data = {}

        # save the materials
        if self.materials:
            data['materials'] = {}
            for key, material in self.materials.items():
                data['materials'][key] = {attr: value for attr, value in material.__dict__.items(
                ) if not attr.startswith('_') and value is not None}

        # save the sections
        if self.sections:
            data['sections'] = {}
            for key, section in self.sections.items():
                data['sections'][key] = {'type': section.__class__.__name__}
                data['sections'][key].update({attr: value for attr, value in section.__dict__.items(
                ) if not attr.startswith('_') and value is not None})

        # save the joints
        if self.joints:
            data['joints'] = {}
            for key, joint in self.joints.items():
                data['joints'][key] = {attr: value for attr, value in joint.__dict__.items(
                ) if not attr.startswith('_') and value is not None}

        # save the frames
        if self.elements:
            data['frames'] = {}
            for key, element in self.elements.items():
                data['frames'][key] = {attr: value for attr, value in element.__dict__.items(
                ) if not attr.startswith('_') and value is not None}

        # save the supports
        if self.supports:
            data['supports'] = {}
            for key, support in self.supports.items():
                data['supports'][key] = {attr: value for attr, value in support.__dict__.items(
                ) if not attr.startswith('_') and value is not None}

        # save the load patterns
        if self.load_patterns:
            data['load_patterns'] = {}
            for key, loadPattern in self.load_patterns.items():
                data['load_patterns'][key] = {'name': loadPattern.name}

                # save loads at joints
                if loadPattern.point_loads_at_joints:
                    data['load_patterns'][key]['joints'] = {}
                    for _joint, point_loads in loadPattern.loads_at_joints.items():
                        data['load_patterns'][key]['joints'][_joint] = []
                        for pointLoad in point_loads:
                            data['load_patterns'][key]['joints'][_joint].append(
                                {attr: value for attr, value in pointLoad.__dict__.items() if not attr.startswith('_') and value is not None})

                # save loads at frames
                if loadPattern.uniformly_distributed_loads_at_elements:  # loadPattern.point_loads_at_frames or
                    data['load_patterns'][key]['frames'] = {}

                    for _frame, distributed_loads in loadPattern.uniformly_distributed_loads_at_elements.items():
                        if not _frame in data['load_patterns'][key]['frames']:
                            data['load_patterns'][key]['frames'][_frame] = []
                        for distributedLoad in distributed_loads:
                            _data = {
                                'type': distributedLoad.__class__.__name__}
                            _data.update({attr: value for attr, value in distributedLoad.__dict__.items(
                            ) if not attr.startswith('_') and value is not None})
                            data['load_patterns'][key]['frames'][_frame].append(
                                _data)

        # save displacements
        if self.displacements:
            data['displacements'] = {}
            for key, displacements in self.displacements.items():
                data['displacements'][key] = {}
                for joint, displacement in displacements.items():
                    data['displacements'][key][joint] = {attr: value for attr, value in displacement.__dict__.items(
                    ) if not attr.startswith('_') and value is not None}

        # save reactions
        if self.reactions:
            data['reactions'] = {}
            for key, reactions in self.reactions.items():
                data['reactions'][key] = {}
                for joint, reaction in reactions.items():
                    data['reactions'][key][joint] = {attr: value for attr, value in reaction.__dict__.items(
                    ) if not attr.startswith('_') and value is not None}

        # save end actions
        if self.end_actions:
            data['end_actions'] = {}
            for key, end_actions in self.end_actions.items():
                data['end_actions'][key] = {}
                for element, end_action in end_actions.items():
                    data['end_actions'][key][element] = {attr: value for attr, value in end_action.__dict__.items(
                    ) if not attr.startswith('_') and value is not None}

        # save internal forces
        if self.internal_forces:
            data['internal_forces'] = {}
            for key, internal_forces in self.internal_forces.items():
                data['internal_forces'][key] = {}
                for element, internal_force in internal_forces.items():
                    data['internal_forces'][key][element] = {attr: value for attr, value in internal_force.__dict__.items(
                    ) if not attr.startswith('_') and value is not None}

        # save internal displacements
        if self.internal_displacements:
            data['internal_displacements'] = {}
            for key, internal_displacements in self.internal_displacements.items():
                data['internal_displacements'][key] = {}
                for element, internal_displacement in internal_displacements.items():
                    data['internal_displacements'][key][element] = {
                        attr: value for attr, value in internal_displacement.__dict__.items() if not attr.startswith('_') and value is not None}

        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)
