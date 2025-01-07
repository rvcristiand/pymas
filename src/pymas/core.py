import json
import numpy as np

from scipy.sparse import coo_matrix
from pymas.primitives import *


class Structure:
    """Model and analyse linear framed structures subjected to static loads.

    The available type of structures are:

    - '3D'       : 6 degrees of freedom (ux, uy, uz, rx, ry, rz) applicable for
                   a general tridimensional structural analysis.
    - '3D truss' : 3 degrees of freedom (ux, uy, uz) applicable for the
                   analysis of 3D trusses.
    - 'beam'     : 2 degrees of freedom (uy, rz) applicable for the analysis of
                   beams.


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
    add_material(name, [modulus_elasticity, modulus_elasticity_shear])
        Add a material to the model.
    add_section(name, [area, torsion_constant, inertia_y, inertia_z])
        Add a cross section to the model.
    add_rectangular_section(name, base, height)
        Add a rectangular cross section to the model.
    add_joint(name, [x, y, z])
        Add a joint to the model.
    add_truss(name, joint_j, joint_k, material, section)
        Add a truss element to the model.
    add_frame(name, joint_j, joint_k, material, section)
        Add a frame element to the model.
    add_support(joint, [r_ux, r_uy, r_uz, r_rx, r_ry, r_rz])
        Add a support to a joint of the model.
    add_load_pattern(name)
        Add a load pattern to the model.
    add_point_load(load_pattern, joint, [fx, fy, fz, mx, my, mz])
        Add a point load to a joint of the model.
    add_distributed_load(load_pattern, element, [fx, fy, fz, mx, my, mz])
        Add a uniformly distributed load to a element of the model.
    get_flags_degrees_freedom()
        Returns the flags of degrees of freedom of the model.
    set_flags_degrees_freedom()
        Set the flags of degrees of freedom of the the model.
    number_active_degrees_freedom()
        Get the number of active degrees of freedom of the model.
    get_joint_indices()
        Returns the joint indices of the model.
    set_joint_indices()
        Set the joint indices of the model.
    get_stiffness_matrix()
        Returns the stiffness matrix of the model.
    set_stiffness_matrix()
        Set the stiffness matrix of the model.
    analyse_load_pattern(load_pattern)
        Analyse the model subjected to a load pattern.
    run_analysis()
        Analyse the model subjected to the load patterns.
    export(filename)
        Export the model.
    """

    def __init__(self, type='3D'):
        """Instantiate a Structure object.

        Parameters
        ----------
        type : str
            Type of structure.
        """
        # initialize the internal variables

        # flags of degrees of freedom of the structure
        self._ux = None
        self._uy = None
        self._uz = None
        self._rx = None
        self._ry = None
        self._rz = None
        # array of flags of degrees of freedom of the structure
        self._flags_dof = np.array([])
        # dictionary of joint indices the structure
        self._joint_indices = {}
        # matrix stiffness of the structure
        self._k = np.array([])

        # type of the structure
        self.type = type

        # material and section dictionaries
        self.materials = {}
        self.sections = {}

        # joint and element dictionaries
        self.joints = {}
        self.elements = {}

        # support and load pattern dictionaries
        self.supports = {}
        self.load_patterns = {}

        # joint displacements, element end actions and support reactions dicts
        self.displacements = {}
        self.end_actions = {}
        self.reactions = {}

        # internal element forces and displacements dictionaries
        self.internal_forces = {}
        self.internal_displacements = {}

    def add_material(self, name, modulus_elasticity=None,
                     modulus_elasticity_shear=None):
        """Add a material to the model.

        Parameters
        ----------
        name : str
            Name of the material.
        modulus_elasticity : float, optional
            Modulus of elasticity of the material.
        modulus_elasticity_shear : float, optional
            Modulus of elasticity in shear of the material.

        Returns
        -------
        material : Material
            Material.
        """
        # material properties
        E = modulus_elasticity
        G = modulus_elasticity_shear

        # add a material object to the dictionary of materials
        material = self.materials[name] = Material(self, name, E, G)

        return material

    def add_section(self, name, area=None, torsion_constant=None,
                    inertia_y=None, inertia_z=None):
        """Add a cross section to the model.

        Parameters
        ----------
        name : str
            Name of the cross section.
        area : float, optional
            Area of the cross section.
        torsion_constant : float, optional
            Torsion constant of the cross section.
        inertia_y : float, optional
            Moment of inertia of the cross section with respect to the
            local y-axis.
        inertia_z : float, optional
            Moment of inertia of the cross section with respect to the
            local z-axis.

        Returns
        -------
        section : Section
            Cross section.
        """
        # cross section properties
        A = area
        J = torsion_constant
        Iy = inertia_y
        Iz = inertia_z

        # add a section object to the dictionary of cross sections
        section = self.sections[name] = Section(self, name, A, J, Iy, Iz)

        return section

    def add_rectangular_section(self, name, base, height):
        """Add a rectangular cross section to the model.

        Parameters
        ----------
        name : str
            Name of the rectangular cross section.
        base : float
            Base of the rectangular cross section.
        height : float
            Height of the rectangular cross section.

        Returns
        -------
        rect_sect : RectangularSection
            Rectangular cross section.
        """
        # create a rectangular cross section object
        rect_sect = RectangularSection(self, name, base, height)

        # add the rect cross section object to the dict of cross sections
        self.sections[name] = rect_sect

        return rect_sect

    def add_joint(self, name, x=None, y=None, z=None):
        """Add a joint to the model.

        Parameters
        ----------
        name : str
            Name of the joint.
        x : float, optional
            Coordinate X of the joint.
        y : float, optional
            Coordinate Y of the joint.
        z : float, optional
            Coordinate Z of the joint.

        Returns
        -------
        joint : Joint
            Joint.
        """
        # add a joint object to the dictionary of joints
        joint = self.joints[name] = Joint(self, name, x, y, z)

        return joint

    def add_truss(self, name, joint_j, joint_k, material, section):
        """Add a truss element to the model.

        Parameters
        ----------
        name : str
            Name of the truss.
        joint_j : str
            Name of the near joint of the truss.
        joint_k : str
            Name of the far joint of the truss.
        material : str
            Name of the material of the truss.
        section : str
            Name of the section of the truss.

        Returns
        -------
        truss : Truss
            Truss element.
        """
        # create a truss object
        truss = Truss(self, name, joint_j, joint_k, material, section)

        # add the truss object to the dictionary of elements
        self.elements[name] = truss

        return truss

    def add_frame(self, name, joint_j, joint_k, material, section):
        """Add a frame element to the model.

        Parameters
        ----------
        name : str
            Name of the frame.
        joint_j : str
            Name of the near joint of the frame.
        joint_k : str
            Name of the far joint of the frame.
        material : str
            Name of the material of the frame.
        section : str
            Name of the section of the frame.

        Returns
        -------
        element : Frame
            Frame element.
        """
        # create a frame object
        element = Frame(self, name, joint_j, joint_k, material, section)

        # add the frame object to the dictionary of elements
        self.elements[name] = element

        return element

    def add_support(self, joint, r_ux=None, r_uy=None, r_uz=None, r_rx=None,
                    r_ry=None, r_rz=None):
        """Add a support to a joint of the model.

        Parameters
        ----------
        joint : str
            Name of the joint.
        r_ux : bool, optional
            Indicates whether the support restrains the displacement of the
            joint along the global x-axis.
        r_uy : bool, optional
            Indicates whether the support restrains the displacement of the
            joint along the global y-axis.
        r_uz : bool, optional
            Indicates whether the support restrains the displacement of the
            joint along the global z-axis.
        r_rx : bool, optional
            Indicates whether the support restrains the displacement of the
            joint around the global x-axis.
        r_ry : bool, optional
            Indicates whether the support restrains the displacement of the
            joint around the global y-axis.
        r_rz : bool, optional
            Indicates whether the support restrains the displacement of the
            joint around the global z-axis.

        Returns
        -------
        support : Support
            Support.
        """
        # create a support object
        support = Support(self, joint, r_ux, r_uy, r_uz, r_rx, r_ry, r_rz)

        # add the support object to the dictionary of supports
        self.supports[joint] = support

        return support

    def add_load_pattern(self, name):
        """Add a load pattern to the model.

        Parameters
        ----------
        name : str
            Name of the load pattern.

        Returns
        -------
        loadPattern : LoadPattern.
            Load pattern.
        """
        # add a load pattern to the dictionary of load patterns
        loadPattern = self.load_patterns[name] = LoadPattern(self, name)

        return loadPattern

    def add_point_load(self, load_pattern, joint, fx=None, fy=None, fz=None,
                       mx=None, my=None, mz=None):
        """Add a point load to a joint of the model.

        Parameters
        ----------
        load_pattern : str
            Name of the load pattern.
        joint : str
            Name of the joint.
        fx : float, optional
            Intensity of the point load along the global x-axis.
        fy : float, optional
            Intensity of the point load along the global y-axis.
        fz : float, optional
            Intensity of the point load along the global z-axis.
        mx : float, optional
            Intensity of the point load around the global x-axis.
        my : float, optional
            Intensity of the point load around the global y-axis.
        mz : float, optional
            Intensity of the point load around the global z-axis.

        Returns
        -------
        pointLoad : PointLoad
            Point load.
        """
        # get the load pattern object from the dictionary of load patterns
        loadPattern = self.load_patterns[load_pattern]
        # add the point load to the load pattern
        pointLoad = loadPattern.add_point_load(joint, fx, fy, fz, mx, my, mz)

        return pointLoad

    def add_distributed_load(self, load_pattern, element, fx=None, fy=None,
                             fz=None, mx=None, my=None, mz=None):
        """Add a uniformly distributed load to an element of the model.

        Parameters
        ----------
        load_pattern : str
            Name of the load pattern.
        element : str
            Name of the element.
        fx : float, optional
            Intensity of the distributed load along the local x-axis.
        fy : float, optional
            Intensity of the distributed load along the local y-axis.
        fz : float, optional
            Intensity of the distributed load along the local z-axis.
        mx : float, optional
            Intensity of the distributed load around the local x-axis.
        my : float, optional
            Intensity of the distributed load around the local y-axis.
        mz : float, optional
            Intensity of the distributed load around the local z-axis.
        
        Returns
        -------
        distributedLoad : DistributedLoad
            Uniformly distributed load.
        """
        # get the load pattern object from the dictionary of load patterns
        lPattern = self.load_patterns[load_pattern]
        # add the uniformly distributed load to the load pattern
        dLoad = lPattern.add_distributed_load(element, fx, fy, fz, mx, my, mz)

        return dLoad

    def get_flags_degrees_freedom(self):
        """Returns the flags of degrees of freedom of the model.

        Returns
        -------
        ndarray
            Flags of degrees of freedom.
        """
        return self._flags_dof

    def set_flags_degrees_freedom(self):
        """Set the flags of degrees of freedom of the model."""

        # types of structures
        type_str = ['3D', 'beam']

        # set the flags of degrees of freedom
        if self.type == '3D':
            self._ux = self._uy = self._uz = True
            self._rx = self._ry = self._rz = True
        elif self.type == '3D truss':
            self._ux = self._uy = self._uz = True
            self._rx = self._ry = self._rz = False
        elif self.type == 'beam':
            self._uy = self._rz = True
            self._ux = self._uz = self._rx = self._ry = False
        else:
            raise ValueError(f"Invalid type of structure '{self.type}'. "
                             f"Choose one from a valid option ('{type_str}')")

        # set the array of degrees of freedom of the model
        self._flags_dof = np.array([self._ux, self._uy, self._uz,
                                self._rx, self._ry, self._rz])

    def number_active_degrees_freedom(self):
        """
        Get the number of active degrees of freedom of the model.

        Returns
        -------
        int
            Number of active degrees of freedom.
        """
        return np.count_nonzero(self.get_flags_degrees_freedom())

    def get_joint_indices(self):
        """Returns the joint indices of the model.

        Returns
        -------
        dict
            Joint indices of the model.
        """
        return self._joint_indices

    def set_joint_indices(self):
        """Set the joint indices of the structure."""
        # number of joints
        n_j = len(self.joints)
        # number of active degrees of freedom
        n_dof = self.number_active_degrees_freedom()
        # create the joint indices of the structure
        j_indices = np.arange(n_j * n_dof).reshape(n_j, n_dof)

        # set the dictionary of joint indices
        self._joint_indices = {j: i for j, i in zip(self.joints, j_indices)}

    def get_stiffness_matrix(self):
        """Returns the stiffness matrix of the model.

        Returns
        -------
        ndarray
            Stiffness matrix.
        """
        return self._k

    def set_stiffness_matrix(self):
        """Set the stiffness matrix of the structure."""
        # number of joints
        n_j = len(self.joints)
        # number of elements
        n_e = len(self.elements)

        # flags of degrees of freedom
        flags_dof = self.get_flags_degrees_freedom()

        # number of active degrees of freedom
        n_dof = self.number_active_degrees_freedom()
        # number of degrees of freedom per element
        n_dof_e = 2 * n_dof
        # number of degrees of freedom of the structure 
        n_dof_s = n_j * n_dof

        # joint indices
        j_i = self.get_joint_indices()

        # number of items of the stiffness matrix of the elements
        n = n_dof_e**2

        # row and column positions of the striffness matrices items
        # of the elements in the stiffness matrix of the structure
        rows = np.empty(n_e * n, dtype=int)
        cols = np.empty_like(rows)
        # data of the stiffness matrices of the frames
        data = np.empty_like(rows, dtype=float)

        # assembly the element stiffness matrices
        for i, elem in enumerate(self.elements.values()):
            # start and end indices
            start = i * n
            end = (i + 1) * n
            # get the joint indices of the near and far element joints,...
            e_i = np.concatenate((j_i[elem.joint_j], j_i[elem.joint_k]))
            # ..., to create a readonly view of the joint indices, with the
            # shape of the element stiffness matrix
            e_i = np.broadcast_to(e_i, 2 * (n_dof_e,))
            # get the global matrix stiffness of the element
            k_e = elem.get_global_stiffness_matrix()

            # collapse the replicated joint indices to get the rows
            # and columns positions of the elements of the stiffness matrix
            rows[start:end] = e_i.flatten('F')
            cols[start:end] = e_i.flatten()
            # collapse the element matrix stiffness
            data[start:end] = k_e.flatten()

        # create the stiffness matrix of the structure
        k_s = coo_matrix((data, (rows, cols)), 2 * (n_dof_s, )).toarray()

        # modify the stiffness matrix of the structure by the supports
        for joint, support in self.supports.items():
            # joint indices of the support
            indices = j_i[joint]

            # restrains of the support
            restraints = support.get_restraints()

            # modify the stiffness matrix of the structure
            for index in indices[restraints]:
                k_s[index] = np.zeros(n_dof_s)
                k_s[:, index] = np.zeros(n_dof_s)
                k_s[index, index] = 1

        # set the stiffness matrix of the structure
        self._k = k_s

    def analyse_load_pattern(self, load_pattern):
        """
        Analyse the structure subjected to a load pattern.

        Parameters
        ----------
        load_pattern : str
            Load pattern name.
        """
        # flags of degrees of freedom
        flags_dof = self.get_flags_degrees_freedom()

        # number of joints
        n_j = len(self.joints)
        # joint indices of the structure
        j_i = self.get_joint_indices()

        # load pattern object
        loadPattern = self.load_patterns[load_pattern]

        f = loadPattern.get_f()
        f_support = np.copy(f)

        for j, support in self.supports.items():
            indices = j_i[j]
            restrains = support.get_restraints()

            for index in indices[restrains]:
                f_support[index, 0] = 0

        # find displacements
        u = np.linalg.solve(
            self.get_stiffness_matrix(), f_support)

        # store displacements
        l_p_d = {}

        for j in self.joints:
            indices = j_i[j]

            displacement = np.array(6 * [None])
            displacement[flags_dof] = u[indices, 0]

            l_p_d[j] = Displacement(self, load_pattern, j, *displacement)

        self.displacements[load_pattern] = l_p_d

        # store element end actions
        flags_dof_elem = np.tile(flags_dof, 2)
        n = np.count_nonzero(flags_dof_elem)
        rows = []
        cols = []
        data = []

        load_pattern_end_actions = {}

        for key, elem in self.elements.items():
            i_element = np.concatenate((j_i[elem.joint_j], j_i[elem.joint_k]))

            k_element = elem.get_local_stiffness_matrix()
            t = elem.get_rotation_transformation_matrix()

            u_element = np.zeros((12, 1))
            u_element[flags_dof_elem] = u[i_element]
            u_element = np.dot(np.transpose(t), u_element)
            f_fixed = np.zeros((12, 1))

            if key in loadPattern.distributed_loads:
                for distributed_load in loadPattern.distributed_loads[key]:
                    f_fixed[flags_dof_elem] += distributed_load.get_f_fixed()

            f_end_actions = np.dot(k_element, u_element).flatten()
            f_end_actions += np.dot(np.transpose(t), f_fixed).flatten()
            f_end_actions = f_end_actions.tolist()

            load_pattern_end_actions[key] = EndActions(
                self, load_pattern, key, *f_end_actions)

            # reactions
            if elem.joint_j in self.supports or elem.joint_k in self.supports:

                rows.extend(i_element.tolist())
                cols.extend(len(i_element) * [0])
                data.extend(
                    np.dot(t, load_pattern_end_actions[key].get_end_actions()).flatten()[flags_dof_elem].tolist())

        self.end_actions[load_pattern] = load_pattern_end_actions

        # store reactions
        f += loadPattern.get_f_fixed()
        f_end_actions = coo_matrix(
            (data, (rows, cols)), (n_j * n, 1)).toarray()

        load_pattern_reactions = {}

        for j in self.supports:
            indices = j_i[j]
            reactions = f_end_actions[indices] - f[indices]
            load_pattern_reactions[j] = Reaction(
                self, load_pattern, j, *reactions[:, 0])

        self.reactions[load_pattern] = load_pattern_reactions

        # store internal forces
        load_pattern_internal_forces = {}

        for key, elem in self.elements.items():
            load_pattern_internal_forces[key] = InternalForces(
                self, load_pattern, key, **elem.get_internal_forces(load_pattern))

        self.internal_forces[load_pattern] = load_pattern_internal_forces

        # store internal displacements
        load_pattern_internal_displacements = {}

        for key, elem in self.elements.items():
            load_pattern_internal_displacements[key] = InternalDisplacements(
                self, load_pattern, key, **elem.get_internal_displacements(load_pattern))

        self.internal_displacements[load_pattern] = load_pattern_internal_displacements

    def run_analysis(self):
        """Analyse the structure subjected to the load patterns."""
        # set the flags of the degrees of freedom
        self.set_flags_degrees_freedom()

        # set joint indices
        self.set_joint_indices()

        # set the stiffness matrix of the structure
        self.set_stiffness_matrix()

        # solve the structure due to each load pattern
        for load_pattern in self.load_patterns:
            self.analyse_load_pattern(load_pattern)

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
                if loadPattern.point_loads:
                    data['load_patterns'][key]['joints'] = {}
                    for _joint, point_loads in loadPattern.point_loads.items():
                        data['load_patterns'][key]['joints'][_joint] = []
                        for pointLoad in point_loads:
                            data['load_patterns'][key]['joints'][_joint].append(
                                {attr: value for attr, value in pointLoad.__dict__.items() if not attr.startswith('_') and value is not None})

                # save loads at frames
                if loadPattern.distributed_loads:  # loadPattern.point_loads_at_frames or
                    data['load_patterns'][key]['frames'] = {}

                    for _frame, distributed_loads in loadPattern.distributed_loads.items():
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
