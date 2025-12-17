import json
import numpy as np
from scipy.sparse import coo_matrix
from pymas.primitives import *


class Structure:
    """Models and analyzes linear framed structures subjected to static loads.

    This class provides functionality to define, analyze, and export structural
    models. It supports various structure types, including 3D general analysis,
    3D trusses, and beams, each with specific degrees of freedom. Users can add
    materials, cross sections, joints, elements (trusses and frames), supports,
    and load patterns to build a complete structural model.

    The available types of structures are:

    - '3D': 6 degrees of freedom (ux, uy, uz, rx, ry, rz) for general 3D analysis.
    - '3D truss': 3 degrees of freedom (ux, uy, uz) for 3D trusses.
    - 'beam': 2 degrees of freedom (uy, rz) for beam analysis.

    Note: Other structure types (plane truss, plane element, plane grid, etc.)
    may be implemented in the future.

    Attributes:
        type (str): Structure type of the model.
        materials (dict): Materials of the model.
        sections (dict): Cross sections of the model.
        joints (dict): Joints of the model.
        elements (dict): Elements of the model.
        supports (dict): Joint supports of the model.
        load_patterns (dict): Load patterns of the model.
        displacements (dict): Joint displacements of the model.
        end_actions (dict): Element end actions of the model.
        reactions (dict): Joint support reactions of the model.
        internal_forces (dict): Internal element forces of the model.
        internal_displacements (dict): Internal element displacements of the model.
    """

    def __init__(self, type='3D'):
        """Instantiates a Structure object.

        Initializes a new structural model with a specified type and sets up
        internal variables and empty dictionaries for various structural
        components and analysis results.

        Args:
            type (str, optional): The structure type.
                Defaults to '3D'.
        """
        # initialize the internal variables

        # degrees of freedom
        self._ux = None
        self._uy = None
        self._uz = None
        self._rx = None
        self._ry = None
        self._rz = None
        # array degrees of freedom
        self._dof = np.array([])
        # dictionary of joint indices
        self._j_i = {}
        # matrix stiffness of the structure
        self._k = np.array([])

        # type of structure
        self.type = type

        # material and cross section dictionaries
        self.materials = {}
        self.sections = {}

        # joint and element dictionaries
        self.joints = {}
        self.elements = {}

        # joint support and load pattern dictionaries
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

        Creates and adds a new material object to the model's material dictionary.

        Args:
            name (str): Name of the material.
            modulus_elasticity (float, optional): Elastic modulus of the material.
            modulus_elasticity_shear (float, optional): Shear modulus of the material.

        Returns:
            Material: Material object.
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

        Creates and adds a new generic cross section object to the model's section dictionary.

        Args:
            name (str): Name of the cross section.
            area (float, optional): Area of the cross section.
            torsion_constant (float, optional): Torsion constant of the cross section.
            inertia_y (float, optional): Inertia of the cross section with respect to the local y-axis.
            inertia_z (float, optional): Inertia of the cross section with respect to the local z-axis.

        Returns:
            Section: Cross section.
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

        Creates and adds a new rectangular cross section object to the model's section dictionary,
        calculating its area, torsion constant, and moments of inertia based on base and height.

        Args:
            name (str): Name of the rectangular cross section.
            base (float): Base of the rectangular cross section.
            height (float): Height of the rectangular cross section.

        Returns:
            RectangularSection: Rectangular cross section.
        """
        # create a rectangular cross section object
        rect_sect = RectangularSection(self, name, base, height)

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

    def add_truss(self, name, joint_j, joint_k, material, section):
        """Add a truss element to the model.

        Creates and adds a new truss element object to the model's element dictionary, connecting two
        specified joints with a given material and cross section.

        Args:
            name (str): Name of the truss.
            joint_j (str): Name of the near joint of the truss.
            joint_k (str): Name of the far joint of the truss.
            material (str): Name of the material of the truss.
            section (str): Name of the section of the truss.

        Returns:
            Truss: Truss element.
        """
        # create a truss object
        truss = Truss(self, name, joint_j, joint_k, material, section)

        # add the truss object to the dictionary of elements
        self.elements[name] = truss

        return truss

    def add_frame(self, name, joint_j, joint_k, material, section):
        """Add a frame element to the model.

        Creates and adds a new frame element object to the model's element dictionary, connecting two
        specified joints with a given material and cross section.

        Args:
            name (str): Name of the frame.
            joint_j (str): Name of the near joint of the frame.
            joint_k (str): Name of the far joint of the frame.
            material (str): Name of the material of the frame.
            section (str): Name of the section of the frame.

        Returns:
            Frame: Frame element.
        """
        # create a frame object
        element = Frame(self, name, joint_j, joint_k, material, section)

        # add the frame object to the dictionary of elements
        self.elements[name] = element

        return element

    def add_support(self, joint, r_ux=None, r_uy=None, r_uz=None, r_rx=None,
                    r_ry=None, r_rz=None):
        """Add a joint support (restrain) to the model.

        Creates and adds a new support object to the model's support dictionary for a specified joint,
        defining which degrees of freedom are restrained.

        Args:
            joint (str): Name of the joint.
            r_ux (bool, optional): Whether the global x-axis translation is restrained.
            r_uy (bool, optional): Whether the global y-axis translation is restrained.
            r_uz (bool, optional): Whether the global z-axis translation is restrained.
            r_rx (bool, optional): Whether the global x-axis rotation is restrained.
            r_ry (bool, optional): Whether the global y-axis rotation is restrained.
            r_rz (bool, optional): Whether the global z-axis rotation is restrained.

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

        Creates and adds a new load pattern object to the model's load pattern dictionary.

        Args:
            name (str): Name of the load pattern.

        Returns:
            LoadPattern: Load pattern.
        """
        # add a load pattern object to the dictionary of load patterns
        loadPattern = self.load_patterns[name] = LoadPattern(self, name)

        return loadPattern

    def add_joint_point_load(self, load_pattern, joint, fx=None, fy=None,
                             fz=None, mx=None, my=None, mz=None):
        """Add a joint point load to the model.

        Adds a point load or moment to a specific joint as part of a given load pattern.

        Args:
            load_pattern (str): Name of the load pattern.
            joint (str): Name of the joint.
            fx (float, optional): Intensity of the point load along the global x-axis.
            fy (float, optional): Intensity of the point load along the global y-axis.
            fz (float, optional): Intensity of the point load along the global z-axis.
            mx (float, optional): Intensity of the point load around the global x-axis.
            my (float, optional): Intensity of the point load around the global y-axis.
            mz (float, optional): Intensity of the point load around the global z-axis.

        Returns:
            PointLoad: Point load.
        """
        # get the load pattern object from the dictionary of load patterns
        lP = self.load_patterns[load_pattern]
        # add the point load to the load pattern
        pointLoad = lP.add_joint_point_load(joint, fx, fy, fz, mx, my, mz)

        return pointLoad

    def add_element_point_load(self, load_pattern, element, dist, fx=None,
                               fy=None, fz=None, mx=None, my=None, mz=None):
        """Add an element point load to the model.

        Adds a point load or moment to a specific element at a given distance from its near joint, as part of
        a given load pattern.

        Args:
            load_pattern (str): Name of the load pattern.
            element (str): Name of the element.
            dist (float): Distance of the point load from the near joint.
            fx (float, optional): Intensity of the point load along the local x-axis.
            fy (float, optional): Intensity of the point load along the local y-axis.
            fz (float, optional): Intensity of the point load along the local z-axis.
            mx (float, optional): Intensity of the point load around the local x-axis.
            my (float, optional): Intensity of the point load around the local y-axis.
            mz (float, optional): Intensity of the point load around the local z-axis.

        Returns:
            ElementPointLoad: Element point load.
        """
        # get the load pattern object from the dictionary of load patterns
        lP = self.load_patterns[load_pattern]
        # add the point load to the load pattern
        pL = lP.add_element_point_load(element, dist, fx, fy, fz, mx, my, mz)

        return pL

    def add_distributed_load(self, load_pattern, element, fx=None, fy=None,
                             fz=None, mx=None, my=None, mz=None):
        """Add an element uniformly distributed load to the model.

        Adds a uniformly distributed load or moment to a specific element as part of a given load pattern.

        Args:
            load_pattern (str): Name of the load pattern.
            element (str): Name of the element.
            fx (float, optional): Intensity of the distributed load along the local x-axis.
            fy (float, optional): Intensity of the distributed load along the local y-axis.
            fz (float, optional): Intensity of the distributed load along the local z-axis.
            mx (float, optional): Intensity of the distributed load around the local x-axis.
            my (float, optional): Intensity of the distributed load around the local y-axis.
            mz (float, optional): Intensity of the distributed load around the local z-axis.

        Returns:
            DistributedLoad: Uniformly distributed load.
        """
        # get the load pattern object from the dictionary of load patterns
        lP = self.load_patterns[load_pattern]
        # add the uniformly distributed load to the load pattern
        dL = lP.add_element_distributed_load(element, fx, fy, fz, mx, my, mz)

        return dL

    def get_degrees_freedom(self):
        """Return the degrees of freedom of the model.

        Returns:
            ndarray: Degrees of freedom of the model.
        """
        return self._dof

    def set_degrees_freedom(self):
        """Set the degrees of freedom of the model.

        This method configures the active translational and rotational degrees of freedom (ux, uy, uz, rx,
        ry, rz) according to the `type` attribute of the structure (e.g., '3D', '3D truss', 'beam').

        Raises:
            ValueError: If an invalid structure type is specified.
        """
        # types of structures
        type_str = ['3D', '3D truss', 'beam']

        # set the degrees of freedom
        if self.type == '3D':
            self._ux = self._uy = self._uz = ux = uy = uz = True
            self._rx = self._ry = self._rz = rx = ry = rz = True
        elif self.type == '3D truss':
            self._ux = self._uy = self._uz = ux = uy = uz = True
            self._rx = self._ry = self._rz = rx = ry = rz = False
        elif self.type == 'beam':
            self._uy = self._rz = uy = rz = True
            self._ux = self._uz = ux = uz = False
            self._rx = self._ry = rx = ry = False
        else:
            raise ValueError(f"Invalid type of structure '{self.type}'. "
                             f"Choose one from a valid option ('{type_str}')")

        # set the degrees of freedom of the model
        self._dof = np.array([ux, uy, uz, rx, ry, rz])

    def number_active_degrees_freedom(self):
        """Get the number of active degrees of freedom of the model.

        Returns:
            int: Number of active degrees of freedom of the model.
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

        This method assigns unique, contiguous global indices to each active degree of freedom for every
        joint in the model, storing them in the `_j_i` dictionary.
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

        This method calculates the local stiffness matrices for all elements, transforms them to the global
        coordinate system, and assembles them into the global stiffness matrix of the structure. It also
        applies modifications due to supports.
        """
        # number of joints
        n_j = len(self.joints)
        # number of elements
        n_e = len(self.elements)

        # degrees of freedom
        dof = self.get_degrees_freedom()

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
            k_e = elem.global_stiffness_matrix()

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
            restrains = support.restrain_vector()

            # modify the stiffness matrix of the structure
            for index in indices[restrains]:
                k_s[index] = k_s[:, index] = np.zeros(n_dof_s)
                k_s[index, index] = 1

        # set the stiffness matrix of the structure
        self._k = k_s

    def analyse_load_pattern(self, load_pattern):
        """Analyse the model subjected to a load pattern.

        This method calculates the joint displacements, element end actions, support reactions, and internal forces
        and displacements for a given load pattern.

        Args:
            load_pattern (str): Load pattern name.
        """
        # degrees of freedom of the joints
        dof_joints = self.get_degrees_freedom()
        # degrees of freedom of the elements
        dof_element = np.tile(dof_joints, 2)

        # number active degrees of freedom per joint
        n_dof_joint = np.count_nonzero(dof_joints)
        # number active degrees of freedom per element
        n_dof_element = np.count_nonzero(dof_element)
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
            restrains = support.restrain_vector()

            for index in indices[restrains]:
                load_vector_support[index, 0] = 0

        # find displacements
        u = np.linalg.solve(self.get_stiffness_matrix(), load_vector_support)

        # store displacements
        l_p_d = {}

        for j in self.joints:
            indices = j_i[j]

            displacement = np.full(6, None)
            displacement[dof_joints] = u[indices, 0]

            l_p_d[j] = Displacement(self, load_pattern, j, *displacement)

        self.displacements[load_pattern] = l_p_d

        # store element end actions
        rows = []
        cols = []
        data = []

        l_p_e_a = {}

        for key, elem in self.elements.items():
            i_e = np.concatenate((j_i[elem.joint_j], j_i[elem.joint_k]))

            k_e = elem.local_stiffness_matrix()
            t_e = elem.rotation_transformation_matrix()

            u_e = np.zeros((12, 1))
            u_e[dof_element] = u[i_e]
            u_e = np.dot(np.transpose(t_e), u_e)

            f_fixed = np.zeros((12, 1))

            if key in loadPattern.element_point_loads:
                for pLoad in loadPattern.element_point_loads[key]:
                    f_fixed += pLoad.fixed_load_vector()

            if key in loadPattern.element_distributed_loads:
                for dLoad in loadPattern.element_distributed_loads[key]:
                    f_fixed += dLoad.fixed_load_vector()

            f_end_actions = np.full(12, None)
            f_end_actions[dof_element] = np.ravel(np.dot(k_e, u_e) + f_fixed)[dof_element]

            l_p_e_a[key] = EndActions(self, load_pattern, key, *f_end_actions)

            # reactions
            if elem.joint_j in self.supports or elem.joint_k in self.supports:
                rows.extend(i_e.tolist())
                cols.extend(n_dof_element * [0])
                data.extend(
                    np.dot(t_e, l_p_e_a[key].get_end_actions()).flatten()[dof_element].tolist())

        self.end_actions[load_pattern] = l_p_e_a

        # store reactions
        load_vector += loadPattern.fixed_load_vector()
        f_end_actions = coo_matrix(
            (data, (rows, cols)), (n_j * n_dof_joint, 1)).toarray()

        load_pattern_reactions = {}

        for j in self.supports:
            indices = j_i[j]
            reactions = np.full(6, None)
            reactions[dof_joints] = np.ravel(f_end_actions[indices] - load_vector[indices])

            load_pattern_reactions[j] = Reaction(
                self, load_pattern, j, *reactions)

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
        """Analyse the structure subjected to all load patterns.

        This method first sets up the degrees of freedom, joint indices, and the global stiffness matrix, then
        proceeds to analyze the structure for each load pattern sequentially.

        """
        # set the flags of the degrees of freedom
        self.set_degrees_freedom()

        # set joint indices
        self.set_joint_indices()

        # set the stiffness matrix of the structure
        self.set_stiffness_matrix()

        # solve the structure due to each load pattern
        for load_pattern in self.load_patterns:
            self.analyse_load_pattern(load_pattern)

    def export(self, filename):
        """Save the structure data to a file in JSON format.

        This method serializes the model's materials, sections, joints, elements, supports, load patterns (including
        their applied loads), and analysis results (displacements, reactions, end actions, internal forces, and
        internal displacements) into a JSON format.

        Args:
            filename (str): Filename.
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
                if loadPattern.joint_point_loads:
                    data['load_patterns'][key]['joints'] = {}
                    for _joint, point_loads in loadPattern.joint_point_loads.items():
                        data['load_patterns'][key]['joints'][_joint] = []
                        for pointLoad in point_loads:
                            data['load_patterns'][key]['joints'][_joint].append(
                                {attr: value for attr, value in pointLoad.__dict__.items() if not attr.startswith('_') and value is not None})

                # save loads at frames
                if loadPattern.element_distributed_loads:  # loadPattern.point_loads_at_frames or
                    data['load_patterns'][key]['frames'] = {}

                    for _frame, distributed_loads in loadPattern.element_distributed_loads.items():
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
