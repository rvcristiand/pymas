import numpy as np

from scipy.spatial.transform import Rotation
from scipy.sparse import bsr_matrix, coo_matrix
from pymas.classtools import AttrDisplay


class Material(AttrDisplay):
    """Linear elastic material.

    This class stores the fundamental properties of a material used in the
    structural model.

    Attributes:
        name (str): Name of the material.
        E (float): Modulus of elasticity of the material.
        G (float): Modulus of elasticity in shear of the material.
    """

    def __init__(self, parent, name, modulus_elasticity=None,
                 modulus_elasticity_shear=None):
        """Instantiate a Material object.

        Args:
            parent (Structure): Structure object.
            name (str): Name of the material.
            modulus_elasticity (float): Modulus of elasticity of the material.
            modulus_elasticity_shear (float): Modulus of elasticity in shear of
                the material.
        """
        self._parent = parent
        self.name = name
        self.E = modulus_elasticity
        self.G = modulus_elasticity_shear


class Section(AttrDisplay):
    """Cross section.

    This class defines the geometric properties of a cross section used for
    structural elements.

    Attributes:
        name (str): Name of the cross section.
        A (float): Area of the cross section.
        J (float): Torsion constant of the cross section.
        Iy (float): Inertia of the cross section with respect to the local
            y-axis.
        Iz (float): Inertia of the cross section with respect to the local
            z-axis.
    """

    def __init__(self, parent, name, area=None, torsion_constant=None,
                 inertia_y=None, inertia_z=None):
        """Instantiate a Section object.

        Args:
            parent (Structure): Structure object.
            name (str): Name of the cross section.
            area (float): Area of the cross section.
            torsion (float): Torsion constant of the cross section.
            inertia_y (float): Inertia of the cross section with respect to the
                local y-axis.
            inertia_z (float): Inertia of the cross section with respect to the
                local z-axis.
        """
        self._parent = parent
        self.name = name
        self.A = area
        self.J = torsion_constant
        self.Iy = inertia_y
        self.Iz = inertia_z


class CircularSection(Section):
    """Circular cross section.

    This class extends the generic `Section` class to automatically calculate
    geometric properties for a circular shape based on its diameter.

    Attributes:
        name (str): Name of the cross section.
        diameter (float): Diameter of the cross section.
        A (float): Area of the cross section.
        J (float): Torsion constant of the cross section.
        Iy (float): Inertia of the cross section with respect to the local
            y-axis.
        Iz (float): Inertia of the cross section with respect to the local
            z-axis.
    """

    def __init__(self, parent, name, diameter):
        """Instantiate a CircularSection object.

        Args:
            parent (Structure): Structure object.
            name (str): Name of the circular cross section.
            diameter (float): Diameter of the circular cross section.
        """
        radius = diameter / 2
        A = np.pi * radius**2
        J = np.pi * radius**4 / 2
        Iy = Iz = np.pi * radius**4 / 4

        self._parent = parent
        self.diameter = diameter
        super().__init__(parent, name, A, J, Iy, Iz)


class RectangularSection(Section):
    """Rectangular cross section.

    This class extends the generic `Section` class to automatically calculate
    geometric properties for a rectangular shape based on its base and height.

    Attributes:
        name (str): Name of the cross section.
        base (float): Base of the cross section.
        height (float): Height of the cross section.
        A (float): Area of the cross section.
        J (float): Torsion constant of the cross section.
        Iy (float): Inertia of the cross section with respect to the local
            y-axis.
        Iz (float): Inertia of the cross section with respect to the local
            z-axis.
    """

    def __init__(self, parent, name, base, height):
        """Instantiate a RectangularSection object.

        Args:
            parent (Structure): Structure object.
            name (str): Name of the rectangular cross section.
            base (float): Base of the rectangular cross section.
            height (float): Height of the rectangular cross section.
        """
        a, b = sorted((base, height))

        A = base * height
        J = (1/3 - 0.21 * (a/b) * (1 - (1/12) * (a/b)**4)) * b * a**3
        Iy = (1 / 12) * height * base ** 3
        Iz = (1 / 12) * base * height ** 3

        self._parent = parent
        self.base = base
        self.height = height
        super().__init__(parent, name, A, J, Iy, Iz)


class Joint(AttrDisplay):
    """End of frames.

    A joint defines a node in the structural model with specific spatial
    coordinates.

    Attributes:
        name (str): Name of the joint.
        x (float): Coordinate X of the joint.
        y (float): Coordinate Y of the joint.
        z (float): Coordinate Z of the joint.

    Methods:
        position(): Return the position vector of the joint.
    """

    def __init__(self, parent, name, x=None, y=None, z=None):
        """Instantiate a Joint object.

        Args:
            parent (Structure): Structure object.
            name (str): Name of the joint.
            x (float, optional): Coordinate X of the joint.
            y (float, optional): Coordinate Y of the joint.
            z (float, optional): Coordinate Z of the joint.
        """
        self._parent = parent
        self.name = name
        self.x = x
        self.y = y
        self.z = z

    def position(self):
        """Return the position of the joint.

        Returns:
            ndarray: Position vector of the joint.
        """
        x = self.x if self.x is not None else 0
        y = self.y if self.y is not None else 0
        z = self.z if self.z is not None else 0

        return np.array([x, y, z])


class Frame(AttrDisplay):
    """Long elements interconnected at rigid joints.

    This class models frame elements, considering axial, torsional, and biaxial
    bending deformations.

    Attributes:
        name (str): Name of the frame.
        joint_j (str): Name of the near joint of the frame.
        joint_k (str): Name of the far joint of the frame.
        material (str): Name of the material of the frame.
        section (str): Name of the cross section of the frame.
        axial (bool): Whether to consider axial deformation of the frame.
        torsional (bool): Whether to consider torsional deformation of the
            frame.
        bending_y (bool): Whether to consider bending deformation around the
            local y-axis of the frame.
        bending_z (bool): Whether to consider bending deformation around the
            local z-axis of the frame.
    """

    def __init__(self, parent, name, joint_j, joint_k, material, section,
                 axial=None, torsional=None, bending_y=None, bending_z=None):
        """Instantiate a Frame object.

        Args:
            parent (Structure): Structure object.
            name (str): Name of the frame.
            joint_j (str): Name of the near joint of the frame.
            joint_k (str): Name of the far joint of the frame.
            material (str): Name of the material of the frame.
            section (str): Name of the cross section of the frame.
            axial (bool, optional): Consideration of axial deformation of the
                frame. Defaults to None.
            torsional (bool, optional): Consideration of torsional deformation
                of the frame. Defaults to None.
            bending_y (bool, optional): Consideration of bending around the
                local y-axis of the frame. Defaults to None.
            bending_z (bool, optional): Consideration of bending around the
                local z-axis of the frame. Defaults to None.
        """
        self._parent = parent
        self.name = name
        self.joint_j = joint_j
        self.joint_k = joint_k
        self.material = material
        self.section = section
        self.axial = axial
        self.torsional = torsional
        self.bending_y = bending_y
        self.bending_z = bending_z

    def length(self):
        """Return the length of the frame.

        Calculates the distance between the near and far joints.

        Returns:
            float: Length of the frame.
        """
        j = self._parent.joints[self.joint_j].position()
        k = self._parent.joints[self.joint_k].position()

        return np.linalg.norm(k - j)

    def direction_cosines_vector(self):
        """Return the direction cosines of the frame.

        Calculates the unit vector along the frame's length in the global
        coordinate system.

        Returns:
            ndarray: Direction cosines of the frame.
        """
        j = self._parent.joints[self.joint_j].position()
        k = self._parent.joints[self.joint_k].position()
        vector = k - j

        return vector / np.linalg.norm(vector)

    def rotation_matrix(self):
        """Return the rotation matrix of the frame.

        This matrix transforms vectors from the local coordinate system of the
        frame to the global coordinate system.

        Returns:
            ndarray: Rotation matrix of the frame.
        """
        v_from = np.array([1, 0, 0])
        v_to = self.direction_cosines_vector()

        if np.all(v_from == v_to):
            return Rotation.from_quat([0, 0, 0, 1]).as_matrix()

        elif np.all(v_from == -v_to):
            return Rotation.from_quat([0, 0, 1, 0]).as_matrix()

        else:
            w = np.cross(v_from, v_to)
            w /= np.linalg.norm(w)
            theta = np.arccos(np.dot(v_from, v_to))
            quaternion = np.hstack((w * np.sin(theta/2), np.cos(theta/2)))

        return Rotation.from_quat(quaternion).as_matrix()

    def rotation_transformation_matrix(self):
        """Return the rotation transformation matrix of the frame.

        This matrix transforms displacement and force vectors between the local
        and global coordinate systems for an element with 6 degrees of freedom
        at each end.

        Returns:
            ndarray: Rotation transformation matrix of the frame.
        """
        indptr = np.array([0, 1, 2, 3, 4])
        indices = np.array([0, 1, 2, 3])
        data = np.tile(self.rotation_matrix(), (4, 1, 1))

        return bsr_matrix((data, indices, indptr), shape=(12, 12)).toarray()

    def local_stiffness_matrix(self):
        """Return the local stiffness matrix of the frame.

        Calculates the 12x12 stiffness matrix for the frame element in its
        local coordinate system, considering axial, shear, and bending
        deformations.

        Returns:
            ndarray: Local stiffness matrix of the frame.
        """
        L = self.length()

        material = self._parent.materials[self.material]
        E = material.E if material.E is not None else 0
        G = material.G if material.G is not None else 0

        section = self._parent.sections[self.section]
        A = section.A if section.A is not None else 0
        J = section.J if section.J is not None else 0
        Iy = section.Iy if section.Iy is not None else 0
        Iz = section.Iz if section.Iz is not None else 0

        rows, cols, data = [], [], []

        # axial
        if self.axial:
            ael = A * E / L
            rows.extend([0, 6, 6, 0])
            cols.extend([0, 0, 6, 6])
            data.extend([ael, -ael, ael, -ael])

        # torsional
        if self.torsional:
            gjl = G * J / L
            rows.extend([3, 9, 9, 3])
            cols.extend([3, 3, 9, 9])
            data.extend([gjl, -gjl, gjl, -gjl])

        # bending z
        if self.bending_z:
            e_iz_l3 = 12 * E * Iz / L ** 3
            e_iz_l2 = 6 * E * Iz / L ** 2
            e_iz_l = E * Iz / L

            rows.extend([
                1, 7, 7, 1,
                1, 5, 1, 11, 5, 7, 7, 11,
                5, 11, 11, 5
            ])
            cols.extend([
                1, 1, 7, 7,
                5, 1, 11, 1, 7, 5, 11, 7,
                5, 5, 11, 11
            ])
            data.extend([
                e_iz_l3, -e_iz_l3, e_iz_l3, -e_iz_l3,
                e_iz_l2, e_iz_l2, e_iz_l2, e_iz_l2,
                -e_iz_l2, -e_iz_l2, -e_iz_l2, -e_iz_l2,
                4 * e_iz_l, 2 * e_iz_l, 4 * e_iz_l, 2 * e_iz_l
            ])

        # bending y
        if self.bending_y:
            e_iy_l3 = 12 * E * Iy / L ** 3
            e_iy_l2 = 6 * E * Iy / L ** 2
            e_iy_l = E * Iy / L

            rows.extend([
                2, 8, 8, 2,
                4, 8, 8, 10, 2, 4, 2, 10,
                4, 10, 10, 4
            ])
            cols.extend([
                2, 2, 8, 8,
                8, 4, 10, 8, 4, 2, 10, 2,
                4, 4, 10, 10
            ])
            data.extend([
                e_iy_l3, -e_iy_l3, e_iy_l3, -e_iy_l3,
                e_iy_l2, e_iy_l2, e_iy_l2, e_iy_l2,
                -e_iy_l2, -e_iy_l2, -e_iy_l2, -e_iy_l2,
                4 * e_iy_l, 2 * e_iy_l, 4 * e_iy_l, 2 * e_iy_l
            ])

        return coo_matrix((data, (rows, cols)), (12, 12)).toarray()

    def global_stiffness_matrix(self):
        """Return the global stiffness matrix of the frame.

        Transforms the local stiffness matrix of the frame into the global
        coordinate system and filters it based on the active degrees of freedom
        of the structure.

        Returns:
            ndarray: Global stiffness matrix of the frame.
        """
        # degrees of freedom
        dof = self._parent.get_degrees_freedom()
        # degrees of freedom of the frame
        dof_frame = np.nonzero(np.tile(dof, 2))[0]

        # local stiffness matrix of the frame
        k_local = self.local_stiffness_matrix()
        # rotation transformation matrix of the frame
        t = self.rotation_transformation_matrix()
        # global matrix stiffness of the frame
        k_global = np.dot(np.dot(t, k_local), np.transpose(t))

        return k_global[dof_frame[:, None], dof_frame]

    def get_internal_forces(self, load_pattern, no_div=100):
        """Get the internal forces of the element.

        Calculates the axial forces (fx), shear forces (fy, fz), and bending
        moments (mx, my, mz) at various divisions along the element's length
        for a specified load pattern.

        Args:
            load_pattern (str): Name of the load pattern.
            no_div (float, optional): Number of divisions.

        Returns:
            dict: Internal forces of the element.
        """
        loadPattern = self._parent.load_patterns[load_pattern]
        endActions = self._parent.end_actions[load_pattern][self.name]

        length = self.length()

        fx_j = endActions.fx_j if (
            endActions.fx_j is not None and self.axial) else 0
        fy_j = endActions.fy_j if (
            endActions.fy_j is not None and self.bending_z) else 0
        fz_j = endActions.fz_j if (
            endActions.fz_j is not None and self.bending_y) else 0
        mx_j = endActions.mx_j if (
            endActions.mx_j is not None and self.torsional) else 0
        my_j = endActions.my_j if (
            endActions.my_j is not None and self.bending_y) else 0
        mz_j = endActions.mz_j if (
            endActions.mz_j is not None and self.bending_z) else 0

        internal_forces = {}
        internal_forces['fx'] = np.full(shape=no_div+1, fill_value=-fx_j)
        internal_forces['fy'] = np.full(shape=no_div+1, fill_value=fy_j)
        internal_forces['fz'] = np.full(shape=no_div+1, fill_value=fz_j)
        internal_forces['mx'] = np.full(shape=no_div+1, fill_value=-mx_j)
        internal_forces['my'] = np.full(shape=no_div+1, fill_value=my_j)
        internal_forces['mz'] = np.full(shape=no_div+1, fill_value=-mz_j)

        for i in range(no_div+1):
            x = (i / no_div) * length
            if self.bending_y:
                internal_forces['my'][i] += fz_j * x
            if self.bending_z:
                internal_forces['mz'][i] += fy_j * x

        if self.name in loadPattern.frame_distributed_loads:
            for distributed_load in \
                    loadPattern.frame_distributed_loads[self.name]:
                fx = distributed_load.fx if (
                    distributed_load.fx is not None and self.axial) else 0
                fy = distributed_load.fy if (
                    distributed_load.fy is not None and self.bending_z) else 0
                fz = distributed_load.fz if (
                    distributed_load.fz is not None and self.bending_y) else 0

                for i in range(no_div+1):
                    x = (i / no_div) * length
                    if self.axial:
                        internal_forces['fx'][i] -= fx * x
                    if self.bending_z:
                        internal_forces['fy'][i] += fy * x
                        internal_forces['mz'][i] += fy * x ** 2 / 2
                    if self.bending_y:
                        internal_forces['fz'][i] += fz * x
                        internal_forces['my'][i] += fz * x ** 2 / 2

        internal_forces['fx'] = internal_forces['fx'].tolist()
        internal_forces['fy'] = internal_forces['fy'].tolist()
        internal_forces['fz'] = internal_forces['fz'].tolist()
        internal_forces['mx'] = internal_forces['mx'].tolist()
        internal_forces['my'] = internal_forces['my'].tolist()
        internal_forces['mz'] = internal_forces['mz'].tolist()

        return internal_forces

    def get_internal_displacements(self, load_pattern, no_div=100):
        """Get the internal displacements.

        Calculates the axial (ux), shear (uy, uz), and rotational (rx, ry, rz)
        displacements at various divisions along the element's length for a
        specified load pattern.

        Args:
            load_pattern (str): Name of the load pattern.
            np_div (float, optional): Number of divisions.

        Returns:
            dict: Internal displacements of the element.
        """
        material = self._parent.materials[self.material]
        section = self._parent.sections[self.section]
        loadPattern = self._parent.load_patterns[load_pattern]
        end_actions = self._parent.end_actions[load_pattern][self.name]
        j_joint_displ = \
            self._parent.displacements[load_pattern][self.joint_j]

        length = self.length()
        E = material.E if material.E is not None else 0
        G = material.G if material.G is not None else 0
        A = section.A if section.A is not None else 0
        J = section.J if section.J is not None else 0
        Iy = section.Iy if section.Iy is not None else 0
        Iz = section.Iz if section.Iz is not None else 0

        end_actions = self._parent.end_actions[load_pattern][self.name]
        fx_j, fy_j, fz_j, mx_j, my_j, mz_j = end_actions.get_end_actions()[:6]

        # Aplicar filtros a las fuerzas del extremo inicial
        fx_j = fx_j if self.axial else 0
        fy_j = fy_j if self.bending_z else 0
        fz_j = fz_j if self.bending_y else 0
        mx_j = mx_j if self.torsional else 0
        my_j = my_j if self.bending_y else 0
        mz_j = mz_j if self.bending_z else 0

        j_joint_displ = \
            self._parent.displacements[load_pattern][self.joint_j].displacement_vector()
        j_joint_displ = np.dot(np.transpose(
            self.rotation_transformation_matrix())[:6, :6], j_joint_displ)
        ux_j, uy_j, uz_j, rx_j, ry_j, rz_j = j_joint_displ

        # Aplicar filtros a los desplazamientos del nodo inicial
        ux_j = ux_j if self.axial else 0
        uy_j = uy_j if self.bending_z else 0
        uz_j = uz_j if self.bending_y else 0
        rx_j = rx_j if self.torsional else 0
        ry_j = ry_j if self.bending_y else 0
        rz_j = rz_j if self.bending_z else 0

        internal_displacements = {}
        internal_displacements['ux'] = np.full(shape=no_div+1, fill_value=ux_j)
        internal_displacements['uy'] = np.full(shape=no_div+1, fill_value=uy_j)
        internal_displacements['uz'] = np.full(shape=no_div+1, fill_value=uz_j)
        internal_displacements['rx'] = np.full(shape=no_div+1, fill_value=rx_j)
        internal_displacements['ry'] = np.full(shape=no_div+1, fill_value=ry_j)
        internal_displacements['rz'] = np.full(shape=no_div+1, fill_value=rz_j)

        for i in range(no_div+1):
            x = (i / no_div) * length
            if self.axial and E * A != 0:
                internal_displacements['ux'][i] -= fx_j * x / (E * A)
            if self.bending_z and E * Iz != 0:
                internal_displacements['uy'][i] += fy_j * x ** 3 / (6 * E * Iz)
                internal_displacements['uy'][i] -= mz_j * x ** 2 / (2 * E * Iz)
                internal_displacements['uy'][i] += rz_j * x
            if self.bending_y and E * Iy != 0:
                internal_displacements['uz'][i] += fz_j * x ** 3 / (6 * E * Iy)
                internal_displacements['uz'][i] += my_j * x ** 2 / (2 * E * Iy)
                internal_displacements['uz'][i] -= ry_j * x
            if self.torsional and G * J != 0:
                internal_displacements['rx'][i] -= mx_j * x / (G * J)
            if self.bending_y and E * Iz != 0:
                internal_displacements['ry'][i] -= fz_j * x ** 2 / (2 * E * Iz)
                internal_displacements['ry'][i] -= my_j * x / (E * Iz)
            if self.bending_z and E * Iz != 0:
                internal_displacements['rz'][i] += fy_j * x ** 2 / (2 * E * Iz)
            if self.bending_z and E * Iy != 0:
                internal_displacements['rz'][i] += mz_j * x / (E * Iy)

        if self.name in loadPattern.frame_distributed_loads:
            for distributed_load in loadPattern.frame_distributed_loads[self.name]:
                fx = distributed_load.fx if (
                    distributed_load.fx is not None and self.axial) else 0
                fy = distributed_load.fy if (
                    distributed_load.fy is not None and self.bending_z) else 0
                fz = distributed_load.fz if (
                    distributed_load.fz is not None and self.bending_y) else 0

                for i in range(no_div+1):
                    x = (i / no_div) * length
                    if self.axial and E * A != 0:
                        internal_displacements['ux'][i] -= fx * \
                            x ** 2 / (2 * E * A)
                    if self.bending_z and E * Iz != 0:
                        internal_displacements['uy'][i] += fy * \
                            x ** 4 / (24 * E * Iz)
                    if self.bending_y and E * Iy != 0:
                        internal_displacements['uz'][i] += fz * \
                            x ** 4 / (24 * E * Iy)
                    if self.bending_y and E * Iz != 0:
                        internal_displacements['ry'][i] -= fz * \
                            x ** 3 / (6 * E * Iz)
                    if self.bending_z and E * Iz != 0:
                        internal_displacements['rz'][i] += fy * \
                            x ** 3 / (6 * E * Iz)

        internal_displacements['ux'] = internal_displacements['ux'].tolist()
        internal_displacements['uy'] = internal_displacements['uy'].tolist()
        internal_displacements['uz'] = internal_displacements['uz'].tolist()
        internal_displacements['rx'] = internal_displacements['rx'].tolist()
        internal_displacements['ry'] = internal_displacements['ry'].tolist()
        internal_displacements['rz'] = internal_displacements['rz'].tolist()

        return internal_displacements


class Support(AttrDisplay):
    """Point of support.

    This class defines the boundary conditions for a joint, specifying which
    degrees of freedom are restrained.

    Attributes:
        joint (str): Name of the joint.
        r_ux (bool): Indicates whether the support restrains the displacement
            of the joint along the global x-axis.
        r_uy (bool): Indicates whether the support restrains the displacement
            of the joint along the global y-axis.
        r_uz (bool): Indicates whether the support restrains the displacement
            of the joint along the global z-axis.
        r_rx (bool): Indicates whether the support restrains the rotation of
            the joint around the global x-axis.
        r_ry (bool): Indicates whether the support restrains the rotation of
            the joint around the global y-axis.
        r_rz (bool): Indicates whether the support restrains the rotation of
            the joint around the global z-axis.
    """

    def __init__(self, parent, joint, r_ux=None, r_uy=None, r_uz=None,
                 r_rx=None, r_ry=None, r_rz=None):
        """Instantiate a Support object.

        Args:
            parent (Structure): Structure object.
            joint (str): Name of the joint of the support.
            r_ux (bool, optional): Whether the support restrains displacement
                of the joint along the global x-axis.
            r_uy (bool, optional): Whether the support restrains displacement
                of the joint along the global y-axis.
            r_uz (bool, optional): Whether the support restrains displacement
                of the joint along the global z-axis.
            r_rx (bool, optional): Whether the support restrains displacement
                of the joint around the global x-axis.
            r_ry (bool, optional): Whether the support restrains displacement
                of the joint around the global y-axis.
            r_rz (bool, optional): Whether the support restrains displacement
                of the joint around the global z-axis.
        """
        self._parent = parent
        self.joint = joint
        self.r_ux = r_ux
        self.r_uy = r_uy
        self.r_uz = r_uz
        self.r_rx = r_rx
        self.r_ry = r_ry
        self.r_rz = r_rz

    def restrains(self):
        """Return the restrain vector of the support.

        This vector is filtered to include only the active degrees of freedom
        of the parent structure.

        Returns:
            ndarray: Restrain vector.
        """
        # degrees of freedom
        dof = self._parent.get_degrees_freedom()

        r_ux = self.r_ux if self.r_ux is not None else False
        r_uy = self.r_uy if self.r_uy is not None else False
        r_uz = self.r_uz if self.r_uz is not None else False
        r_rx = self.r_rx if self.r_rx is not None else False
        r_ry = self.r_ry if self.r_ry is not None else False
        r_rz = self.r_rz if self.r_rz is not None else False

        return np.array([r_ux, r_uy, r_uz, r_rx, r_ry, r_rz])[dof]


class LoadPattern(AttrDisplay):
    """Load pattern.

    A load pattern groups different types of loads (joint point loads and frame
    point and distributed loads) that are applied simultaneously.

    Attributes:
        name (str): Name of the load pattern.
        joint_point_loads (dict): Joint point loads of the load pattern.
        frame_point_loads (dict): Frame point loads of the load pattern.
        frame_distributed_loads (dict): Frame uniformly distributed loads
            of the load pattern.
    """

    def __init__(self, parent, name):
        """Instantiate a LoadPattern object.

        Args:
            parent (Structure): Structure object.
            name (str): Name of the load pattern.
        """
        self._parent = parent
        self.name = name
        self.joint_point_loads = {}
        self.frame_point_loads = {}
        self.frame_distributed_loads = {}

    def add_joint_point_load(self, joint, fx=None, fy=None, fz=None, mx=None,
                             my=None, mz=None):
        """Add a joint point load to the dictionary of joint point loads.

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
            JointPointLoad: Joint point load.
        """
        pointLoad = JointPointLoad(self._parent, self.name, joint, fx, fy, fz,
                                   mx, my, mz)

        try:
            self.joint_point_loads[joint].append(pointLoad)
        except KeyError:
            self.joint_point_loads[joint] = [pointLoad]

        return pointLoad

    def add_frame_point_load(self, frame, dist, fx=None, fy=None, fz=None,
                             mx=None, my=None, mz=None):
        """Add a frame point load to the dictionary of frame point loads.

        Args:
            frame (str): Name of the frame.
            dist (float): Distance of the point load from the near joint along
                the local x-axis.
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
        pointLoad = FramePointLoad(self._parent, self.name, frame, dist, fx,
                                   fy, fz, mx, my, mz)

        try:
            self.frame_point_loads[frame].append(pointLoad)
        except KeyError:
            self.frame_point_loads[frame] = [pointLoad]

        return pointLoad

    def add_frame_distributed_load(self, frame, fx=None, fy=None, fz=None,
                                   mx=None, my=None, mz=None):
        """Add an element uniformly distributed load to the dictionary of
        element uniformly distributed loads.

        Args:
            element (str): Name of the element.
            fx (float, optional): Intensity of the uniformly distributed load
                along the local x-axis.
            fy (float, optional): Intensity of the uniformly distributed load
                along the local y-axis.
            fz (float, optional): Intensity of the uniformly distributed load
                along the local z-axis.
            mx (float, optional): Intensity of the uniformly distributed load
                around the local x-axis.
            my (float, optional): Intensity of the uniformly distributed load
                around the local y-axis.
            mz (float, optional): Intensity of the uniformly distributed load
                around the local z-axis.

        Returns:
            DistributedLoad: DistributedLoad object.
        """
        distributedLoad = DistributedLoad(self._parent, self.name, frame, fx,
                                          fy, fz, mx, my, mz)

        try:
            self.frame_distributed_loads[frame].append(distributedLoad)
        except KeyError:
            self.frame_distributed_loads[frame] = [distributedLoad]

        return distributedLoad

    def load_vector(self):
        """ Returns the load vector of the load pattern.

        This vector is calculated as the actual load vector minus the fixed-end
        load vector.

        Returns:
            ndarray: Load vector of the load pattern.
        """
        return self.actual_load_vector() - self.fixed_load_vector()

    def actual_load_vector(self):
        """Returns the actual load vector of the load pattern.

        This vector contains the applied forces and moments at each joint due
        to point loads, considering only the active degrees of freedom.

        Returns:
            ndarray: Actual load vector of the load pattern.
        """
        # number of joints
        n_j = len(self._parent.joints)
        # number active degrees of freedom
        n_dof = self._parent.number_active_degrees_freedom()
        # number of joint point loads
        n_point_loads = len(self.joint_point_loads)
        # joint indices of the structure
        j_i = self._parent.get_joint_indices()

        # row positions of the load vectors items of the point loads
        rows = np.empty(n_dof * n_point_loads, dtype=int)
        cols = np.zeros_like(rows, dtype=int)
        # items of the load vectors of the point loads
        data = np.zeros_like(rows, dtype=float)

        # assembly the point load vectors
        for i, (joint, p_loads) in enumerate(self.joint_point_loads.items()):
            start = i * n_dof
            end = (i + 1) * n_dof
            # row positions of the load vectors items of the point loads
            rows[start:end] = j_i[joint]

            # items of the point loads
            for pointLoad in p_loads:
                data[start:end] += pointLoad.load_vector()

        return coo_matrix((data, (rows, cols)), (n_dof * n_j, 1)).toarray()

    def fixed_load_vector(self):
        """Returns the fixed-end load vector of the load pattern.

        This vector represents the forces and moments that would develop at
        the element ends if all joints were fully restrained, due to frame
        loads.

        Returns:
            ndarray: Fixed-end load vector.
        """
        # number of joints
        n_j = len(self._parent.joints)
        # degrees of freedom
        dof_joints = self._parent.get_degrees_freedom()
        dof_frame = np.tile(dof_joints, 2)
        # number active degrees of freedom
        n_dof = self._parent.number_active_degrees_freedom()
        # number of frame distributed loads
        n_distributed_loads = len(self.frame_distributed_loads)
        # joint indices of the structure
        j_i = self._parent.get_joint_indices()

        # row positions of the load vectors items of the distributed loads
        rows = np.empty(2 * n_dof * n_distributed_loads, dtype=int)
        cols = np.zeros_like(rows)
        # items of the load vector of the distributed loads
        data = np.zeros_like(rows, dtype=float)

        # assembly the distributed load vector
        for i, (frame, d_l) in enumerate(self.frame_distributed_loads.items()):
            start = 2 * i * n_dof
            end = 2 * (i + 1) * n_dof

            # frame object
            frame = self._parent.frames[frame]
            joint_j, joint_k = frame.joint_j, frame.joint_k
            t = frame.rotation_transformation_matrix()

            # row positions of the elements of the distributed load at frame
            rows[start:end] = np.concatenate((j_i[joint_j], j_i[joint_k]))

            # data of the elements
            for dL in d_l:
                data[start:end] += np.dot(t, dL.fixed_load_vector()
                                          ).flatten()[dof_frame]

        return coo_matrix((data, (rows, cols)), (n_dof * n_j, 1)).toarray()


class JointPointLoad(AttrDisplay):
    """Joint point load.

    Represents a point load or moment applied directly to a joint.

    Attributes:
        load_pattern (str): Name of the load pattern.
        joint (str): Name of the joint.
        fx (float): Intensity of the point load along the global x-axis.
        fy (float): Intensity of the point load along the global y-axis.
        fz (float): Intensity of the point load along the global z-axis.
        mx (float): Intensity of the point load around the global x-axis.
        my (float): Intensity of the point load around the global y-axis.
        mz (float): Intensity of the point load around the global z-axis.
    """

    def __init__(self, parent, load_pattern, joint, fx=None, fy=None, fz=None,
                 mx=None, my=None, mz=None):
        """Instantiate a JointPointLoad object.

        Args:
            parent (Structure): Structure object.
            load_pattern (str): Name of the load pattern.
            joint (str): Name of the joint.
            fx (float, optional): Intensity of the joint point load along the
                global x-axis.
            fy (float, optional): Intensity of the joint point load along the
                global y-axis.
            fz (float, optional): Intensity of the joint point load along the
                global z-axis.
            mx (float, optional): Intensity of the joint point load around the
                global x-axis.
            my (float, optional): Intensity of the joint point load around the
                global y-axis.
            mz (float, optional): Intensity of the joint point load around the
                global z-axis.
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.joint = joint
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

    def load_vector(self):
        """Returns the load vector of the joint point load.

        This vector contains the forces and moments applied by this specific
        point load, filtered to include only the active degrees of freedom of
        the parent structure.

        Returns:
            ndarray: Load vector.
        """
        # degrees of freedom
        dof = self._parent.get_degrees_freedom()

        fx = self.fx if self.fx is not None else 0
        fy = self.fy if self.fy is not None else 0
        fz = self.fz if self.fz is not None else 0
        mx = self.mx if self.mx is not None else 0
        my = self.my if self.my is not None else 0
        mz = self.mz if self.mz is not None else 0

        return np.array([fx, fy, fz, mx, my, mz])[dof]


class FramePointLoad(AttrDisplay):
    """Element point load.

    This load is defined in the element's local coordinate system.

    Attributes:
        load_pattern (str): Name of the load pattern.
        element (str): Name of the element.
        dist (float): Distance of the point load from the near joint along the
            local x-axis.
        fx (float): Intensity of the point load along the local x-axis.
        fy (float): Intensity of the point load along the local y-axis.
        fz (float): Intensity of the point load along the local z-axis.
        mx (float): Intensity of the point load around the local x-axis.
        my (float): Intensity of the point load around the local y-axis.
        mz (float): Intensity of the point load around the local z-axis.
    """

    def __init__(self, parent, load_pattern, frame, dist, fx=None, fy=None,
                 fz=None, mx=None, my=None, mz=None):
        """Instantiate an ElementPointLoad object.

        Args:
            parent (Structure): Structure object.
            load_pattern (str): Name of the load pattern.
            element (str): Name of the element.
            dist (float): Distance of the point load from the near joint along
                the local x-axis.
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
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.frame = frame
        self.dist = dist
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

    def fixed_load_vector(self):
        """Returns the fixed-end load vector of the element point load.

        This vector represents the forces and moments that would develop at the element ends if all joints were
        fully restrained, due to this specific point load.

        Returns:
            ndarray: Fixed-end load vector.
        """
        # uniformly distributed forces
        fx = self.fx if self.fx is not None else 0
        fy = self.fy if self.fy is not None else 0
        fz = self.fz if self.fz is not None else 0
        mx = self.mx if self.mx is not None else 0
        my = self.my if self.my is not None else 0
        mz = self.mz if self.mz is not None else 0

        # dimensions of the element
        L = self._parent.frames[self.frame].length()
        a = self.dist
        b = L - a

        load_vector = np.empty((2 * 6, 1))

        # fx
        load_vector[0] = -fx * b / L
        load_vector[6] = -fx * a / L

        # fy
        load_vector[1] = -fy * b**2 * (3*a + b) / L**3
        load_vector[7] = -fy * a**2 * (a + 3*b) / L**3

        load_vector[5] = -fy * a * b**2 / L**2
        load_vector[11] = fy * a**2 * b / L**2

        # fz
        load_vector[2] = -fz * b**2 * (3*a + b) / L**3
        load_vector[8] = -fz * a**2 * (a + 3*b) / L**3

        load_vector[4] = fz * a * b**2 / L**2
        load_vector[10] = -fz * a**2 * b / L**2

        # mx
        load_vector[3] = -mx * b / L
        load_vector[9] = -mx * a / L

        # my
        load_vector[2] += -6 * my * a * b / L**3
        load_vector[8] += 6 * my * a * b / L**3

        load_vector[4] += my * b * (2*a - b) / L**2
        load_vector[10] += my * a * (2*b - a) / L**2

        # mz
        load_vector[1] += 6 * mz * a * b / L**3
        load_vector[7] += -6 * mz * a * b / L**3

        load_vector[5] += mz * b * (2*a - b) / L**2
        load_vector[11] += mz * a * (2*b - a) / L**2

        return load_vector


class DistributedLoad(AttrDisplay):
    """Frame uniformly distributed load.

    This load is defined in the frame's local coordinate system.

    Attributes:
        load_pattern (str): Name of the load pattern.
        frame (str): Name of the frame.
        fx (float): Intensity of the uniformly distributed load along the local
            x-axis.
        fy (float): Intensity of the uniformly distributed load along the local
            y-axis.
        fz (float): Intensity of the uniformly distributed load along the local
            z-axis.
        mx (float): Intensity of the uniformly distributed load around the
            local x-axis.
        my (float): Intensity of the uniformly distributed load around the
            local y-axis.
        mz (float): Intensity of the uniformly distributed load around the
            local z-axis.
    """

    def __init__(self, parent, load_pattern, frame, fx=None, fy=None,
                 fz=None, mx=None, my=None, mz=None):
        """Instantiate a DistributedLoad object.

        Args:
            parent (Structure): Structure object.
            load_pattern (str): Name of the load pattern.
            element (str): Name of the element.
            fx (float, optional): Intensity of the uniformly distributed load
                along the local x-axis.
            fy (float, optional): Intensity of the uniformly distributed load
                along the local y-axis.
            fz (float, optional): Intensity of the uniformly distributed load
                along the local z-axis.
            mx (float, optional): Intensity of the uniformly distributed load
                around the local x-axis.
            my (float, optional): Intensity of the uniformly distributed load
                around the local y-axis.
            mz (float, optional): Intensity of the uniformly distributed load
                around the local z-axis.
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.frame = frame
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

    def fixed_load_vector(self):
        """Returns the fixed-end load vector of the distributed load.

        This vector represents the forces and moments that would develop at the
        frame ends if all joints were fully restrained, due to this specific
        distributed load.

        Returns:
            ndarray: Fixed-end load vector (load_vector[flags_dof_element]).

        Raises:
            NotImplementedError: If uniformly distributed moments around the
            y- or z-axis (my, mz) are set.
        """
        # uniformly distributed forces
        fx = self.fx if self.fx is not None else 0
        fy = self.fy if self.fy is not None else 0
        fz = self.fz if self.fz is not None else 0
        mx = self.mx if self.mx is not None else 0
        my = self.my if self.my is not None else 0
        mz = self.mz if self.mz is not None else 0

        L = self._parent.frames[self.frame].length()
        load_vector = np.empty((2 * 6, 1))

        # fx
        load_vector[0] = load_vector[6] = -fx * L / 2
        # fy
        load_vector[1] = load_vector[7] = -fy * L / 2

        load_vector[5] = -fy * L**2 / 12
        load_vector[11] = -load_vector[5]
        # fz
        load_vector[2] = load_vector[8] = -fz * L / 2

        load_vector[4] = fz * L**2 / 12
        load_vector[10] = -load_vector[4]
        # mx
        load_vector[3] = load_vector[9] = -mx * L / 2
        # my or mz
        if self.my not in (None, 0) or self.mz not in (None, 0):
            raise NotImplementedError(
                "Uniformly distributed moments around the y-axis (my) or"
                "z-axis (mz) are not currently implemented. Please remove"
                "these loads or implement their effects."
            )

        return load_vector


class Displacement(AttrDisplay):
    """Joint displacements.

    Attributes:
        load_pattern (str): Name of the load pattern.
        joint (str): Name of the joint.
        ux (float): Displacement along x-axis.
        uy (float): Displacement along y-axis.
        uz (float): Displacement along z-axis.
        rx (float): Displacement around x-axis.
        ry (float): Displacement around y-axis.
        rz (float): Displacement around z-axis.
    """

    def __init__(self, parent, load_pattern, joint, ux=None, uy=None, uz=None,
                 rx=None, ry=None, rz=None):
        """Instantiate a Displacements object.

        Args:
            parent (Structure): Structure.
            load_pattern (str): Name of the load pattern.
            joint (str): Name of the joint.
            ux (float, optional): Displacement along x-axis.
            uy (float, optional): Displacement along y-axis.
            uz (float, optional): Displacement along z-axis.
            rx (float, optional): Displacement around x-axis.
            ry (float, optional): Displacement around y-axis.
            rz (float, optional): Displacement around z-axis.
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.joint = joint
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def displacement_vector(self):
        """Returns the displacement vector.

        Returns:
            ndarray: Displacement vector.
        """

        ux = self.ux if self.ux is not None else 0
        uy = self.uy if self.uy is not None else 0
        uz = self.uz if self.uz is not None else 0
        rx = self.rx if self.rx is not None else 0
        ry = self.ry if self.ry is not None else 0
        rz = self.rz if self.rz is not None else 0

        return np.array([ux, uy, uz, rx, ry, rz])


class EndActions(AttrDisplay):
    """End actions.

    These are typically calculated in the frame's local coordinate system.

    Attributes:
        load_pattern (str): Load pattern name.
        element (str): Frame name.
        fx_j (float): Force along x-axis at near joint.
        fy_j (float): Force along y-axis at near joint.
        fz_j (float): Force along z-axis at near joint.
        mx_j (float): Moment around x-axis at near joint.
        my_j (float): Moment around y-axis at near joint.
        mz_j (float): Moment around z-axis at near joint.
        fx_k (float): Force along x-axis at far joint.
        fy_k (float): Force along y-axis at far joint.
        fz_k (float): Force along z-axis at far joint.
        mx_k (float): Moment around x-axis at far joint.
        my_k (float): Moment around y-axis at far joint.
        mz_k (float): Moment around z-axis at far joint.
    """

    def __init__(self, parent, load_pattern, frame, fx_j=None, fy_j=None,
                 fz_j=None, mx_j=None, my_j=None, mz_j=None, fx_k=None,
                 fy_k=None, fz_k=None, mx_k=None, my_k=None, mz_k=None):
        """ Initialize a EndActions object.

        Args:
            parent (Structure): Parent structure object.
            load_pattern (str): Load pattern name.
            element (str): Frame name.
            fx_j (float, optional): Force along the x-axis at joint_j.
                Defaults to None.
            fy_j (float, optional): Force along the y-axis at joint_j.
                Defaults to None.
            fz_j (float, optional): Force along the z-axis at joint_j.
                Defaults to None.
            mx_j (float, optional): Moment around the x-axis at joint_j.
                Defaults to None.
            my_j (float, optional): Moment around the y-axis at joint_j.
                Defaults to None.
            mz_j (float, optional): Moment around the z-axis at joint_j.
                Defaults to None.
            fx_k (float, optional): Force along the x-axis at joint_k.
                Defaults to None.
            fy_k (float, optional): Force along the y-axis at joint_k.
                Defaults to None.
            fz_k (float, optional): Force along the z-axis at joint_k.
                Defaults to None.
            mx_k (float, optional): Moment around the x-axis at joint_k.
                Defaults to None.
            my_k (float, optional): Moment around the y-axis at joint_k.
                Defaults to None.
            mz_k (float, optional): Moment around the z-axis at joint_k.
                Defaults to None.
        """
        self._parent = parent
        self.element = frame
        self.load_pattern = load_pattern
        self.fx_j = fx_j
        self.fy_j = fy_j
        self.fz_j = fz_j
        self.mx_j = mx_j
        self.my_j = my_j
        self.mz_j = mz_j
        self.fx_k = fx_k
        self.fy_k = fy_k
        self.fz_k = fz_k
        self.mx_k = mx_k
        self.my_k = my_k
        self.mz_k = mz_k

    def get_end_actions(self):
        """ Returns the vector of end actions for the element.

        Returns:
            ndarray: Forces and moments at both ends.
        """

        fx_j = self.fx_j if self.fx_j is not None else 0
        fy_j = self.fy_j if self.fy_j is not None else 0
        fz_j = self.fz_j if self.fz_j is not None else 0
        mx_j = self.mx_j if self.mx_j is not None else 0
        my_j = self.my_j if self.my_j is not None else 0
        mz_j = self.mz_j if self.mz_j is not None else 0

        fx_k = self.fx_k if self.fx_k is not None else 0
        fy_k = self.fy_k if self.fy_k is not None else 0
        fz_k = self.fz_k if self.fz_k is not None else 0
        mx_k = self.mx_k if self.mx_k is not None else 0
        my_k = self.my_k if self.my_k is not None else 0
        mz_k = self.mz_k if self.mz_k is not None else 0

        return np.array([fx_j, fy_j, fz_j, mx_j, my_j, mz_j, fx_k, fy_k, fz_k,
                         mx_k, my_k, mz_k]).T


class Reaction(AttrDisplay):
    """ Reaction forces and moments at a joint.

    Attributes:
        load_pattern (str): Load pattern name.
        joint (str): Joint name.
        fx (float): Force along the x-axis.
        fy (float): Force along the y-axis.
        fz (float): Force along the z-axis.
        mx (float): Moment around the x-axis.
        my (float): Moment around the y-axis.
        mz (float): Moment around the z-axis.

    Methods:
        get_reactions(): Return the reaction load vector.
    """

    def __init__(self, parent, load_pattern, joint, fx=None, fy=None, fz=None,
                 mx=None, my=None, mz=None):
        """ Instantiate a Reaction object.

        Args:
            parent (Structure): Structure object.
            load_pattern (str): Load pattern name.
            joint (str): Joint name.
            fx (float, optional): Force along the x-axis. Defaults to None.
            fy (float, optional): Force along the y-axis. Defaults to None.
            fz (float, optional): Force along the z-axis. Defaults to None.
            mx (float, optional): Moment around the x-axis. Defaults to None.
            my (float, optional): Moment around the y-axis. Defaults to None.
            mz (float, optional): Moment around the z-axis. Defaults to None.
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.joint = joint
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

    def get_reactions(self):
        """ Get reactions.

        Returns:
            ndarray: Forces and moments at a joint.
        """
        return np.array([self.fx, self.fy, self.fz, self.mx, self.my,
                         self.mz])


class InternalForces(AttrDisplay):
    """ Internal forces of an element.

    Attributes:
        load_pattern (str): Load pattern name.
        element (str): Element name.
        fx (list[float], optional): Internal forces along the x-axis.
        fy (list[float], optional): Internal forces along the y-axis.
        fz (list[float], optional): Internal forces along the z-axis.
        mx (list[float], optional): Internal moments around the x-axis.
        my (list[float], optional): Internal moments around the y-axis.
        mz (list[float], optional): Internal moments around the z-axis.
    """

    def __init__(self, parent, load_pattern, element, fx=None, fy=None,
                 fz=None, mx=None, my=None, mz=None):
        """ Instantiate an InternalForces object.

        Args:
            parent (Structure): Structure object.
            load_pattern (str): Load pattern name.
            element (str): Element name.
            fx (list[float], optional): Internal forces along the x-axis.
            fy (list[float], optional): Internal forces along the y-axis.
            fz (list[float], optional): Internal forces along the z-axis.
            mx (list[float], optional): Internal moments around the x-axis.
            my (list[float], optional): Internal moments around the y-axis.
            mz (list[float], optional): Internal moments around the z-axis.
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.element = element
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz


class InternalDisplacements(AttrDisplay):
    """ Internal displacement.

    Attributes:
        load_pattern (str): Load pattern name.
        frame (str): Frame name.
        ux (list[float], optional): Internal displacements along the x-axis.
        uy (list[float], optional): Internal displacements along the y-axis.
        uz (list[float], optional): Internal displacements along the z-axis.
        rx (list[float], optional): Internal displacements around the x-axis.
        ry (list[float], optional): Internal displacements around the y-axis.
        rz (list[float], optional): Internal displacements around the z-axis.
    """

    def __init__(self, parent, load_pattern, frame, ux=None, uy=None,
                 uz=None, rx=None, ry=None, rz=None):
        """ Instantiate an InternalDisplacements object.

        Args:
            parent (Structure): Structure.
            load_pattern (str): Load pattern name.
            frame (str): Frame name.
            ux (list[float], optional): Internal displacements along the
                x-axis.
            uy (list[float], optional): Internal displacements along the
                y-axis.
            uz (list[float], optional): Internal displacements along the
                z-axis.
            rx (list[float], optional): Internal displacements around the
                x-axis.
            ry (list[float], optional): Internal displacements around the
                y-axis.
            rz (list[float], optional): Internal displacements around the
                z-axis.
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.frame = frame
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.rx = rx
        self.ry = ry
        self.rz = rz
