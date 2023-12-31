import numpy as np

from scipy.spatial.transform import Rotation
from scipy.sparse import bsr_matrix, coo_matrix
from pymas.classtools import AttrDisplay


class Material(AttrDisplay):
    """Linear elastic material.

    Attributes
    ----------
    name : str
        Name of the material.
    E : float
        Modulus of elasticity of the material.
    G : float
        Modulus of elasticity in shear of the material.
    """

    def __init__(self, parent, name, modulus_elasticity,
                 modulus_elasticity_shear):
        """Instantiate a Material object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        name : str
            Name of the material.
        modulus_elasticity : float
            Modulus of elasticity of the material.
        modulus_elasticity_shear : float
            Modulus of elasticity in shear of the material.
        """
        self._parent = parent
        self.name = name
        self.E = modulus_elasticity
        self.G = modulus_elasticity_shear


class Section(AttrDisplay):
    """Cross section.

    Attributes
    ----------
    name : str
        Name of the cross section.
    A : float
        Area of the cross section.
    J : float
        Torsion constant of the cross section.
    Iyy : float
        Inertia of the cross section with respect to the local y-axis.
    Izz : float
        Inertia of the cross section with respect to the local z-axis.
    """

    def __init__(self, parent, name, area, torsion_constant, inertia_yy,
                 inertia_zz):
        """Instantiate a Section object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        name : str
            Name of the cross section.
        area : float
            Area of the cross section.
        torsion : float
            Torsion constant of the cross section.
        inertia_yy : float
            Inertia of the cross section with respect to the local y-axis.
        inertia_zz : float
            Inertia of the cross section with respect to the local z-axis.
        """
        self._parent = parent
        self.name = name
        self.A = area
        self.J = torsion_constant
        self.Iyy = inertia_yy
        self.Izz = inertia_zz


class RectangularSection(Section):
    """Rectangular cross section.

    Attributes
    ----------
    name : str
        Name of the rectangular cross section.
    base : float
        Base of the rectangular cross section.
    height : float
        Height of the rectangular cross section.
    A : float
        Area of the rectangular cross section.
    J : float
        Torsion constant of the rectangular cross section.
    Iy : float
        Inertia of the rectangular cross section with respect to the local
        y-axis.
    Iz : float
        Inertia of the rectangular cross section with respect to the local
        z-axis.
    """

    def __init__(self, parent, name, base, height):
        """Instantiate a RectangularSection object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        name : str
            Name of the rectangular cross section.
        width : float
            Width of the rectangular cross section.
        height : float
            Height of the rectangular cross section.
        """
        a = min(base, height)
        b = max(base, height)

        A = base * height
        J = (1/3 - 0.21*(a/b)*(1 - (1/12)*(a/b)**4)) * b * a**3
        Iyy = (1/12) * height * base**3
        Izz = (1/12) * base * height**3

        self.base = base
        self.height = height

        super().__init__(parent, name, A, J, Iyy, Izz)


class Joint(AttrDisplay):
    """End of elements.

    Attributes
    ----------
    name : str
        Name of the joint.
    x : float
        Coordinate X of the joint.
    y : float
        Coordinate Y of the joint.
    z : float
        Coordinate Z of the joint.

    Methods
    -------
    get_coordinates()
        Returns the coordinates of the joint.
    """

    def __init__(self, parent, name, x=None, y=None, z=None):
        """Instantiate a Joint object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        name : str
            Name of the joint.
        x : float, optional
            Coordinate X of the joint.
        y : float, optional
            Coordinate Y of the joint.
        z : float, optional
            Coordinate Z of the joint.
        """
        self._parent = parent
        self.name = name
        self.x = x
        self.y = y
        self.z = z

    def get_coordinates(self):
        """Returns the coordinates of the joint.

        Returns
        -------
        ndarray
            Coordinates of the joint.
        """
        x = self.x if self.x is not None else 0
        y = self.y if self.y is not None else 0
        z = self.z if self.z is not None else 0

        return np.array([x, y, z])


class Truss(AttrDisplay):
    """Long elements in comparison to their cross section interconnected at
    hinged joints.

    Attributes
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

    Methods
    -------
    get_length()
        Returns the length of the truss.
    get_direction_cosines()
        Returns the direction cosines of the truss.
    get_rotation()
        Returns the rotation of the truss.
    get_rotation_transformation_matrix()
        Returns the rotation transformation matrix of the truss.
    get_local_stiffness_matrix()
        Returns the local stiffness matrix of the truss.
    get_global_stiffness_matrix()
        Returns the global stiffness matrix of the truss.
    """

    def __init__(self, parent, name, joint_j, joint_k, material, section):
        """Instantiate a Truss object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        name : str
            Name of the truss.
        joint_j : str
            Name of the near joint of the truss.
        joint_k : str
            Name of the far joint of the truss.
        material : str
            Name of the material of the truss.
        section : str
            Name of the cross section of the truss.
        """
        self._parent = parent
        self.name = name
        self.joint_j = joint_j
        self.joint_k = joint_k
        self.material = material
        self.section = section

    def get_length(self):
        """Returns the length of the truss.

        Returns
        -------
        float
            Length of the truss.
        """
        j = self._parent.joints[self.joint_j].get_coordinates()
        k = self._parent.joints[self.joint_k].get_coordinates()

        return np.linalg.norm(k - j)

    def get_direction_cosines(self):
        """Returns the direction cosines of the truss.

        Returns
        -------
        ndarray
            Direction cosines of the truss.
        """
        j = self._parent.joints[self.joint_j].get_coordinates()
        k = self._parent.joints[self.joint_k].get_coordinates()
        vector = k - j

        return vector / np.linalg.norm(vector)

    def get_matrix_rotation(self):
        """Returns the matrix rotation of the truss.

        Returns
        -------
        ndarray
            Matrix rotation of the truss.
        """
        v_from = np.array([1, 0, 0])
        v_to = self.get_direction_cosines()

        if np.all(v_from == v_to):
            return Rotation.from_quat([0, 0, 0, 1]).as_matrix()

        elif np.all(v_from == -v_to):
            return Rotation.from_quat([0, 0, 1, 0]).as_matrix()

        else:
            w = np.cross(v_from, v_to)
            w = w / np.linalg.norm(w)
            theta = np.arccos(np.dot(v_from, v_to))

        return Rotation.from_quat(np.array([*(np.sin(theta/2) * w),
                                            np.cos(theta/2)])).as_matrix()

    def get_rotation_transformation_matrix(self):
        """Returns the rotation transformation matrix of the truss.

        Returns
        -------
        ndarray
            Rotation transformation matrix of the truss.
        """
        indptr = np.array([0, 1, 2, 3, 4])
        indices = np.array([0, 1, 2, 3])
        data = np.tile(self.get_matrix_rotation(), (4, 1, 1))

        return bsr_matrix((data, indices, indptr), shape=(12, 12)).toarray()

    def get_local_stiffness_matrix(self):
        """Returns the local stiffness matrix of the truss.

        Returns
        -------
        ndarray
            Local stiffness matrix of the truss.
        """
        L = self.get_length()

        material = self._parent.materials[self.material]
        E = material.E

        section = self._parent.sections[self.section]
        A = section.A

        ael = A * E / L

        # AE / L
        rows = np.array([0, 6, 0, 6])
        cols = np.array([0, 6, 6, 0])
        data = np.array([ael, ael, -ael, -ael])


        return coo_matrix((data, (rows, cols)), shape=(12, 12)).toarray()

    def get_global_stiffness_matrix(self):
        """Returns the global stiffness matrix of the truss.

        Returns
        -------
        k_global : ndarray
            Global stiffness matrix of the truss.
        """
        # get the local siffness matrix of the frame
        k_local = self.get_local_stiffness_matrix()

        # get the rotation transformation matrix of the frame
        t = self.get_rotation_transformation_matrix()

        # calculate the global matrix sfiffness of the frame
        k_global = np.dot(np.dot(t, k_local), np.transpose(t))

        return k_global


class Frame(Truss):
    """Long elements in comparison to their section interconnected at rigid
    joints.

    Attributes
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
        Name of the cross section of the frame.

    Methods
    -------
    get_length()
        Returns the length of the frame.
    get_direction_cosines()
        Returns the direction cosines of the frame.
    get_rotation()
        Returns the rotation of the frame.
    get_rotation_transformation_matrix()
        Returns the rotation transformation matrix of the frame.
    get_local_stiffness_matrix()
        Returns the local stiffness matrix of the frame.
    get_global_stiffness_matrix()
        Returns the global stiffness matrix of the frame.
    get_internal_forces(load_pattern[, no_div])
        Returns the internal forces of the frame.
    """

    def __init__(self, parent, name, joint_j, joint_k, material, section):
        """Instantiate a Frame object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        name : str
            Name of the frame.
        joint_j : str
            Name of the near joint of the frame.
        joint_k : str
            Name of the far joint of the frame.
        material : str
            Name of the material of the frame.
        section : str
            Name of the cross section of the frame.
        """
        self._parent = parent
        self.name = name
        self.joint_j = joint_j
        self.joint_k = joint_k
        self.material = material
        self.section = section

    def get_local_stiffness_matrix(self):
        """Returns the local stiffness matrix of the frame.

        Returns
        -------
        ndarray
            Local stiffness matrix of the frame.
        """
        L = self.get_length()

        material = self._parent.materials[self.material]
        E = material.E
        G = material.G

        section = self._parent.sections[self.section]
        A = section.A
        J = section.J
        Iyy = section.Iyy
        Izz = section.Izz

        el = E / L
        el2 = E / L ** 2
        el3 = E / L ** 3

        ael = A * el
        gjl = J * G / L

        e_iy_l = Iyy * el
        e_iz_l = Izz * el

        e_iy_l2 = 6 * Iyy * el2
        e_iz_l2 = 6 * Izz * el2

        e_iy_l3 = 12 * Iyy * el3
        e_iz_l3 = 12 * Izz * el3

        rows = np.empty(40, dtype=int)
        cols = np.empty(40, dtype=int)
        data = np.empty(40)

        # AE / L
        rows[:4] = np.array([0, 6, 0, 6])
        cols[:4] = np.array([0, 6, 6, 0])
        data[:4] = np.array(2 * [ael] + 2 * [-ael])

        # GJ / L
        rows[4:8] = np.array([3, 9, 3, 9])
        cols[4:8] = np.array([3, 9, 9, 3])
        data[4:8] = np.array(2 * [gjl] + 2 * [-gjl])

        # 12EI / L^3
        rows[8:12] = np.array([1, 7, 1, 7])
        cols[8:12] = np.array([1, 7, 7, 1])
        data[8:12] = np.array(2 * [e_iz_l3] + 2 * [-e_iz_l3])

        rows[12:16] = np.array([2, 8, 2, 8])
        cols[12:16] = np.array([2, 8, 8, 2])
        data[12:16] = np.array(2 * [e_iy_l3] + 2 * [-e_iy_l3])

        # 6EI / L^2
        rows[16:20] = np.array([1, 5, 1, 11])
        cols[16:20] = np.array([5, 1, 11, 1])
        data[16:20] = np.array(4 * [e_iz_l2])

        rows[20:24] = np.array([5, 7, 7, 11])
        cols[20:24] = np.array([7, 5, 11, 7])
        data[20:24] = np.array(4 * [-e_iz_l2])

        rows[24:28] = np.array([2, 4, 2, 10])
        cols[24:28] = np.array([4, 2, 10, 2])
        data[24:28] = np.array(4 * [-e_iy_l2])

        rows[28:32] = np.array([4, 8, 8, 10])
        cols[28:32] = np.array([8, 4, 10, 8])
        data[28:32] = np.array(4 * [e_iy_l2])

        # 4EI / L
        rows[32:36] = np.array([4, 10, 5, 11])
        cols[32:36] = np.array([4, 10, 5, 11])
        data[32:36] = np.array(2 * [4 * e_iy_l] + 2 * [4 * e_iz_l])

        rows[36:] = np.array([10, 4, 11, 5])
        cols[36:] = np.array([4, 10, 5, 11])
        data[36:] = np.array(2 * [2 * e_iy_l] + 2 * [2 * e_iz_l])

        return coo_matrix((data, (rows, cols)), shape=(12, 12)).toarray()

    def get_internal_forces(self, load_pattern, no_div=100):
        """Get the internal forces of the frame.

        Parameters
        ----------
        load_pattern : str
            Name of the load pattern.
        no_div : float, optional
            Number divisions.

        Returns
        -------
        internal_forces : dict
            Internal forces of the frame.
        """
        loadPattern = self._parent.load_patterns[load_pattern]
        end_actions = self._parent.end_actions[load_pattern][self.name]

        length = self.get_length()

        fx_j = end_actions.fx_j if end_actions.fx_j is not None else 0
        fy_j = end_actions.fy_j if end_actions.fy_j is not None else 0
        fz_j = end_actions.fz_j if end_actions.fz_j is not None else 0
        mx_j = end_actions.mx_j if end_actions.mx_j is not None else 0
        my_j = end_actions.my_j if end_actions.my_j is not None else 0
        mz_j = end_actions.mz_j if end_actions.mz_j is not None else 0

        internal_forces = {}
        internal_forces['fx'] = np.full(shape=no_div+1, fill_value=-fx_j)
        internal_forces['fy'] = np.full(shape=no_div+1, fill_value=fy_j)
        internal_forces['fz'] = np.full(shape=no_div+1, fill_value=fz_j)
        internal_forces['mx'] = np.full(shape=no_div+1, fill_value=-mx_j)
        internal_forces['my'] = np.full(shape=no_div+1, fill_value=my_j)
        internal_forces['mz'] = np.full(shape=no_div+1, fill_value=-mz_j)

        for i in range(no_div+1):
            x = (i / no_div) * length
            internal_forces['my'][i] += fz_j * x
            internal_forces['mz'][i] += fy_j * x

        # internal_forces['fx'][-1] += fx_k
        # internal_forces['fy'][-1] += fy_k
        # internal_forces['fz'][-1] += fz_k
        # internal_forces['mx'][-1] += rx_k
        # internal_forces['my'][-1] += ry_k
        # internal_forces['mz'][-1] += rz_k

        if self.name in loadPattern.uniformly_distributed_loads_at_elements:
            for distributed_load in \
                loadPattern.uniformly_distributed_loads_at_elements[self.name]:
                wx = distributed_load.wx if distributed_load.wx is not None else 0
                wy = distributed_load.wy if distributed_load.wy is not None else 0
                wz = distributed_load.wz if distributed_load.wz is not None else 0

                for i in range(no_div+1):
                    x = (i / no_div) * length
                    internal_forces['fx'][i] -= wx * x
                    internal_forces['fy'][i] += wy * x
                    internal_forces['fz'][i] += wz * x

                    internal_forces['my'][i] += wz * x ** 2 / 2
                    internal_forces['mz'][i] += wy * x ** 2 / 2

        # if self.name in loadPattern.point_loads_at_frames:
        #     for point_load in loadPattern.point_loads_at_frames[self.name]:
        #         wx = point_load.fx if point_load.fx is not None else (0, 0)
        #         wy = point_load.fy if point_load.fy is not None else (0, 0)
        #         wz = point_load.fz if point_load.fz is not None else (0, 0)
        #         mx = point_load.mx if point_load.mx is not None else (0, 0)
        #         my = point_load.my if point_load.my is not None else (0, 0)
        #         mz = point_load.mz if point_load.mz is not None else (0, 0)

        #         for i in range(no_div+1):
        #             x = (i / no_div)
        #             internal_forces['fx'][i] -= fx[0] if x > fx[1] else 0
        #             internal_forces['fy'][i] += fy[0] if x > fy[1] else 0
        #             internal_forces['fz'][i] += fz[0] if x > fz[1] else 0
        #             internal_forces['mx'][i] -= mx[0] if x > mx[1] else 0
        #             internal_forces['my'][i] += my[0] if x > my[1] else 0
        #             internal_forces['mz'][i] -= mz[0] if x > mz[1] else 0

        #             internal_forces['my'][i] += fz[0] * \
        #                 (x - fz[1]) if x > fz[1] else 0
        #             internal_forces['mz'][i] += fy[0] * \
        #                 (x - fy[1]) if x > fy[1] else 0

        internal_forces['fx'] = internal_forces['fx'].tolist()
        internal_forces['fy'] = internal_forces['fy'].tolist()
        internal_forces['fz'] = internal_forces['fz'].tolist()
        internal_forces['mx'] = internal_forces['mx'].tolist()
        internal_forces['my'] = internal_forces['my'].tolist()
        internal_forces['mz'] = internal_forces['mz'].tolist()

        return internal_forces

    def get_internal_displacements(self, load_pattern, no_div=100):
        """Get the internal displacements.

        Parameters
        ----------
        load_pattern : str
            Name of the load pattern.
        np_div : float, optional
            Number divisions.

        Returns
        -------
        internal_displacements : dict
            Internal displacements of the frame.
        """
        material = self._parent.materials[self.material]
        section = self._parent.sections[self.section]
        loadPattern = self._parent.load_patterns[load_pattern]
        end_actions = self._parent.end_actions[load_pattern][self.name]
        j_joint_displamcement = self._parent.displacements[load_pattern][self.joint_j]

        length = self.get_length()
        E = material.E if material.E is not None else 0
        G = material.G if material.G is not None else 0
        A = section.A if section.A is not None else 0
        J = section.J if section.J is not None else 0
        Iy = section.Iyy if section.Iyy is not None else 0
        Iz = section.Izz if section.Izz is not None else 0

        end_actions = self._parent.end_actions[load_pattern][self.name]
        fx_j, fy_j, fz_j, mx_j, my_j, mz_j = end_actions.get_end_actions()[:6]

        j_joint_displamcement = self._parent.displacements[load_pattern][self.joint_j].get_displacements(
        )
        j_joint_displamcement = np.dot(np.transpose(
            self.get_rotation_transformation_matrix())[:6, :6], j_joint_displamcement)
        ux_j, uy_j, uz_j, rx_j, ry_j, rz_j = j_joint_displamcement

        internal_displacements = {}
        internal_displacements['ux'] = np.full(shape=no_div+1, fill_value=ux_j)
        internal_displacements['uy'] = np.full(shape=no_div+1, fill_value=uy_j)
        internal_displacements['uz'] = np.full(shape=no_div+1, fill_value=uz_j)
        internal_displacements['rx'] = np.full(shape=no_div+1, fill_value=rx_j)
        internal_displacements['ry'] = np.full(shape=no_div+1, fill_value=ry_j)
        internal_displacements['rz'] = np.full(shape=no_div+1, fill_value=rz_j)

        for i in range(no_div+1):
            x = (i / no_div) * length
            internal_displacements['ux'][i] -= fx_j * x / (E * A)
            internal_displacements['uy'][i] += fy_j * x ** 3 / (6 * E * Iz)
            internal_displacements['uy'][i] -= mz_j * x ** 2 / (2 * E * Iz)
            internal_displacements['uy'][i] += rz_j * x
            internal_displacements['uz'][i] += fz_j * x ** 3 / (6 * E * Iy)
            internal_displacements['uz'][i] += my_j * x ** 2 / (2 * E * Iy)
            internal_displacements['uz'][i] -= ry_j * x
            internal_displacements['rx'][i] -= mx_j * x / (G * J)
            internal_displacements['ry'][i] -= fz_j * x ** 2 / (2 * E * Iz)
            internal_displacements['ry'][i] -= my_j * x / (E * Iz)
            internal_displacements['rz'][i] += fy_j * x ** 2 / (2 * E * Iz)
            internal_displacements['rz'][i] += mz_j * x / (E * Iy)

        if self.name in loadPattern.uniformly_distributed_loads_at_elements:
            for distributed_load in loadPattern.uniformly_distributed_loads_at_elements[self.name]:
                wx = distributed_load.wx if distributed_load.wx is not None else 0
                wy = distributed_load.wy if distributed_load.wy is not None else 0
                wz = distributed_load.wz if distributed_load.wz is not None else 0

                for i in range(no_div+1):
                    x = (i / no_div) * length
                    internal_displacements['ux'][i] -= wx * \
                        x ** 2 / (2 * E * A)
                    internal_displacements['uy'][i] += wy * \
                        x ** 4 / (24 * E * Iz)
                    internal_displacements['uz'][i] += wz * \
                        x ** 4 / (24 * E * Iy)

                    internal_displacements['ry'][i] -= wz * \
                        x ** 3 / (6 * E * Iz)
                    internal_displacements['rz'][i] += wy * \
                        x ** 3 / (6 * E * Iz)

        # if self.name in loadPattern.point_loads_at_elements:
        #     for point_load in loadPattern.point_loads_at_frames[self.name]:
        #         wx = point_load.fx if point_load.fx is not None else (0, 0)
        #         wy = point_load.fy if point_load.fy is not None else (0, 0)
        #         wz = point_load.fz if point_load.fz is not None else (0, 0)
        #         mx = point_load.mx if point_load.mx is not None else (0, 0)
        #         my = point_load.my if point_load.my is not None else (0, 0)
        #         mz = point_load.mz if point_load.mz is not None else (0, 0)

        #         for i in range(no_div+1):
        #             x = (i / no_div)
        #             internal_displacements['ux'][i] -= fx[0] * \
        #                 x / (E * A) if x > fx[1] else 0
        #             internal_displacements['uy'][i] -= fy[0] * (
        #                 x ** 3 / 6 - fy[1] * x ** 2 / 2) / (E * Iz) if x > fy[1] else 0
        #             internal_displacements['uy'][i] += mz[0] * \
        #                 x ** 2 / (2 * E * Iz) if x > mz[1] else 0
        #             internal_displacements['uz'][i] -= fz[0] * (
        #                 x ** 3 / 6 - fz[1] * x ** 2 / 2) / (E * Iy) if x > fz[1] else 0
        #             internal_displacements['uz'][i] -= my[0] * \
        #                 x ** 2 / (2 * E * Iy) if x > my[1] else 0
        #             internal_displacements['rx'][i] -= mx[0] * \
        #                 x / (G * J) if x > fz[1] else 0
        #             internal_displacements['ry'][i] -= fz[0] * \
        #                 (x ** 2 / 2 - fy[1] * x) / (E * Iz) if x > fy[1] else 0
        #             internal_displacements['ry'][i] -= my[0] * \
        #                 x / (E * Iz) if x > my[1] else 0
        #             internal_displacements['rz'][i] -= fy[0] * \
        #                 (x ** 2 / 2 - fy[1] * x) / (E * Iy) if x > fy[1] else 0
        #             internal_displacements['rz'][i] += mz[0] * \
        #                 x / (E * Iy) if x > mz[1] else 0

        internal_displacements['ux'] = internal_displacements['ux'].tolist()
        internal_displacements['uy'] = internal_displacements['uy'].tolist()
        internal_displacements['uz'] = internal_displacements['uz'].tolist()
        internal_displacements['rx'] = internal_displacements['rx'].tolist()
        internal_displacements['ry'] = internal_displacements['ry'].tolist()
        internal_displacements['rz'] = internal_displacements['rz'].tolist()

        return internal_displacements


class Support(AttrDisplay):
    """Point of support.

    Attributes
    ----------
    joint : str
        Name of the joint.
    r_ux : bool
        Flag to indicate whether the translation of the joint of the support
        along the global x-axis is constrained.
    r_uy : bool
        Flag to indicate whether the translation of the joint of the support
        along the global y-axis is constrained.
    r_uz : bool
        Flag to indicate whether the translation of the joint of the support
        along the global z-axis is constrained.
    r_rx : bool
        Flag to indicate whether the rotation of the joint of the support
        around the global x-axis is constrained.
    r_ry : bool
        Flag to indicate whether the rotation of the joint of the support
        around the global y-axis is constrained.
    r_rz : bool
        Flag to indicate whether the rotation of the joint of the support
        around the global z-axis is constrained.

    Methods
    -------
    get_restraints()
        Get the constraint flags of the support.
    """

    def __init__(self, parent, joint, r_ux=None, r_uy=None, r_uz=None,
                 r_rx=None, r_ry=None, r_rz=None):
        """Instantiate a Support object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        joint : str
            Name of the joint of the support.
        r_ux : bool, optional
            Flag to indicate whether the translation of the joint of the support
            along the global x-axis is constrained.
        r_uy : bool, optional
            Flag to indicate whether the translation of the joint of the support
            along the global y-axis is constrained.
        r_uz : bool, optional
            Flag to indicate whether the translation of the joint of the support
            along the global z-axis is constrained.
        r_rx : bool, optional
            Flag to indicate whether the rotation of the joint of the support
            around the global x-axis is constrained.
        r_ry : bool, optional
            Flag to indicate whether the rotation of the joint of the support
            around the global y-axis is constrained.
        r_rz : bool, optional
            Flag to indicate whether the rotation of the joint of the support
            around the global z-axis is constrained
        """
        self._parent = parent
        self.joint = joint
        self.r_ux = r_ux
        self.r_uy = r_uy
        self.r_uz = r_uz
        self.r_rx = r_rx
        self.r_ry = r_ry
        self.r_rz = r_rz

    def get_restraints(self):
        """Get the constraint flags of the support.

        Returns
        -------
        ndarray
            Constraint flags.
        """
        r_ux = self.r_ux if self.r_ux is not None else False
        r_uy = self.r_uy if self.r_uy is not None else False
        r_uz = self.r_uz if self.r_uz is not None else False
        r_rx = self.r_rx if self.r_rx is not None else False
        r_ry = self.r_ry if self.r_ry is not None else False
        r_rz = self.r_rz if self.r_rz is not None else False

        restrains = np.array([r_ux, r_uy, r_uz, r_rx, r_ry, r_rz])

        return restrains


class LoadPattern(AttrDisplay):
    """Load pattern.

    Attributes
    ----------
    name : str
        Name of the load pattern.
    loads_at_joints : dict
        Point loads at joints of the load pattern.
    uniformly_distributed_loads_at_elements : dict
        Uniformly distributed loads at elements of the load pattern.

    Methods
    -------
    add_point_load_at_joint(joint, [fx, fy, fz, mx, my, mz])
        Add a point load at joint to the dictionary of point loads at joints.
    add_uniformly_distributed_load(frame, [fx, fy, fz])
        Add a uniformly distributed load at element to the dictionary of
        distributed loads at elements.
    get_f()
        Get the load vector.
    get_f_fixed()
        Get f fixed.
    """

    def __init__(self, parent, name):
        """Instantiate a LoadPattern object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        name : str
            Name of the load pattern.
        """
        self._parent = parent
        self.name = name
        self.point_loads_at_joints = {}
        # self.point_loads_at_frames = {}
        self.uniformly_distributed_loads_at_elements = {}

    def add_point_load_at_joint(self, joint, fx=None, fy=None, fz=None,
                                mx=None, my=None, mz=None):
        """Add a point load at joint to the dictionary of point loads at joints.

        Parameters
        ----------
        joint : str
            Name of the joint of the point load to add.
        fx : float, optional
            Force of the point load at joint to add along the global x-axis.
        fy : float, optional
            Force of the point load at joint to add along the global y-axis.
        fz : float, optional
            Force of the point load at joint to add along the global z-axis.
        mx : float, optional
            Force of the point load at joint to add around the global x-axis.
        my : float, optional
            Force of the point load at joint to add around the global y-axis.
        mz : float, optional
            Force of the point load at joint to add around the global z-axis.

        Returns
        -------
        pointLoad : PointLoadAtJoint
            Point load at joint.
        """
        pointLoad = PointLoadAtJoint(self._parent, self.name, joint, fx, fy,
                                     fz, mx, my, mz)

        try:
            self.point_loads_at_joints[joint].append(pointLoad)
        except KeyError:
            self.point_loads_at_joints[joint] = [pointLoad]

        return pointLoad

    # def add_point_load_at_frame(self, frame, fx, fy, fz, mx, my, mz):
    #     """Add a point load at frame to the dictionary of point loads at frames.

    #     Parameters
    #     ----------
    #     frame : Frame
    #         Frame name.
    #     fx : tuple, optional
    #         (value, position).
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
    #     pointLoad = PointLoadAtFrame(self._parent, self.name, frame,
    #                                  fx, fy, fz, mx, my, mz)

    #     if frame in self.point_loads_at_frames:
    #         self.point_loads_at_frames[frame].append(pointLoad)
    #     else:
    #         self.point_loads_at_frames[frame] = [pointLoad]

    #     return pointLoad

    def add_uniformly_distributed_load_at_element(self, element, wx=None,
                                                  wy=None, wz=None):
        """Add a uniformly distributed load at element to the dictionary of
        uniformly distributed loads at the elements of the structure.

        Parameters
        ----------
        element : str
            Name of the element of the uniformly distributed load to add.
        fx : float, optional
            Force of the uniformly distributed load at element to add along the
            local x-axis.
        fy : float, optional
            Force of the uniformly distributed load at element to add along the
            local y-axis.
        fz : float, optional
            Force of the uniformly distributed load at element along the local
            z-axis.

        Returns
        -------
        distributedLoad : DistributedLoad
            Distributed load.
        """
        distributedLoad = UniformlyDistributedLoad(self._parent, self.name,
                                                   element, wx, wy, wz)

        try:
            self.uniformly_distributed_loads_at_elements[element].append(
                distributedLoad)
        except KeyError:
            self.uniformly_distributed_loads_at_elements[element] = \
                [distributedLoad]

        return distributedLoad

    def get_f(self):
        """
        Get the load vector of the load pattern.

        Returns
        -------
        coo_matrix
            Load vector of the load pattern.
        """

        return self.get_f_actual() - self.get_f_fixed()

    def get_f_actual(self):
        """Returns f actual.

        Returns
        -------
        ndarray
            f actual
        """
        # number of joint displacements per joint
        no = 6

        # joint indices of the structure
        joint_indices = self._parent.get_joint_indices()

        # number of joints loaded
        no_joint_loaded = len(self.point_loads_at_joints)

        # row and column positions of the elements of the load vectors of the
        # point loads at joint
        rows = np.empty(no_joint_loaded * no, dtype=int)
        cols = np.zeros(no_joint_loaded * no, dtype=int)
        # data of the elements of the load vectors of the point loads at joint
        data = np.zeros(no_joint_loaded * no)

        # assembly the load vectors of the point loads at joint
        for i, (joint, loads) in enumerate(self.point_loads_at_joints.items()):
            # row positions of the elements of the point load at joint
            rows[i * no:(i + 1) * no] = joint_indices[joint]

            # data of the elements
            for pointLoad in loads:
                data[i * no:(i + 1) * no] += pointLoad.get_load()

        return coo_matrix((data, (rows, cols)),
                          (no * len(self._parent.joints), 1)).toarray()

    def get_f_fixed(self):
        """
        Get f fixed.

        Returns
        -------
        coo_matrix
            f fixed.
        """
        # number displacements per joint
        no = 6

        # joint indices of the structure
        indexes = self._parent.get_joint_indices()

        # # point loads
        # n = 2 * no * len(self.point_loads_at_frames)

        # rows = np.empty(n, dtype=int)
        # cols = np.zeros(n, dtype=int)
        # data = np.zeros(n)

        # for i, (frame, point_loads) in \
        #     enumerate(self.point_loads_at_frames.items()):
        #     frame = self._parent.frames[frame]
        #     joint_j, joint_k = frame.joint_j, frame.joint_k

        #     rows[i * 2 * no:(i + 1) * 2 * no] = \
        #     np.concatenate((indexes[joint_j], indexes[joint_k]))
        #     for point_load in point_loads:
        #         data[i * 2 * no:(i + 1) * 2 * no] += \
        #     point_load.get_f_fixed().flatten()

        # point_loads = coo_matrix((data, (rows, cols)), (no * len(self._parent.joints), 1))

        # distributed loads
        n = 2 * no * len(self.uniformly_distributed_loads_at_elements)

        # row positions of the elements of the load vectors of the
        # distributed loads at joint
        rows = np.empty(n, dtype=int)
        cols = np.zeros(n, dtype=int)
        data = np.zeros(n)

        for i, (element, distributed_loads) in enumerate(
                self.uniformly_distributed_loads_at_elements.items()):
            # element object
            element = self._parent.elements[element]

            # element's joint objects
            joint_j, joint_k = element.joint_j, element.joint_k

            # row positions of the elements of the distributed load at frame
            rows[2 * i * no:2 * (i + 1) * no] = \
                np.concatenate((indexes[joint_j], indexes[joint_k]))

            # data of the elements
            for distributed_load in distributed_loads:
                data[i * 2 * no:(i + 1) * 2 * no] += \
                    distributed_load.get_f_fixed().flatten()

        distributed_loads = coo_matrix((data, (rows, cols)),
                                       (no * len(indexes), 1)).toarray()

        return distributed_loads  # point_loads +


class PointLoadAtJoint(AttrDisplay):
    """Point load at joint.

    Attributes
    ----------
    load_pattern : str
        Name of the load pattern of the point load at joint.
    joint : str
        Name of the joint of the point load at joint.
    fx : float
        Force of the point load at joint along the global x-axis.
    fy : float
        Force of the point load at joint along the global y-axis.
    fz : float
        Force of the point load at joint along the global z-axis.
    mx : float
        Force of the point load at joint around the global x-axis.
    my : float
        Force of the point load at joint around the global y-axis.
    mz : float
        Force of the point load at joint around the global z-axis.

    Methods
    -------
     get_load()
        Get the load vector of the point load at joint.
    """

    def __init__(self, parent, load_pattern, joint,
                 fx=None, fy=None, fz=None, mx=None, my=None, mz=None):
        """Instantiate a PointLoadAtJoint object.

        Parameters
        ----------
        parent : Structure.
            Structure object.
        load_pattern : str
            Name of the load pattern of the point load at joint.
        joint : str
            Name of the joint.
        fx : float, optional
            Force of the point load at joint along the global x-axis.
        fy : float, optional
            Force of the point load at joint along the global y-axis.
        fz : float, optional
            Force of the point load at joint along the global z-axis.
        mx : float, optional
            Force of the point load at joint around the global x-axis.
        my : float, optional
            Force of the point load at joint around the global y-axis.
        mz : float, optional
            Force of the point load at joint around the global z-axis.
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

    def get_load(self):
        """Get the load vector of the point load at joint.

        Returns
        -------
        ndarray
            Load vector of the point load at joint.
        """
        fx = self.fx if self.fx is not None else 0
        fy = self.fy if self.fy is not None else 0
        fz = self.fz if self.fz is not None else 0
        mx = self.mx if self.mx is not None else 0
        my = self.my if self.my is not None else 0
        mz = self.mz if self.mz is not None else 0

        return np.array([fx, fy, fz, mx, my, mz])


# class PointLoadAtFrame(AttrDisplay):
#     """Point load at frame.

#     Attributes
#     ----------
#     load_pattern : str
#         Load pattern name.
#     frame : str
#         Frame name.
#     fx : tuple
#         (value, position).
#     fy : tuple
#         (value, position).
#     fz : tuple
#         (value, position).
#     mx : tuple
#         (value, position).
#     my : tuple
#         (value, position).
#     mz : tuple
#         (value, position).
#     """

#     def __init__(self, parent, load_pattern, frame,
#                  fx=None, fy=None, fz=None, mx=None, my=None, mz=None):
#         """Instantiate a PointLoadAtFrame object.

#         Parameters
#         ----------
#         parent : Structure
#             Structure object.
#         load_pattern : str
#             Load pattern name.
#         frame : str
#             Frame name.
#         fx : tuple, optional
#             (value, position).
#         fy : tuple, optional
#             (value, position).
#         fz : tuple, optional
#             (value, position).
#         mx : tuple, optional
#             (value, position).
#         my : tuple, optional
#             (value, position).
#         mz : tuple, optional
#             (value, position).
#         """
#         self._parent = parent
#         self.load_pattern = load_pattern
#         self.frame = frame
#         self.fx = fx
#         self.fy = fy
#         self.fz = fz
#         self.mx = mx
#         self.my = my
#         self.mz = mz

#     def get_f_fixed(self):
#         """Get f fixed """
#         frame = self._parent.frames[self.frame]
#         fx = self.fx if self.fx is not None else (0, 0)
#         fy = self.fy if self.fy is not None else (0, 0)
#         fz = self.fz if self.fz is not None else (0, 0)
#         mx = self.mx if self.mx is not None else (0, 0)
#         my = self.my if self.my is not None else (0, 0)
#         mz = self.mz if self.mz is not None else (0, 0)
#         L = frame.get_length()

#         f_local = np.empty((2 * 6, 1))

#         # fx
#         P = fx[0]
#         a = fx[1] * L
#         b = L - a

#         f_local[0] = -P*b / L
#         f_local[6] = -P*a / L

#         # fy
#         P = -fy[0]
#         a = fy[1] * L
#         b = L - a

#         f_local[1] = P*b**2*(3*a+b) / L ** 3
#         f_local[7] = P*a**2*(a+3*b) / L ** 3

#         f_local[5] = P*a*b**2 / L**2
#         f_local[11] = -P*a**2*b / L**2

#         # fz
#         P = -fz[0]
#         a = fz[1] * L
#         b = L - a

#         f_local[2] = P*b**2*(3*a+b) / L ** 3
#         f_local[8] = P*a**2*(a+3*b) / L ** 3

#         f_local[4] = -P*a*b**2 / L**2
#         f_local[10] = P*a**2*b / L**2

#         # mx
#         T = mx[0]
#         a = mx[1] * L
#         b = L - b

#         f_local[3] = -T*b / L
#         f_local[9] = -T*a / L

#         # my
#         M = my[0]
#         a = my[1] * L
#         b = L - a

#         f_local[2] += -6*M*a*b / L ** 3
#         f_local[8] += 6*M*a*b / L **3

#         f_local[4] += M*b*(2*a-b) / L ** 2
#         f_local[10] += M*a*(2*b-a) / L ** 2

#         # mz
#         M = mz[0]
#         a = mz[1] * L
#         b = L - a

#         f_local[1] += 6*M*a*b / L ** 3
#         f_local[7] += -6*M*a*b / L ** 3

#         f_local[5] += M*b*(2*a-b) / L ** 2
#         f_local[11] += M*a*(2*b-a) / L ** 2

#         f_local = f_local

#         return np.dot(frame.get_matrix_rotation(), f_local)[np.tile(self._parent.get_flags_active_joint_displacements(), 2)]


class UniformlyDistributedLoad(AttrDisplay):
    """Uniformly distributed load at element.

    Attributes
    ----------
    load_pattern : str
        Name of the load pattern of the uniformly distributed load at element.
    element : str
        Name of the element of the uniformly distributed load at element.
    wx : float
        Force of the uniformly distributed load at element along the local
        x-axis.
    wy : float
        Force of the uniformly distributed load at element along the local
        y-axis.
    wz : float
        Force of the uniformly distributed load at element along the local
        z-axis.

    Methods
    -------
    get_load()
        Get the load vector.
    """

    def __init__(self, parent, load_pattern, element, wx=None, wy=None,
                 wz=None):
        """Instantiate a UniformlyDistributedLoad object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        load_pattern : str
            Name of the load pattern of the uniformly distributed load at frame.
        frame : str
            Name of the element of the uniformly distributed load at frame.
        wx : float, optional
            Force of the uniformly distributed load at element along the local
            x-axis.
        wy : float, optional
            Force of the uniformly distributed load at element along the local
            y-axis.
        wz : float, optional
            Force of the uniformly distributed load at element along the local
            z-axis.
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.element = element
        self.wx = wx
        self.wy = wy
        self.wz = wz

    def get_f_fixed(self):
        """Get f fixed.

        Returns
        -------

        """
        # element object
        element = self._parent.elements[self.element]
        # element's length
        l = element.get_length()

        # uniformly distributed forces
        wx = self.wx if self.wx is not None else 0
        wy = self.wy if self.wy is not None else 0
        wz = self.wz if self.wz is not None else 0

        # support reaction
        wxl_2 = wx * l / 2
        wyl_2 = wy * l / 2
        wzl_2 = wz * l / 2

        wyl2_12 = wy * l ** 2 / 12
        wzl_12 = wz * l ** 2 / 12

        # load vector in local system
        f_local = np.array([[-wxl_2, -wyl_2, -wzl_2, 0,  wzl_12, -wyl2_12,
                             -wxl_2, -wyl_2, -wzl_2, 0, -wzl_12,  wyl2_12]]).T

        # load vector in global system
        f_global = np.dot(element.get_rotation_transformation_matrix(), f_local)

        return f_global


class Displacements(AttrDisplay):
    """
    Displacements.

    Attributes
    ----------
    load_pattern : str
        Load pattern.
    joint : str
        Joint name.
    ux : float
        Translation along x-axis.
    uy : float
        Translation along y-axis.
    uz : float
        Translation along z-axis.
    rx : float
        Rotation around x-axis.
    ry : float
        Rotation around y-axis.
    rz : float
        Rotation around z-axis.

    Methods
    -------
    get_displacements()
        Get the displacement vector.
    """

    def __init__(self, parent, load_pattern, joint, ux=None, uy=None, uz=None,
                 rx=None, ry=None, rz=None):
        """Instantiate a Displacements object.

        Parameters
        ----------
        parent : Structure
            Structure.
        load_pattern : str
            Load pattern name.
        joint : str
            Joint name.
        ux : float, optional
            Translation along x-axis.
        uy : float, optional
            Translation along y-axis.
        uz : float, optional
            Translation along z-axis.
        rx : float, optional
            Rotation around x-axis.
        ry : float, optional
            Rotation around y-axis.
        rz : float, optional
            Rotation around z-axis.
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

    def get_displacements(self):
        """Get displacements"""

        ux = self.ux if self.ux is not None else 0
        uy = self.uy if self.uy is not None else 0
        uz = self.uz if self.uz is not None else 0
        rx = self.rx if self.rx is not None else 0
        ry = self.ry if self.ry is not None else 0
        rz = self.rz if self.rz is not None else 0

        return np.array([ux, uy, uz, rx, ry, rz])


class EndActions(AttrDisplay):
    """
    End actions.

    Attributes
    ----------
    load_pattern : str
        Load pattern name.
    frame : str
        Frame name.
    fx_j : float
        Force along x-axis at near joint.
    fy_j : float
        Force along y-axis at near joint.
    fz_j : float
        Force along z-axis at near joint.
    mx_j : float
        Moment around x-axis at near joint.
    my_j : float
        Moment around y-axis at near joint.
    mz_j : float
        Moment around z-axis at near joint.
    fx_k : float
        Force along x-axis at far joint.
    fy_k : float
        Force along y-axis at far joint.
    fz_k : float
        Force along z-axis at far joint.
    mx_k : float
        Moment around x-axis at far joint.
    my_k : float
        Moment around y-axis at far joint.
    mz_k : float
        Moment around z-axis at far joint.

    Methods
    -------
    get_end_actions()
        Get the end actions vector.
    """

    def __init__(self, parent, load_pattern, frame, fx_j=None, fy_j=None,
                 fz_j=None, mx_j=None, my_j=None, mz_j=None, fx_k=None,
                 fy_k=None, fz_k=None, mx_k=None, my_k=None, mz_k=None):
        """
        Instantiate a EndActions object.

        Parameters
        ----------
        parent : Structure 
            Structure
        load_pattern : str
            Load pattern name.
        frame : str
            Frame name.
        fx_j : float, optional
            Force along x-axis at joint_j.
        fy_j : float, optional
            Force along y-axis at joint_j.
        fz_j : float, optional
            Force along z-axis at joint_j.
        mx_j : float, optionl
            Moment around xaxis at joint_j.
        my_j : float, optional
            Moment around y-axis at joint_j.
        mz_j : float, optional
            Moment around z-axis at joint_j.
        fx_k : float, optional
            Force along x-axis at joint_k.
        fy_k : float, optional
            Force along y-axis at joint_k.
        fz_k : float, optional
            Force along z-axis at joint_k.
        mx_k : float, optional
            Moment around x-axis at joint_k.
        my_k : float, optional
            Moment around y-axis at joint_k.
        mz_k : float, optional
            Moment around z-axis at joint_k.
        """
        self._parent = parent
        self.frame = frame
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
        """Get end actions"""

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

        return np.array([fx_j, fy_j, fz_j, mx_j, my_j, mz_j, fx_k, fy_k, fz_k, mx_k, my_k, mz_k])


class Reaction(AttrDisplay):
    """
    Reaction.

    Attributes
    ----------
    load_pattern : str
        Load pattern name.
    joint : str
        Joint name.
    fx : float
        Force along x-axis.
    fy : float
        Force along y-axis.
    fz : float
        Force along z-axis.
    mx : float
        Moment around x-axis.
    my : float
        Moment around y-axis.
    mz : float
        Moment around z-axis.

    Methods
    -------
    get_reactions()
        Get the load vector.
    """

    def __init__(self, parent, load_pattern, joint, fx=None, fy=None, fz=None,
                 mx=None, my=None, mz=None):
        """
        Instantiate a Reaction.

        Parameters
        ----------
        parent : Structure
            Structure.
        load_pattern : str
            Load pattern name.
        joint : str
            Joint name.
        fx : float, optional
            Force along 'x' axis.
        fy : float, optional
            Force along 'y' axis.
        fz : float, optional
            Force along 'z' axis.
        mx : float, optional
            Moment around 'x' axis.
        my : float, optional
            Moment around 'y' axis.
        mz : float, optional
            Moment around 'z' axis.
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
        """Get reactions"""
        return np.array([self.fx, self.fy, self.fz, self.mx, self.my,
                         self.mz])


class InternalForces(AttrDisplay):
    """
    Internal forces.

    Attributes
    ----------
    load_pattern : str
        Load pattern name.
    frame : str
        Frame name.
    fx : list, optional
        Internal forces along x-axis. 
    fy : list, optional
        Internal forces along y-axis.
    fz : list, optional
        Internal forces along z-axis.
    mx : list, optional
        Interntal forces around x-axis.
    my : list, optional
        Interntal forces around y-axis.
    mz : list, optional
        Interntal forces around z-axis.
    """

    def __init__(self, parent, load_pattern, frame, fx=None, fy=None, fz=None,
                 mx=None, my=None, mz=None):
        """
        Instantiate a InternalForces object.

        Parameters
        ----------
        parent : Structure
            Structure.
        load_pattern : str
            Load pattern name
        frame : str
            Frame name.
        fx : list, optional
            Internal forces along x-axis. 
        fy : list, optional
            Internal forces along y-axis.
        fz : list, optional
            Internal forces along z-axis.
        mx : list, optional
            Interntal forces around x-axis.
        my : list, optional
            Interntal forces around y-axis.
        mz : list, optional
            Interntal forces around z-axis.
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


class InternalDisplacements(AttrDisplay):
    """
    Internal displacement.

    Attributes
    ----------
    load_pattern : str
        Load pattern name.
    frame : str
        Frame name.
    ux : list, optional
        Internal displacements along x-axis.
    uy : list, optional
        Internal displacements along y-axis.
    uz : list, optional
        Internal displacements along z-axis.
    rx : list, optional
        Internal displacements around x-axis.
    ry : list, optional
        Internal displacements around y-axis.
    rz : list, optional
        Internal displacements around z-axis.
    """

    def __init__(self, parent, load_pattern, frame, ux=None, uy=None, uz=None,
                 rx=None, ry=None, rz=None):
        """
        Instantiate a InternalDisplacements object.

        Parameters
        ----------
        parent : Structure
            Structure
        load_pattern : str
            Load pattern name.
        frane : str
            Frame name.
        ux : list, optional
            Internal displacements along x-axis.
        uy : list, optional
            Internal displacements along y-axis.
        uz : list, optional
            Internal displacements along z-axis.
        rx : list, optional
            Internal displacements around x-axis.
        ry : list, optional
            Internal displacements around y-axis.
        rz : list, optional
            Internal displacements around z-axis.
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
