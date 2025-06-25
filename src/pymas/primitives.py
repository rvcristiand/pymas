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

    def __init__(self, parent, name, modulus_elasticity=None,
                 modulus_elasticity_shear=None):
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
    Iy : float
        Inertia of the cross section with respect to the local y-axis.
    Iz : float
        Inertia of the cross section with respect to the local z-axis.
    """

    def __init__(self, parent, name, area=None, torsion_constant=None,
                 inertia_y=None, inertia_z=None):
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
        inertia_y : float
            Inertia of the cross section with respect to the local y-axis.
        inertia_z : float
            Inertia of the cross section with respect to the local z-axis.
        """
        self._parent = parent
        self.name = name
        self.A = area
        self.J = torsion_constant
        self.Iy = inertia_y
        self.Iz = inertia_z


class RectangularSection(Section):
    """Rectangular cross section.

    Attributes
    ----------
    name : str
        Name of the cross section.
    base : float
        Base of the cross section.
    height : float
        Height of the cross section.
    A : float
        Area of the cross section.
    J : float
        Torsion constant of the cross section.
    Iy : float
        Inertia of the cross section with respect to the local y-axis.
    Iz : float
        Inertia of the cross section with respect to the local z-axis.
    """

    def __init__(self, parent, name, base, height):
        """Instantiate a RectangularSection object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        name : str
            Name of the rectangular cross section.
        base : float
            Base of the rectangular cross section.
        height : float
            Height of the rectangular cross section.
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
    coordinate_vector()
        Returns the coordinate vector of the joint.
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

    def coordinate_vector(self):
        """Returns the coordinate vector of the joint.

        Returns
        -------
        ndarray
            Coordinate vector of the joint.
        """
        x = self.x if self.x is not None else 0
        y = self.y if self.y is not None else 0
        z = self.z if self.z is not None else 0

        return np.array([x, y, z])


class Truss(AttrDisplay):
    """Long elements interconnected at hinged joints.

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
        Name of the cross section of the truss.

    Methods
    -------
    length()
        Returns the length of the truss.
    direction_cosines_vector()
        Returns the direction cosines vector of the truss.
    rotation_matrix()
        Returns the rotation matrix of the truss.
    rotation_transformation_matrix()
        Returns the rotation transformation matrix of the truss.
    local_stiffness_matrix()
        Returns the local stiffness matrix of the truss.
    global_stiffness_matrix()
        Returns the global stiffness matrix of the truss.
    TODO get_internal_forces(load_pattern[, no_div])
        Returns the internal forces of the truss.
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

    def length(self):
        """Returns the length of the truss.

        Returns
        -------
        float
            Length of the truss.
        """
        j = self._parent.joints[self.joint_j].coordinate_vector()
        k = self._parent.joints[self.joint_k].coordinate_vector()

        return np.linalg.norm(k - j)

    def direction_cosines_vector(self):
        """Returns the direction cosines of the truss.

        Returns
        -------
        ndarray
            Direction cosines of the truss.
        """
        j  = self._parent.joints[self.joint_j].coordinate_vector()
        k = self._parent.joints[self.joint_k].coordinate_vector()
        vector = k - j 

        return vector / np.linalg.norm(vector)

    def rotation_matrix(self):
        """Returns the rotation matrix of the truss.

        Returns
        -------
        ndarray
            Rotation matrix of the truss.
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
        """Returns the rotation transformation matrix of the truss.

        Returns
        -------
        ndarray
            Rotation transformation matrix of the truss.
        """
        indptr = np.array([0, 1, 2, 3, 4])
        indices = np.array([0, 1, 2, 3])
        data = np.tile(self.rotation_matrix(), (4, 1, 1))

        return bsr_matrix((data, indices, indptr), shape=(12, 12)).toarray()

    def local_stiffness_matrix(self):
        """Returns the local stiffness matrix of the truss.

        Returns
        -------
        ndarray
            Local stiffness matrix of the truss.
        """
        L = self.length()

        material = self._parent.materials[self.material]
        E = material.E if material.E is not None else 0

        section = self._parent.sections[self.section]
        A = section.A if section.A is not None else 0

        ael = A * E / L

        # AE / L
        rows = np.array([0, 6, 0, 6])
        cols = np.array([0, 6, 6, 0])
        data = np.array(2 * [ael] + 2 * [-ael])

        return coo_matrix((data, (rows, cols)), (12, 12)).toarray()

    def global_stiffness_matrix(self):
        """Returns the global stiffness matrix of the truss.

        Returns
        -------
        k_global : ndarray
            Global stiffness matrix of the truss.
        """
        # degrees of freedom
        dof = self._parent.get_degrees_freedom()
        # degrees of freedom of the truss
        dof_truss = np.nonzero(np.tile(dof, 2))[0]

        # local siffness matrix of the truss
        k_local = self.local_stiffness_matrix()
        # rotation transformation matrix of the truss
        t = self.rotation_transformation_matrix()
        # global matrix sfiffness of the truss
        k_global = np.dot(np.dot(t, k_local), np.transpose(t))

        return k_global[dof_truss[:, None], dof_truss]


class Frame(Truss):
    """Long elements interconnected at rigid joints.

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
    length()
        Returns the length of the frame.
    direction_cosines_vector()
        Returns the direction cosines of the frame.
    rotation_matrix()
        Returns the rotation matrix of the frame.
    rotation_transformation_matrix()
        Returns the rotation transformation matrix of the frame.
    local_stiffness_matrix()
        Returns the local stiffness matrix of the frame.
    global_stiffness_matrix()
        Returns the global stiffness matrix of the frame.
    TODO get_internal_forces(load_pattern[, no_div])
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
        super().__init__(parent, name, joint_j, joint_k, material, section)

    def local_stiffness_matrix(self):
        """Returns the local stiffness matrix of the frame.

        Returns
        -------
        ndarray
            Local stiffness matrix of the frame.
        """
        L = self.length()

        material = self._parent.materials[self.material]
        E = material.E if material.E is not None else 0
        G = material.G if material.G is not None else 0

        section = self._parent.sections[self.section]
        J = section.J if section.J is not None else 0
        Iy = section.Iy if section.Iy is not None else 0
        Iz = section.Iz if section.Iz is not None else 0

        el = E / L
        el2 = E / L ** 2
        el3 = E / L ** 3

        gjl = J * G / L

        e_iy_l = Iy * el
        e_iz_l = Iz * el

        e_iy_l2 = 6 * Iy * el2
        e_iz_l2 = 6 * Iz * el2

        e_iy_l3 = 12 * Iy * el3
        e_iz_l3 = 12 * Iz * el3

        rows = np.empty(36, dtype=int)
        cols = np.empty(36, dtype=int)
        data = np.empty(36)

        # GJ / L
        rows[:4] = np.array([3, 9, 3, 9])
        cols[:4] = np.array([3, 9, 9, 3])
        data[:4] = np.array(2 * [gjl] + 2 * [-gjl])

        # 12EI / L^3
        rows[4:8] = np.array([1, 7, 1, 7])
        cols[4:8] = np.array([1, 7, 7, 1])
        data[4:8] = np.array(2 * [e_iz_l3] + 2 * [-e_iz_l3])

        rows[8:12] = np.array([2, 8, 2, 8])
        cols[8:12] = np.array([2, 8, 8, 2])
        data[8:12] = np.array(2 * [e_iy_l3] + 2 * [-e_iy_l3])

        # 6EI / L^2
        rows[12:16] = np.array([1, 5, 1, 11])
        cols[12:16] = np.array([5, 1, 11, 1])
        data[12:16] = np.array(4 * [e_iz_l2])

        rows[16:20] = np.array([5, 7, 7, 11])
        cols[16:20] = np.array([7, 5, 11, 7])
        data[16:20] = np.array(4 * [-e_iz_l2])

        rows[20:24] = np.array([2, 4, 2, 10])
        cols[20:24] = np.array([4, 2, 10, 2])
        data[20:24] = np.array(4 * [-e_iy_l2])

        rows[24:28] = np.array([4, 8, 8, 10])
        cols[24:28] = np.array([8, 4, 10, 8])
        data[24:28] = np.array(4 * [e_iy_l2])

        # 4EI / L
        rows[28:32] = np.array([4, 10, 5, 11])
        cols[28:32] = np.array([4, 10, 5, 11])
        data[28:32] = np.array(2 * [4 * e_iy_l] + 2 * [4 * e_iz_l])

        rows[32:] = np.array([10, 4, 11, 5])
        cols[32:] = np.array([4, 10, 5, 11])
        data[32:] = np.array(2 * [2 * e_iy_l] + 2 * [2 * e_iz_l])

        k_truss = super().local_stiffness_matrix()

        return k_truss + coo_matrix((data, (rows, cols)), (12, 12)).toarray()

    def get_internal_forces(self, load_pattern, no_div=100):
        """Get the internal forces of the element.

        Parameters
        ----------
        load_pattern : str
            Name of the load pattern.
        no_div : float, optional
            Number of divisions.

        Returns
        -------
        internal_forces : dict
            Internal forces of the element.
        """
        loadPattern = self._parent.load_patterns[load_pattern]
        endActions = self._parent.end_actions[load_pattern][self.name]

        length = self.length()

        fx_j = endActions.fx_j if endActions.fx_j is not None else 0
        fy_j = endActions.fy_j if endActions.fy_j is not None else 0
        fz_j = endActions.fz_j if endActions.fz_j is not None else 0
        mx_j = endActions.mx_j if endActions.mx_j is not None else 0
        my_j = endActions.my_j if endActions.my_j is not None else 0
        mz_j = endActions.mz_j if endActions.mz_j is not None else 0

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

        if self.name in loadPattern.element_distributed_loads:
            for distributed_load in \
                loadPattern.element_distributed_loads[self.name]:
                fx = distributed_load.fx if distributed_load.fx is not None else 0
                fy = distributed_load.fy if distributed_load.fy is not None else 0
                fz = distributed_load.fz if distributed_load.fz is not None else 0

                for i in range(no_div+1):
                    x = (i / no_div) * length
                    internal_forces['fx'][i] -= fx * x
                    internal_forces['fy'][i] += fy * x
                    internal_forces['fz'][i] += fz * x

                    internal_forces['my'][i] += fz * x ** 2 / 2
                    internal_forces['mz'][i] += fy * x ** 2 / 2

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
            Internal displacements of the element.
        """
        material = self._parent.materials[self.material]
        section = self._parent.sections[self.section]
        loadPattern = self._parent.load_patterns[load_pattern]
        end_actions = self._parent.end_actions[load_pattern][self.name]
        j_joint_displamcement = self._parent.displacements[load_pattern][self.joint_j]

        length = self.length()
        E = material.E if material.E is not None else 0
        G = material.G if material.G is not None else 0
        A = section.A if section.A is not None else 0
        J = section.J if section.J is not None else 0
        Iy = section.Iy if section.Iy is not None else 0
        Iz = section.Iz if section.Iz is not None else 0

        end_actions = self._parent.end_actions[load_pattern][self.name]
        fx_j, fy_j, fz_j, mx_j, my_j, mz_j = end_actions.get_end_actions()[:6]

        j_joint_displamcement = self._parent.displacements[load_pattern][self.joint_j].displacement_vector(
        )
        j_joint_displamcement = np.dot(np.transpose(
            self.rotation_transformation_matrix())[:6, :6], j_joint_displamcement)
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

        if self.name in loadPattern.element_distributed_loads:
            for distributed_load in loadPattern.element_distributed_loads[self.name]:
                fx = distributed_load.fx if distributed_load.fx is not None else 0
                fy = distributed_load.fy if distributed_load.fy is not None else 0
                fz = distributed_load.fz if distributed_load.fz is not None else 0

                for i in range(no_div+1):
                    x = (i / no_div) * length
                    internal_displacements['ux'][i] -= fx * \
                        x ** 2 / (2 * E * A)
                    internal_displacements['uy'][i] += fy * \
                        x ** 4 / (24 * E * Iz)
                    internal_displacements['uz'][i] += fz * \
                        x ** 4 / (24 * E * Iy)

                    internal_displacements['ry'][i] -= fz * \
                        x ** 3 / (6 * E * Iz)
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

    Attributes
    ----------
    joint : str
        Name of the joint.
    r_ux : bool
        Indicates whether the support restrains the displacement of the joint
        along the global x-axis.
    r_uy : bool
        Indicates whether the support restrains the displacement of the joint
        along the global y-axis.
    r_uz : bool
        Indicates whether the support restrains the displacement of the joint
        along the global z-axis.
    r_rx : bool
        Indicates whether the support restrains the displacement of the joint
        around the global x-axis.
    r_ry : bool
        Indicates whether the support restrains the displacement of the joint
        around the global y-axis.
    r_rz : bool
        Indicates whether the support restrains the displacement of the joint
        around the global z-axis.

    Methods
    -------
    restrain_vector()
        Returns the restrain vector of the support.
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
        """
        self._parent = parent
        self.joint = joint
        self.r_ux = r_ux
        self.r_uy = r_uy
        self.r_uz = r_uz
        self.r_rx = r_rx
        self.r_ry = r_ry
        self.r_rz = r_rz

    def restrain_vector(self):
        """Returns the restrain vector of the support.

        Returns
        -------
        restrains : ndarray
            Restrain vector.
        """
        # degrees of freedom
        dof = self._parent.get_degrees_freedom()

        r_ux = self.r_ux if self.r_ux is not None else False
        r_uy = self.r_uy if self.r_uy is not None else False
        r_uz = self.r_uz if self.r_uz is not None else False
        r_rx = self.r_rx if self.r_rx is not None else False
        r_ry = self.r_ry if self.r_ry is not None else False
        r_rz = self.r_rz if self.r_rz is not None else False

        restrains = np.array([r_ux, r_uy, r_uz, r_rx, r_ry, r_rz])

        return restrains[dof]


class LoadPattern(AttrDisplay):
    """Load pattern.

    Attributes
    ----------
    name : str
        Name of the load pattern.
    joint_point_loads : dict
        Joint point loads of the load pattern.
    element_point_loads : dict
        Element point loads of the load pattern.
    element_distributed_loads : dict
        Element uniformly distributed loads of the load pattern.

    Methods
    -------
    add_joint_point_load(joint, [fx, fy, fz, mx, my, mz])
        Add a joint point load to the dictionary of joint point loads.
    add_element_point_load(joint, dist, [fx, fy, fz, mx, my, mz])
        Add a element point load to the dictionary of element point loads.
    add_element_distributed_load(element, [fx, fy, fz, mx, my, mz])
        Add a element distributed load to the dictionary of uniformly
        distributed loads at elements.
    load_vector()
        Returns the load vector of the load pattern.
    actual_load_vector()
        Returns the actual load vector of the load pattern.
    fixed_load_vector()
        Returns the fixed-end load vector of the load pattern.
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
        self.joint_point_loads = {}
        self.element_point_loads = {}
        self.element_distributed_loads = {}

    def add_joint_point_load(self, joint, fx=None, fy=None, fz=None, mx=None,
                             my=None, mz=None):
        """Add a joint point load to the dictionary of joint point loads.

        Parameters
        ----------
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
        pointLoad : JointPointLoad
            Joint point load.
        """
        pointLoad = JointPointLoad(self._parent, self.name, joint, fx, fy, fz,
                                   mx, my, mz)

        try:
            self.joint_point_loads[joint].append(pointLoad)
        except KeyError:
            self.joint_point_loads[joint] = [pointLoad]

        return pointLoad

    def add_element_point_load(self, element, dist, fx=None, fy=None, fz=None,
                               mx=None, my=None, mz=None):
        """Add a element point load to the dictionary of element point loads.

        Parameters
        ----------
        element : str
            Name of the element.
        dist : float
            Distance of the point load from the near joint along the local
            x-axis.
        fx : float, optional
            Intensity of the point load along the local x-axis.
        fy : float, optional
            Intensity of the point load along the local y-axis.
        fz : float, optional
            Intensity of the point load along the local z-axis.
        mx : float, optional
            Intensity of the point load around the local x-axis.
        my : float, optional
            Intensity of the point load around the local y-axis.
        mz : float, optional
            Intensity of the point load around the local z-axis.

        Returns
        -------
        pointLoad : ElementPointLoad
            Element point load.
        """
        pointLoad = ElementPointLoad(self._parent, self.name, element, fx, fy,
                                     fz, mx, my, mz)

        try:
            self.element_point_loads[element].append(pointLoad)
        except KeyError:
            self.element_point_loads[element] = [pointLoad]

        return pointLoad

    def add_element_distributed_load(self, element, fx=None, fy=None, fz=None,
                             mx=None, my=None, mz=None):
        """Add a element uniformly distributed load to the dictionary of
        element uniformly distributed loads.

        Parameters
        ----------
        element : str
            Name of the element.
        fx : float, optional
            Intensity of the uniformly distributed load along the local
            x-axis.
        fy : float, optional
            Intensity of the uniformly distributed load along the local
            y-axis.
        fz : float, optional
            Intensity of the uniformly distributed load along the local
            z-axis.
        mx : float, optional
            Intensity of the uniformly distributed load around the local
            x-axis.
        my : float, optional
            Intensity of the uniformly distributed load around the local
            y-axis.
        mz : float, optional
            Intensity of the uniformly distributed load around the local
            z-axis.

        Returns
        -------
        distributedLoad : DistributedLoad
            DistributedLoad object.
        """
        distributedLoad = DistributedLoad(self._parent, self.name, element, fx,
                                          fy, fz, mx, my, mz)

        try:
            self.element_distributed_loads[element].append(distributedLoad)
        except KeyError:
            self.element_distributed_loads[element] = [distributedLoad]

        return distributedLoad

    def load_vector(self):
        """
        Returns the load vector of the load pattern.

        Returns
        -------
        ndarray
            Load vector of the load pattern.
        """
        return self.actual_load_vector() - self.fixed_load_vector()

    def actual_load_vector(self):
        """Returns the actual load vector of the load pattern.

        Returns
        -------
        ndarray
            Actual load vector of the load pattern.
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

        Returns
        -------
        ndarray
            Fixed-end load vector.
        """
        # number of joints
        n_j = len(self._parent.joints)
        # degrees of freedom
        dof_joints = self._parent.get_degrees_freedom()
        dof_elements = np.tile(dof_joints, 2)
        # number active degrees of freedom
        n_dof = self._parent.number_active_degrees_freedom()
        # number of element distributed loads
        n_distributed_loads = len(self.element_distributed_loads)
        # joint indices of the structure
        j_i = self._parent.get_joint_indices()

        # row positions of the load vectors items of the distributed loads
        rows = np.empty(2 * n_dof * n_distributed_loads, dtype=int)
        cols = np.zeros_like(rows)
        # items of the load vector of the distributed loads
        data = np.zeros_like(rows, dtype=float)

        # assembly the distributed load vector
        for i, (e, d_l) in enumerate(self.element_distributed_loads.items()):
            start = 2 * i * n_dof
            end = 2 * (i + 1) * n_dof

            # element object
            element = self._parent.elements[e]
            joint_j, joint_k = element.joint_j, element.joint_k
            t = element.rotation_transformation_matrix()

            # row positions of the elements of the distributed load at element
            rows[start:end] = np.concatenate((j_i[joint_j], j_i[joint_k]))

            # data of the elements
            for dL in d_l:
                data[start:end] += np.dot(t, dL.fixed_load_vector()).flatten()[dof_elements]

        return coo_matrix((data, (rows, cols)), (n_dof * n_j, 1)).toarray()


class JointPointLoad(AttrDisplay):
    """Joint point load.

    Attributes
    ----------
    load_pattern : str
        Name of the load pattern.
    joint : str
        Name of the joint.
    fx : float
        Intensity of the point load along the global x-axis.
    fy : float
        Intensity of the point load along the global y-axis.
    fz : float
        Intensity of the point load along the global z-axis.
    mx : float
        Intensity of the point load around the global x-axis.
    my : float
        Intensity of the point load around the global y-axis.
    mz : float
        Intensity of the point load around the global z-axis.

    Methods
    -------
    load_vector()
        Returns the load vector of the joint point load.
    """

    def __init__(self, parent, load_pattern, joint, fx=None, fy=None,
                 fz=None, mx=None, my=None, mz=None):
        """Instantiate a JointPointLoad object.

        Parameters
        ----------
        parent : Structure.
            Structure object.
        load_pattern : str
            Name of the load pattern.
        joint : str
            Name of the joint.
        fx : float, optional
            Intensity of the joint point load along the global x-axis.
        fy : float, optional
            Intensity of the joint point load along the global y-axis.
        fz : float, optional
            Intensity of the joint point load along the global z-axis.
        mx : float, optional
            Intensity of the joint point load around the global x-axis.
        my : float, optional
            Intensity of the joint point load around the global y-axis.
        mz : float, optional
            Intensity of the joint point load around the global z-axis.
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

        Returns
        -------
        ndarray
            Load vector.
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


class ElementPointLoad(AttrDisplay):
    """Element point load.

    Attributes
    ----------
    load_pattern : str
        Name of the load pattern.
    element : str
        Name of the element.
    dist : float
        Distance of the point load from the near joint along the local
        x-axis.
    fx : float
        Intensity of the point load along the local x-axis.
    fy : float
        Intensity of the point load along the local y-axis.
    fz : float
        Intensity of the point load along the local z-axis.
    mx : float
        Intensity of the point load around the local x-axis.
    my : float
        Intensity of the point load around the local y-axis.
    mz : float
        Intensity of the point load around the local z-axis.

    Methods
    -------
    fixed_load_vector()
        Returns the fixed-end load vector of the element point load.
    """

    def __init__(self, parent, load_pattern, element, dist, fx=None, fy=None,
                 fz=None, mx=None, my=None, mz=None):
        """Instantiate a ElementPointLoad object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        load_pattern : str
            Name of the load pattern.
        element : str
            Name of the element.
        dist : float
            Distance of the point load from the near joint along the local
            x-axis.
        fx : float, optional
            Intensity of the point load along the local x-axis.
        fy : float, optional
            Intensity of the point load along the local y-axis.
        fz : float, optional
            Intensity of the point load along the local z-axis.
        mx : float, optional
            Intensity of the point load around the local x-axis.
        my : float, optional
            Intensity of the point load around the local y-axis.
        mz : float, optional
            Intensity of the point load around the local z-axis.
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.element = element
        self.dist = dist
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

    def fixed_load_vector(self):
        """Returns the fixed-end load vector of the element point load.

        Returns
        -------
        ndarray
            Fixed-end load vector.
        """
        # uniformly distributed forces
        fx = self.fx if self.fx is not None else 0
        fy = self.fy if self.fy is not None else 0
        fz = self.fz if self.fz is not None else 0
        mx = self.mx if self.mx is not None else 0
        my = self.my if self.my is not None else 0
        mz = self.mz if self.mz is not None else 0

        # dimensions of the element
        L = self._parent.elements[self.element].length()
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
    """Element uniformly distributed load.

    Attributes
    ----------
    load_pattern : str
        Name of the load pattern.
    element : str
        Name of the element.
    fx : float
        Intensity of the uniformly distributed load along the local x-axis.
    fy : float
        Intensity of the uniformly distributed load along the local y-axis.
    fz : float
        Intensity of the uniformly distributed load along the local z-axis.
    mx : float
        Intensity of the uniformly distributed load around the local x-axis.
    my : float
        Intensity of the uniformly distributed load around the local y-axis.
    mz : float
        Intensity of the uniformly distributed load around the local z-axis.

    Methods
    -------
    fixed_load_vector()
        Returns the fixed-end load vector of the distributed load.
    """

    def __init__(self, parent, load_pattern, element, fx=None, fy=None,
                 fz=None, mx=None, my=None, mz=None):
        """Instantiate a DistributedLoad object.

        Parameters
        ----------
        parent : Structure
            Structure object.
        load_pattern : str
            Name of the load pattern.
        element : str
            Name of the element.
        fx : float, optional
            Intensity of the uniformly distributed load along the local
            x-axis.
        fy : float, optional
            Intensity of the uniformly distributed load along the local
            y-axis.
        fz : float, optional
            Intensity of the uniformly distributed load along the local
            z-axis.
        mx : float, optional
            Intensity of the uniformly distributed load around the local
            x-axis.
        my : float, optional
            Intensity of the uniformly distributed load around the local
            y-axis.
        mz : float, optional
            Intensity of the uniformly distributed load around the local
            z-axis.
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

    def fixed_load_vector(self):
        """Returns the fixed-end load vector of the distributed load.

        Returns
        -------
        load_vector[flags_dof_element] : ndarray
            Fixed-end load vector.

        Raises
        ------
        NotImplementedError
            If uniformly distributed moments around the y- or z-axis (my, mz)
            are set.
        """
        # degrees of freedom
        dof_joints = self._parent.get_degrees_freedom()
        dof_element = np.nonzero(np.tile(dof_joints, 2))[0]

        # uniformly distributed forces
        fx = self.fx if self.fx is not None else 0
        fy = self.fy if self.fy is not None else 0
        fz = self.fz if self.fz is not None else 0
        mx = self.mx if self.mx is not None else 0
        my = self.my if self.my is not None else 0
        mz = self.mz if self.mz is not None else 0 

        L = self._parent.elements[self.element].length() 
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

    Attributes
    ----------
    load_pattern : str
        Name of the load pattern.
    joint : str
        Name of the joint.
    ux : float
        Displacement along x-axis.
    uy : float
        Displacement along y-axis.
    uz : float
        Displacement along z-axis.
    rx : float
        Displacement around x-axis.
    ry : float
        Displacement around y-axis.
    rz : float
        Displacement around z-axis.

    Methods
    -------
    displacement_vector()
        Returns the displacement vector.
    """

    def __init__(self, parent, load_pattern, joint, ux=None, uy=None, uz=None,
                 rx=None, ry=None, rz=None):
        """Instantiate a Displacements object.

        Parameters
        ----------
        parent : Structure
            Structure.
        load_pattern : str
            Name of the load pattern.
        joint : str
            Name of the joint.
        ux : float, optional
            Displacement along x-axis.
        uy : float, optional
            Displacement along y-axis.
        uz : float, optional
            Displacement along z-axis.
        rx : float, optional
            Displacement around x-axis.
        ry : float, optional
            Displacement around y-axis.
        rz : float, optional
            Displacement around z-axis.
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

        Returns
        -------
        ndarray
            Displacement vector.
        """

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
    element : str
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

    def __init__(self, parent, load_pattern, element, fx_j=None, fy_j=None,
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
        element : str
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
        self.element = element
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

        return np.array([fx_j, fy_j, fz_j, mx_j, my_j, mz_j, fx_k, fy_k, fz_k, mx_k, my_k, mz_k]).T


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
    element : str
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

    def __init__(self, parent, load_pattern, element, fx=None, fy=None, fz=None,
                 mx=None, my=None, mz=None):
        """
        Instantiate a InternalForces object.

        Parameters
        ----------
        parent : Structure
            Structure.
        load_pattern : str
            Load pattern name
        element : str
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
        self.element = element
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
    element : str
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

    def __init__(self, parent, load_pattern, element, ux=None, uy=None, uz=None,
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
        self.element = element
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.rx = rx
        self.ry = ry
        self.rz = rz
