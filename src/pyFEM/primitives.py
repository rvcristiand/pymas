import numpy as np

from numpy import linalg
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from scipy.sparse import bsr_matrix, coo_matrix
from pyFEM.classtools import AttrDisplay


class Material(AttrDisplay):
    """
    Linear elastic material.

    Attributes
    ----------
    name : str
        Material name.
    E : float
        Young's modulus.
    G : float
        Shear modulus.
    """

    def __init__(self, parent, name, E=None, G=None):
        """
        Instantiate a Material object.

        Parameters
        ----------
        parent : Structure
            Structure.
        name : str
            Material name.
        E : float, optional
            Young's modulus.
        G : float, optional
            Shear modulus.
        """
        self._parent = parent
        self.name = name
        self.E = E
        self.G = G


class Section(AttrDisplay):
    """
    Cross section.

    Attributes
    ----------
    name : str
        Section name.
    A : float
        Area.
    Ix : float
        Inertia around x-axis.
    Iy : float
        Inertia around y-axis.
    Iz : float
        Inertia around z-axis.
    """

    def __init__(self, parent, name, A=None, Ix=None, Iy=None, Iz=None):
        """
        Instantiate a Section object.

        Parameters
        ----------
        parent : Structure
            Structure.
        name : str
            Section name.
        A : float, optional
            Area.
        Ix : float, optional
            Inertia around x-axis.
        Iy : float, optional
            Inertia around y-axis.
        Iz : float, optional
            Inertia around z-axis.
        """
        self._parent = parent
        self.name = name
        self.A = A
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz


class RectangularSection(Section):
    """
    Rectangular cross section.

    Attributes
    ----------
    name : str
        Section name.
    width : float
        Width.
    height : float
        Height.
    A : float
        Area.
    Ix : float
        Inertia around x-axis.
    Iy : float
        Inertia around y-axis.
    Iz : float
        Inertia around z-axis.
    """

    def __init__(self, parent, name, width, height):
        """
        Instantiate a RectangularSection object.

        Parameters
        ----------
        parent : Structure
            Structure.
        name : str
            Section name.
        width : float
            Width.
        height : float
            Height.
        """
        self.width = width
        self.height = height

        a = min(width, height)
        b = max(width, height)

        A = width * height
        Ix = (1/3 - 0.21 * (a / b) * (1 - (1/12) * (a/b)**4)) * b * a ** 3
        Iy = (1 / 12) * width * height ** 3
        Iz = (1 / 12) * height * width ** 3

        super().__init__(parent, name, A, Ix, Iy, Iz)


class Joint(AttrDisplay):
    """
    End of frames.

    Attributes
    ----------
    name : str
        Joint name.
    x : float
        X coordinate.
    y : float
        Y coordinate.
    z : float
        Z coordinate.

    Methods
    -------
    get_coordinate()
        Get joint coordinate.
    """

    def __init__(self, parent, name, x=None, y=None, z=None):
        """
        Instantiate a Joint object.

        Parameters
        ----------
        parent : Structure
            Structure.
        name : str
            Joint name.
        x : float, optional
            X coordinate.
        y : float, optional
            Y coordinate.
        z : float, optional
            Z coordinate.
        """
        self._parent = parent
        self.name = name
        self.x = x
        self.y = y
        self.z = z

    def get_coordinate(self):
        """Get coordinate"""
        x = self.x if self.x is not None else 0
        y = self.y if self.y is not None else 0
        z = self.z if self.z is not None else 0
        
        return np.array([x, y, z])


class Frame(AttrDisplay):
    """
    Long elements in comparison to their cross section.

    Attributes
    ----------
    name : str
        Frame name.
    joint_j : str
        Near joint name.
    joint_k : str
        Far joint name.
    material : str
        Material name.
    section : str
        Section name.

    Methods
    -------
    get_length()
        Get length.
    get_direction_cosines()
        Get direction cosines.
    get_rotation()
        Get Rotation.
    get_matrix_rotation()
        Get matrix rotation.
    get_local_stiffness_matrix()
        Get local stiffness matrix.
    get_global_stiffness_matrix()
        Get global stiffness matrix.
    get_interntal_forces(load_pattern[, no_div])
        Get internal forces.
    """

    def __init__(self, parent, name, joint_j=None, joint_k=None, material=None, section=None):
        """
        Instantiate a Frame object.

        Parameters
        ----------
        parent : Structure
            Structure.
        name : str
            Frame name.
        joint_j : str, optional
            Near Joint.
        joint_k : str, optional
            Far Joint.
        material : str, optional
            Material.
        section : str, optional
            Section.
        """
        self._parent = parent
        self.name = name
        self.joint_j = joint_j
        self.joint_k = joint_k
        self.material = material
        self.section = section

    def get_length(self):
        """Get length"""
        j = self._parent.joints[self.joint_j]
        k = self._parent.joints[self.joint_k]
        
        return distance.euclidean(j.get_coordinate(), k.get_coordinate())

    def get_direction_cosines(self):
        """Get direction cosines"""
        j = self._parent.joints[self.joint_j]
        k = self._parent.joints[self.joint_k]
        vector = k.get_coordinate() - j.get_coordinate()

        return vector / linalg.norm(vector)

    def get_rotation(self):
        """Get Rotation"""
        v_from = np.array([1, 0, 0])
        v_to = self.get_direction_cosines()

        if np.all(v_from == v_to):
            return Rotation.from_quat([0, 0, 0, 1])

        elif np.all(v_from == -v_to):
            return Rotation.from_quat([0, 0, 1, 0])

        else:
            w = np.cross(v_from, v_to)
            w = w / linalg.norm(w)
            theta = np.arccos(np.dot(v_from, v_to))

            return Rotation.from_quat([x * np.sin(theta/2) for x in w] + [np.cos(theta/2)])

    def get_matrix_rotation(self):
        """Get matrix rotation"""
        
        # rotation as direction cosine matrix
        indptr = np.array([0, 1, 2])
        indices = np.array([0, 1])
        data = np.tile(self.get_rotation().as_matrix(), (2, 1, 1))

        # matrix rotation for a joint
        t1 = bsr_matrix((data, indices, indptr), shape=(6, 6)).toarray()

        active_joint_displacements = np.nonzero(self._parent.get_flags_active_joint_displacements())[0]
        n = 2 * np.size(active_joint_displacements)
        
        t1 = t1[active_joint_displacements[:, None], active_joint_displacements]
        data = np.tile(t1, (2, 1, 1))

        return bsr_matrix((data, indices, indptr), shape=(n, n)).toarray()

    def get_local_stiffness_matrix(self):
        """Get local stiffness matrix"""
        l = self.get_length()

        material = self._parent.materials[self.material]
        e = material.E if material.E is not None else 0
        g = material.G if material.G is not None else 0

        section = self._parent.sections[self.section]
        a = section.A if section.A is not None else 0
        ix = section.Ix if section.Ix is not None else 0
        iy = section.Iy if section.Iy is not None else 0
        iz = section.Iz if section.Iz is not None else 0

        el = e / l
        el2 = e / l ** 2
        el3 = e / l ** 3

        ael = a * el
        gjl = ix * g / l

        e_iy_l = iy * el
        e_iz_l = iz * el

        e_iy_l2 = 6 * iy * el2
        e_iz_l2 = 6 * iz * el2

        e_iy_l3 = 12 * iy * el3
        e_iz_l3 = 12 * iz * el3

        rows = np.empty(40, dtype=int)
        cols = np.empty(40, dtype=int)
        data = np.empty(40)

        # AE / L
        rows[:4] = np.array([0, 6, 0, 6])
        cols[:4] = np.array([0, 6, 6, 0])
        data[:4] = np.array([ael, ael, -ael, -ael])

        # GJ / L
        rows[4:8] = np.array([3, 9, 3, 9])
        cols[4:8] = np.array([3, 9, 9, 3])
        data[4:8] = np.array([gjl, gjl, -gjl, -gjl])

        # 12EI / L^3
        rows[8:12] = np.array([1, 7, 1, 7])
        cols[8:12] = np.array([1, 7, 7, 1])
        data[8:12] = np.array([e_iz_l3, e_iz_l3, -e_iz_l3, -e_iz_l3])

        rows[12:16] = np.array([2, 8, 2, 8])
        cols[12:16] = np.array([2, 8, 8, 2])
        data[12:16] = np.array([e_iy_l3, e_iy_l3, -e_iy_l3, -e_iy_l3])

        # 6EI / L^2
        rows[16:20] = np.array([1, 5, 1, 11])
        cols[16:20] = np.array([5, 1, 11, 1])
        data[16:20] = np.array([e_iz_l2, e_iz_l2, e_iz_l2, e_iz_l2])

        rows[20:24] = np.array([5, 7, 7, 11])
        cols[20:24] = np.array([7, 5, 11, 7])
        data[20:24] = np.array([-e_iz_l2, -e_iz_l2, -e_iz_l2, -e_iz_l2])

        rows[24:28] = np.array([2, 4, 2, 10])
        cols[24:28] = np.array([4, 2, 10, 2])
        data[24:28] = np.array([-e_iy_l2, -e_iy_l2, -e_iy_l2, -e_iy_l2])

        rows[28:32] = np.array([4, 8, 8, 10])
        cols[28:32] = np.array([8, 4, 10, 8])
        data[28:32] = np.array([e_iy_l2, e_iy_l2, e_iy_l2, e_iy_l2])

        # 4EI / L
        rows[32:36] = np.array([4, 10, 5, 11])
        cols[32:36] = np.array([4, 10, 5, 11])
        data[32:36] = np.array([4 * e_iy_l, 4 * e_iy_l, 4 * e_iz_l, 4 * e_iz_l])

        rows[36:] = np.array([10, 4, 11, 5])
        cols[36:] = np.array([4, 10, 5, 11])
        data[36:] = np.array([2 * e_iy_l, 2 * e_iy_l, 2 * e_iz_l, 2 * e_iz_l])

        k = coo_matrix((data, (rows, cols)), shape=(12, 12)).toarray()

        active_joint_displacements = self._parent.get_flags_active_joint_displacements()
        active_frame_displacement = np.nonzero(np.tile(active_joint_displacements, 2))[0]
        
        return k[active_frame_displacement[:, None], active_frame_displacement]

    def get_global_stiffness_matrix(self):
        """Get global stiffness matrix"""
        
        k = self.get_local_stiffness_matrix()
        t = self.get_matrix_rotation()

        return np.dot(np.dot(t, k), np.transpose(t))
        
    def get_internal_forces(self, load_pattern, no_div=100):
        """
        Get internal forces.

        Parameters
        ----------
        load_pattern : str
            Load pattern name.
        no_div : float, optional
            Number divisions.

        Returns
        -------
        internal_forces : dict
            Internal forces.
        """        
        length = self.get_length()
        loadPattern = self._parent.load_patterns[load_pattern]
        end_actions = self._parent.end_actions[load_pattern][self.name]
        fx_j = end_actions.fx_j if end_actions.fx_j is not None else 0
        fy_j = end_actions.fy_j if end_actions.fy_j is not None else 0
        fz_j = end_actions.fz_j if end_actions.fz_j is not None else 0
        mx_j = end_actions.mx_j if end_actions.mx_j is not None else 0
        my_j = end_actions.my_j if end_actions.my_j is not None else 0
        mz_j = end_actions.mz_j if end_actions.mz_j is not None else 0
        fx_k = end_actions.fx_k if end_actions.fx_k is not None else 0
        fy_k = end_actions.fy_k if end_actions.fy_k is not None else 0
        fz_k = end_actions.fz_k if end_actions.fz_k is not None else 0
        mx_k = end_actions.mx_k if end_actions.mx_k is not None else 0
        my_k = end_actions.my_k if end_actions.my_k is not None else 0
        mz_k = end_actions.mz_k if end_actions.mz_k is not None else 0

        internal_forces = {}
        internal_forces['fx'] = np.full(shape=no_div+1, fill_value=-fx_j)
        internal_forces['fy'] = np.full(shape=no_div+1, fill_value= fy_j)
        internal_forces['fz'] = np.full(shape=no_div+1, fill_value= fz_j)
        internal_forces['mx'] = np.full(shape=no_div+1, fill_value=-mx_j)
        internal_forces['my'] = np.full(shape=no_div+1, fill_value= my_j)
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

        if self.name in loadPattern.distributed_loads:
            for distributed_load in loadPattern.distributed_loads[self.name]:
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
        
        if self.name in loadPattern.point_loads_at_frames:
            for point_load in loadPattern.point_loads_at_frames[self.name]:
                fx = point_load.fx if point_load.fx is not None else (0, 0)
                fy = point_load.fy if point_load.fy is not None else (0, 0)
                fz = point_load.fz if point_load.fz is not None else (0, 0)
                mx = point_load.mx if point_load.mx is not None else (0, 0)
                my = point_load.my if point_load.my is not None else (0, 0)
                mz = point_load.mz if point_load.mz is not None else (0, 0)

                for i in range(no_div+1):
                    x = (i / no_div)
                    internal_forces['fx'][i] -= fx[0] if x > fx[1] else 0
                    internal_forces['fy'][i] += fy[0] if x > fy[1] else 0
                    internal_forces['fz'][i] += fz[0] if x > fz[1] else 0
                    internal_forces['mx'][i] -= mx[0] if x > mx[1] else 0
                    internal_forces['my'][i] += my[0] if x > my[1] else 0
                    internal_forces['mz'][i] += mz[0] if x > mz[1] else 0

                    internal_forces['my'][i] += fz[0] * (x - fz[1]) * length if x > fz[1] else 0
                    internal_forces['mz'][i] += fy[0] * (x - fy[1]) * length if x > fy[1] else 0

        flags = self._parent.get_flags_active_joint_displacements()
        internal_forces['fx'] = internal_forces['fx'].tolist() if flags[0] else None
        internal_forces['fy'] = internal_forces['fy'].tolist() if flags[1] else None
        internal_forces['fz'] = internal_forces['fz'].tolist() if flags[2] else None
        internal_forces['mx'] = internal_forces['mx'].tolist() if flags[3] else None
        internal_forces['my'] = internal_forces['my'].tolist() if flags[4] else None
        internal_forces['mz'] = internal_forces['mz'].tolist() if flags[5] else None
        
        return internal_forces
    
    # def get_internal_displacements(self, load_pattern, no_div=10):
    #     """
    #     Get internal displacements.
        
    #     Parameters
    #     ----------
    #     load_pattern : str
    #         Load pattern name.
    #     np_div : float, optional
    #         Number divisions.
    #     """
    #     length = self.get_length()
    #     loadPattern = self._parent.load_patterns[load_pattern]
    #     end_displacements = self._parent.displacements[load ]

    #     internal_displacement['ux'] = 0
    #     internal_displacement['uy'] = np.empty(shape=no_div+1)
    #     internal_displacement['uz'] = 0
    #     internal_displacement['rx'] = 0
    #     internal_displacement['ry'] = 0
    #     internal_displacement['rz'] = 0

    #     return internal_displacement


class Support(AttrDisplay): 
    """
    Point of support.

    Attributes
    ----------
    Joint : str
        Joint name.
    ux : bool
        Flag translation along x-axis restraint.
    uy : bool
        Flag translation along y-axis restraint.
    uz : bool
        Flag translation along z-axis restraint.
    rx : bool
        Flag rotation around x-axis restraint.
    ry : bool
        Flag rotation around y-axis restraint.
    rz : bool
        Flag rotation around z-axis restraint.

    Methods
    -------
    get_restraints()
        Get flags restraints.
    """

    def __init__(self, parent, joint, ux=None, uy=None, uz=None, rx=None, ry=None, rz=None):
        """
        Instantiate a Support object.

        Parameters
        ----------
        parent : Structure
            Structure.
        joint : str
            Joint name.
        ux : bool, optional
            Flag translation along x-axis restaint.
        uy : bool, optional
            Flag translation along y-axis restaint.
        uz : bool, optional
            Flag translation along z-axis restaint.
        rx : bool, optional
            Flag rotation around x-axis restraint.
        ry : bool, optional
            Flag rotation around y-axis restraint.
        rz : bool, optional
            Flag rotation around z-axis restraint.
        """
        self._parent = parent
        self.joint = joint
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def get_restraints(self):
        """Get flags restraints"""        
        ux = self.ux if self.ux is not None else False
        uy = self.uy if self.uy is not None else False
        uz = self.uz if self.uz is not None else False
        rx = self.rx if self.rx is not None else False
        ry = self.ry if self.ry is not None else False
        rz = self.rz if self.rz is not None else False
        
        restrains = np.array([ux, uy, uz, rx, ry, rz])
        
        return restrains[self._parent.get_flags_active_joint_displacements()]


class LoadPattern(AttrDisplay):
    """
    Load pattern.

    Attributes
    ----------
    name : str
        Load pattern name.
    loads_at_joints : dict
        Loads at joints.
    point_loads_at_frames : dict
        Point loads at frames.
    distributed_loads : dict
        Distributed loads at frames.

    Methods
    -------
    add_point_load_at_joint(joint, *args, **kwargs)
        Add a load at joint.
    add_point_load_at_frame(frame, *args, **kwargs)
        Add a point load at frame.
    add_distributed_load(frame, *args, **kwargs)
        Add a distributed load at frame.
    get_f()
        Get the load vector.
    get_f_fixed()
        Get f fixed.
    """

    def __init__(self, parent, name):
        """
        Instantiate a LoadPattern object.

        Parameters
        ----------
        parent : Structure
            Structure
        name : str
            Load pattern name.
        """
        self._parent = parent
        self.name = name
        self.loads_at_joints = {}
        self.point_loads_at_frames = {}
        self.distributed_loads = {}

    def add_point_load_at_joint(self, joint, *args, **kwargs):
        """
        Add a point load at joint.

        Parameters
        ----------
        joint : Joint
            Joint name.

        Returns
        -------
        pointLoad : PointLoadAtJoint
            Point load at joint.
        """
        pointLoad = PointLoadAtJoint(self._parent, self.name, joint, *args, **kwargs)
        
        if joint in self.loads_at_joints:
            self.loads_at_joints[joint].append(pointLoad)
        else:
            self.loads_at_joints[joint] = [pointLoad]

        return pointLoad
    
    def add_point_load_at_frame(self, frame, *args, **kwargs):
        """
        Add a point load at frame.

        Parameters
        ----------
        frame : Frame
            Frame name.
        
        Returns
        -------
        pointLoad : PointLoadAtFrame
            Point load at frame.
        """
        pointLoad = PointLoadAtFrame(self._parent, self.name, frame, *args, **kwargs)
        
        if frame in self.point_loads_at_frames:
            self.point_loads_at_frames[frame].append(pointLoad)
        else:
            self.point_loads_at_frames[frame] = [pointLoad]

        return pointLoad

    def add_distributed_load(self, frame, *args, **kwargs):
        """
        Add a distributed load at frame.

        Parameters
        ----------
        frame : Frame
            Frame name.

        Returns
        -------
        distributedLoad : DistributedLoad
            Distributed load.
        """
        distributedLoad = DistributedLoad(self._parent, self.name, frame, *args, **kwargs)
        
        if frame in self.distributed_loads:
            self.distributed_loads[frame].append(distributedLoad)
        else:
            self.distributed_loads[frame] = [distributedLoad]

        return distributedLoad

    def get_f(self):
        """
        Get the load vector.
        
        Returns
        -------
        coo_matrix
            Load vector.
        """
        no = self._parent.get_number_active_joint_displacements()
        indexes = self._parent.get_indexes()
        n = no * len(self.loads_at_joints)

        rows = np.empty(n, dtype=int)
        cols = np.zeros(n, dtype=int)
        data = np.zeros(n)

        for i, (joint, point_loads) in enumerate(self.loads_at_joints.items()):
            rows[i * no:(i + 1) * no] = indexes[joint]
            for point_load in point_loads:
                data[i * no:(i + 1) * no] += point_load.get_load()

        return coo_matrix((data, (rows, cols)), (no * len(self._parent.joints), 1)) - self.get_f_fixed()

    def get_f_fixed(self):
        """
        Get f fixed.
        
        Returns
        -------
        coo_matrix
            f fixed.
        """
        no = self._parent.get_number_active_joint_displacements()
        indexes = self._parent.get_indexes()

        # point loads
        n = 2 * no * len(self.point_loads_at_frames)

        rows = np.empty(n, dtype=int)
        cols = np.zeros(n, dtype=int)
        data = np.zeros(n)

        for i, (frame, point_loads) in enumerate(self.point_loads_at_frames.items()):
            frame = self._parent.frames[frame]
            joint_j, joint_k = frame.joint_j, frame.joint_k
            
            rows[i * 2 * no:(i + 1) * 2 * no] = np.concatenate((indexes[joint_j], indexes[joint_k]))
            for point_load in point_loads:
                data[i * 2 * no:(i + 1) * 2 * no] += point_load.get_f_fixed().flatten()

        point_loads = coo_matrix((data, (rows, cols)), (no * len(self._parent.joints), 1))

        # distributed loads
        n = 2 * no * len(self.distributed_loads)

        rows = np.empty(n, dtype=int)
        cols = np.zeros(n, dtype=int)
        data = np.zeros(n)        
        
        for i, (frame, distributed_loads) in enumerate(self.distributed_loads.items()):
            frame = self._parent.frames[frame]
            joint_j, joint_k = frame.joint_j, frame.joint_k

            rows[i * 2 * no:(i + 1) * 2 * no] = np.concatenate((indexes[joint_j], indexes[joint_k]))
            for distributed_load in distributed_loads:
                data[i * 2 * no:(i + 1) * 2 * no] += distributed_load.get_f_fixed().flatten()

        distributed_loads = coo_matrix((data, (rows, cols)), (no * len(indexes), 1))

        return point_loads + distributed_loads


class PointLoadAtJoint(AttrDisplay):
    """
    Point load at joint.

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
        Force around x-axis.
    my : float
        Force around y-axis.
    mz : float
        Force around z-axis.

    Methods
    -------
     get_load()
        Get the load vector.
    """

    def __init__(self, parent, load_pattern, joint, fx=None, fy=None, fz=None, mx=None, my=None, mz=None):
        """
        Instantiate a PointLoadAtJoint object.

        Parameters
        ----------
        parent : Structure.
            Structure.
        load_pattern : str
            Load pattern name.
        joint : str
            Joint name.
        fx : float
            Force along x-axis, optional.
        fy : float
            Force along y-axis, optional.
        fz : float
            Force along z-axis, optional.
        mx : float
            Force around x-axis, optional.
        my : float
            Force around y-axis, optional.
        mz : float
            Force around z-axis, optional.
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
        """Get load"""

        fx = self.fx if self.fx is not None else 0
        fy = self.fy if self.fy is not None else 0
        fz = self.fz if self.fz is not None else 0
        mx = self.mx if self.mx is not None else 0
        my = self.my if self.my is not None else 0
        mz = self.mz if self.mz is not None else 0
        
        return np.array([fx, fy, fz, mx, my, mz])[self._parent.get_flags_active_joint_displacements()]


class PointLoadAtFrame(AttrDisplay):
    """
    Point load at frame.

    Attributes
    ----------
    load_pattern : str
        Load pattern name.
    frame : str
        Frame name.
    fx : tuple
        (value, position).
    fy : tuple
        (value, position).
    fz : tuple
        (value, position).
    mx : tuple
        (value, position).
    my : tuple
        (value, position).
    mz : tuple
        (value, position).
    """

    def __init__(self, parent, load_pattern, frame, fx=None, fy=None, fz=None, mx=None, my=None, mz=None):
        """
        Instantiate a PointLoadAtFrame object.

        Parameters
        ----------
        parent : Structure
            Structure.
        load_pattern : str
            Load pattern name.
        frame : str
            Frame name.
        fx : tuple, optional 
            (value, position).
        fy : tuple, optional
            (value, position).
        fz : tuple, optional
            (value, position).
        mx : tuple, optional
            (value, position).
        my : tuple, optional
            (value, position).
        mz : tuple, optional
            (value, position).
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

    def get_f_fixed(self):
        """Get f fixed """
        frame = self._parent.frames[self.frame]
        fx = self.fx if self.fx is not None else (0, 0)
        fy = self.fy if self.fy is not None else (0, 0)
        fz = self.fz if self.fz is not None else (0, 0)
        mx = self.mx if self.mx is not None else (0, 0)
        my = self.my if self.my is not None else (0, 0)
        mz = self.mz if self.mz is not None else (0, 0)
        L = frame.get_length()

        f_local = np.empty((2 * 6, 1))

        # fx
        P = fx[0]
        a = fx[1] * L
        b = L - a

        f_local[0] = -P*b / L
        f_local[6] = -P*a / L

        # fy
        P = -fy[0]
        a = fy[1] * L
        b = L - a

        f_local[1] = P*b**2*(3*a+b) / L ** 3
        f_local[7] = P*a**2*(a+3*b) / L ** 3

        f_local[5] = P*a*b**2 / L**2
        f_local[11] = -P*a**2*b / L**2

        # fz
        P = -fz[0]
        a = fz[1] * L
        b = L - a

        f_local[2] = P*b**2*(3*a+b) / L ** 3
        f_local[8] = P*a**2*(a+3*b) / L ** 3

        f_local[4] = -P*a*b**2 / L**2
        f_local[10] = P*a**2*b / L**2

        # mx
        T = mx[0]
        a = mx[1] * L
        b = L - b

        f_local[3] = -T*b / L
        f_local[9] = -T*a / L

        # my
        M = my[0]
        a = my[1] * L
        b = L - a

        f_local[2] += -6*M*a*b / L ** 3
        f_local[8] += 6*M*a*b / L **3

        f_local[4] += M*b*(2*a-b) / L ** 2
        f_local[10] += M*a*(2*b-a) / L ** 2

        # mz
        M = mz[0]
        a = mz[1] * L
        b = L - a

        f_local[1] += 6*M*a*b / L ** 3
        f_local[7] += -6*M*a*b / L ** 3

        f_local[5] += M*b*(2*a-b) / L ** 2
        f_local[11] += M*a*(2*b-a) / L ** 2

        f_local = f_local[np.tile(self._parent.get_flags_active_joint_displacements(), 2)]

        return np.dot(frame.get_matrix_rotation(), f_local)


class DistributedLoad(AttrDisplay):
    """
    Distributed load at frame.

    Attributes
    ----------
    load_pattern : str
        Load pattern name.
    frame : str
        Frame name.
    system: str
        Coordinate system ('local' by default).
    fx : float
        Distributed force along x-axis.
    fy : float
        Distributed force along y-axis.
    fz : float
        Distributed force along z-axis.

    Methods
    -------
    get_load()
        Get the load vector.
    """

    def __init__(self, parent, load_pattern, frame, fx=None, fy=None, fz=None):
        """
        Instantiate a DistributedLoad object.

        Parameters
        ----------
        parent : Structure
            Structure.
        load_pattern : str
            Load pattern name.
        frame : str
            Frame name.
        fx : float, optional
            Distributed force along x-axis.
        fy : float, optional
            Distributed force along y-axis.
        fz : float, optional
            Distributed force along z-axis.
        """
        self._parent = parent
        self.load_pattern = load_pattern
        self.frame = frame
        self.system = 'local'
        self.fx = fx
        self.fy = fy
        self.fz = fz

    def get_f_fixed(self):
        """Get f fixed"""
        frame = self._parent.frames[self.frame]
        length = frame.get_length()
        
        fx = self.fx if self.fx is not None else 0
        fy = self.fy if self.fy is not None else 0
        fz = self.fz if self.fz is not None else 0

        fx_2 = fx * length / 2
        fy_2 = fy * length / 2
        fz_2 = fz * length / 2

        fy_12 = fy * length ** 2 / 12
        fz_12 = fz * length ** 2 / 12
        
        f_local = np.array([[-fx_2, -fy_2, -fz_2, 0, fz_12, -fy_12, -fx_2, -fy_2, -fz_2, 0, -fz_12, fy_12]]).T
        
        return np.dot(frame.get_matrix_rotation(), f_local[np.tile(self._parent.get_flags_active_joint_displacements(), 2)])


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

    def __init__(self, parent, load_pattern, joint, ux=None, uy=None, uz=None, rx=None, ry=None, rz=None):
        """
        Instantiate a Displacements object.

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

    def get_displacement(self):
        """Get displacements"""
        return np.array([self.ux, self.uy, self.uz, self.rx, self.ry, self.rz])[self._parent.get_flags_active_joint_displacements()]


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

    def __init__(self, parent, load_pattern, frame, fx_j=None, fy_j=None, fz_j=None, mx_j=None, my_j=None, mz_j=None, fx_k=None, fy_k=None, fz_k=None, mx_k=None, my_k=None, mz_k=None):
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
        
        return np.array([self.fx_j, self.fy_j, self.fz_j, self.mx_j, self.my_j, self.mz_j, self.fx_k, self.fy_k, self.fz_k, self.mx_k, self.my_k, self.mz_k])[np.tile(self._parent.get_flags_active_joint_displacements(), 2)]


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

    def __init__(self, parent, load_pattern, joint, fx=None, fy=None, fz=None, mx=None, my=None, mz=None):
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
        return np.array([self.fx, self.fy, self.fz, self.mx, self.my, self.mz])[self._parent.get_flags_active_joint_displacements()]


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

    def __init__(self, parent, load_pattern, frame, fx=None, fy=None, fz=None, mx=None, my=None, mz=None):
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

    def __init__(self, parent, load_pattern, frame, ux=None, uy=None, uz=None, rx=None, ry=None, rz=None):
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


if __name__ == "__main__":
    pass
