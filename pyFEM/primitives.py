import numpy as np

from numpy import linalg

from scipy.spatial import distance
from scipy.spatial.transform import Rotation

from scipy.sparse import bsr_matrix, coo_matrix

from pyFEM.classtools import AttrDisplay, UniqueInstances


class Material(AttrDisplay):
    """
    Linear elastic material.

    Attributes
    ----------
    E : float
        Young's modulus.
    G : float
        Shear modulus.
    """
    __slots__ = ('E', 'G')

    def __init__(self, E=0, G=0):
        """
        Instantiate a Material object.

        Parameters
        ----------
        E : float
            Young's modulus.
        G : float
            Shear modulus.
        """
        self.E = E
        self.G = G


class Section(AttrDisplay):
    """
    Cross section.

    Attributes
    ----------
    A : float
        Area.
    Ix : float
        Inertia around x-axis.
    Iy : float
        Inertia around y-axis.
    Iz : float
        Inertia around z-axis.
    """
    __slots__ = ('A', 'Iy', 'Iz', 'Ix')

    def __init__(self, A=0, Ix=0, Iy=0, Iz=0):
        """
        Instantiate a Section object.

        Parameters
        ----------
        A : float
            Area.
        Ix : float
            Inertia around x-axis.
        Iy : float
            Inertia around y-axis.
        Iz : float
            Inertia around z-axis.
        """
        self.A = A
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz


class RectangularSection(Section):
    """
    Rectangular cross section.

    Attributes
    ----------
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
    __slots__ = ('width', 'height')

    def __init__(self, width, height):
        """
        Instantiate a RectangularSection object.

        Parameters
        ----------
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

        super().__init__(A, Ix, Iy, Iz)


class Joint(AttrDisplay, metaclass=UniqueInstances):
    """
    End of frames.

    Attributes
    ----------
    x : float
        X coordinate.
    y : float
        Y coordinate.
    z : float
        Z coordinate.

    Methods
    -------
    get_coordinate()
        Return joint coordinate.
    """
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0, y=0, z=0):
        """
        Instantiate a Joint object.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.
        z : float
            Z coordinate.
        """
        self.x = x
        self.y = y
        self.z = z

    def get_coordinate(self):
        """Get coordinate"""
        return np.array([self.x, self.y, self.z])


class Frame(AttrDisplay, metaclass=UniqueInstances):
    """
    Long elements in comparison to their cross section.

    Attributes
    ----------
    joint_j : Joint
        Near Joint.
    joint_k : Joint
        Far Joint.
    material : Material
        Material.
    section : Section
        Section.

    Methods
    -------
    get_length()
        Get length.
    get_direction_cosines()
        Get direction cosines.
    get_rotation()
        Get Rotation.
    get_matrix_rotation(active_joint_displacements)
        Get matrix rotation.
    get_local_stiffness_matrix(active_joint_displacements)
        Get local stiffness matrix.
    get_global_stiffness_matrix(active_joint_displacements)
        Get global stiffness matrix.
    """
    __slots__ = ('joint_j', 'joint_k', 'material', 'section')

    def __init__(self, joint_j=None, joint_k=None, material=None, section=None):
        """
        Instantiate a Frame object.

        Parameters
        ----------
        joint_j : Joint
            Near Joint.
        joint_k : Joint
            Far Joint.
        material : Material
            Material.
        section : Section
            Section.
        """
        self.joint_j = joint_j
        self.joint_k = joint_k
        self.material = material
        self.section = section

    def get_length(self):
        """Get length"""
        return distance.euclidean(self.joint_j.get_coordinate(), self.joint_k.get_coordinate())

    def get_direction_cosines(self):
        """Get direction cosines"""
        vector = self.joint_k.get_coordinate() - self.joint_j.get_coordinate()

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

    def get_matrix_rotation(self, active_joint_displacements):
        """
        Get matrix rotation.

        Parameters
        ----------
        active_joint_displacements : array
            Flags active joint displacements.
        """
        # rotation as direction cosine matrix
        indptr = np.array([0, 1, 2])
        indices = np.array([0, 1])
        data = np.tile(self.get_rotation().as_matrix(), (2, 1, 1))

        # matrix rotation for a joint
        t1 = bsr_matrix((data, indices, indptr), shape=(6, 6)).toarray()

        active_joint_displacements = np.nonzero(active_joint_displacements)[0]
        n = 2 * np.size(active_joint_displacements)
        
        t1 = t1[active_joint_displacements[:, None], active_joint_displacements]
        data = np.tile(t1, (2, 1, 1))

        return bsr_matrix((data, indices, indptr), shape=(n, n)).toarray()

    def get_local_stiffness_matrix(self, active_joint_displacements):
        """
        Get local stiffness matrix.

        Parameters
        ----------
        active_joint_displacements : array
            Flags active joint displacements.
        """
        length = self.get_length()

        e = self.material.E

        iy = self.section.Iy
        iz = self.section.Iz

        el = e / length
        el2 = e / length ** 2
        el3 = e / length ** 3

        ael = self.section.A * el
        gjl = self.section.Ix * self.material.G / length

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
        
        active_frame_displacement = np.nonzero(np.tile(active_joint_displacements, 2))[0]
        
        return k[active_frame_displacement[:, None], active_frame_displacement]

    def get_global_stiffness_matrix(self, active_joint_displacements):
        """
        Get global stiffness matrix.

        Parameters
        ----------
        active_joint_displacements : array
            Flags active joint displacements.
        """
        k = self.get_local_stiffness_matrix(active_joint_displacements)
        t = self.get_matrix_rotation(active_joint_displacements)

        return np.dot(np.dot(t, k), np.transpose(t))

# class Truss(metaclass=UniqueInstances):
#     def get_forces(self, load_pattern):
#         displacements = np.append(self.node_i.displacements[load_pattern].displacement,
#                                   self.node_j.displacements[load_pattern].displacement).reshape(-1, 1)
#         return -np.dot(np.linalg.inv(self.get_matrix_transformation()), np.dot(self.get_global_stiff_matrix(),
#                                                                                displacements))[0, 0]

    def get_internal_force(self, end_actions, load_pattern, no_div=20):
        """
        Get internal force.

        Parameters
        ----------
        no_divs : float, optional
            Number divisions.
        """
        length = self.get_length()
        internal_force = {}

        internal_force['fx'] = np.full(shape=no_div+1, fill_value=-end_actions.fx_j)
        internal_force['fy'] = np.full(shape=no_div+1, fill_value= end_actions.fy_j)
        internal_force['fz'] = np.full(shape=no_div+1, fill_value= end_actions.fz_j)
        internal_force['mx'] = np.full(shape=no_div+1, fill_value=-end_actions.rx_j)
        internal_force['my'] = np.full(shape=no_div+1, fill_value= end_actions.ry_j)
        internal_force['mz'] = np.full(shape=no_div+1, fill_value=-end_actions.rz_j)

        for i in range(no_div+1):
            x = (i / no_div) * length
            internal_force['my'][i] += end_actions.fz_j * x
            internal_force['mz'][i] += end_actions.fy_j * x
        
        if self in load_pattern.distributed_loads:
            distributed_load = load_pattern.distributed_loads[self]
            fx = distributed_load.fx
            fy = distributed_load.fy
            fz = distributed_load.fz
            
            for i in range(no_div+1):
                x = (i / no_div) * length
                internal_force['fx'][i] -= fx * x
                internal_force['fy'][i] += fy * x
                internal_force['fz'][i] += fz * x

                internal_force['my'][i] += fz * x ** 2 / 2
                internal_force['mz'][i] += fy * x ** 2 / 2
        
        if self in load_pattern.point_loads_at_frames:
            point_load = load_pattern.point_loads_at_frames[self]

            fx = point_load.fx
            fy = point_load.fy
            fz = point_load.fz
            mx = point_load.mx
            my = point_load.my
            mz = point_load.mz

            for i in range(no_div+1):
                x = (i / no_div) * length
                internal_force['fx'][i] -= fx[0] if x > fx[1] else 0
                internal_force['fy'][i] += fy[0] if x > fy[1] else 0
                internal_force['fz'][i] += fz[0] if x > fz[1] else 0
                internal_force['mx'][i] -= mx[0] if x > mx[1] else 0
                internal_force['my'][i] += my[0] if x > my[1] else 0
                internal_force['mz'][i] += mz[0] if x > mz[1] else 0

        return internal_force

class Support(AttrDisplay): 
    """
    Point of support.

    Attributes
    ----------
    ux : bool
        Flag translation along x-axis restaint.
    uy : bool
        Flag translation along y-axis restaint.
    uz : bool
        Flag translation along z-axis restaint.
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
    __slots__ = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')

    def __init__(self, ux=False, uy=False, uz=False, rx=False, ry=False, rz=False):
        """
        Instantiate a Support object.

        Parameters
        ----------
        ux : bool
            Flag translation along x-axis restaint.
        uy : bool
            Flag translation along y-axis restaint.
        uz : bool
            Flag translation along z-axis restaint.
        rx : bool
            Flag rotation around x-axis restraint.
        ry : bool
            Flag rotation around y-axis restraint.
        rz : bool
            Flag rotation around z-axis restraint.
        """
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def get_restraints(self, flags_joint_displacements):
        """
        Get flags restraints.

        Parameters
        ----------
        flags_joint_displacements : array
            Flags active joint displacements.
        """
        return np.array([getattr(self, name) for name in self.__slots__])[flags_joint_displacements]


class LoadPattern(AttrDisplay):
    """
    Load pattern.

    Attributes
    ----------
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
    get_number_point_loads_at_joints()
        Get number joint with loads.
    get_number_point_loads_at_frames()
        Get number loads at frames.
    get_number_distributed_loads()
        Get number frames with distributed load.
    get_f(flag_displacements, indexes)
        Get the load vector.
    get_f_fixed(flag_joint_displacements, indexes)
        Get f fixed.
    """
    __slots__ = ("loads_at_joints", "point_loads_at_frames", "distributed_loads")

    def __init__(self):
        """Instantiate a LoadPatter object"""
        self.loads_at_joints = {}
        self.point_loads_at_frames = {}
        self.distributed_loads = {}

    def add_point_load_at_joint(self, joint, *args, **kwargs):
        """
        Add a point load at joint.

        Parameters
        ----------
        joint : Joint
            Joint.
        """
        self.loads_at_joints[joint] = PointLoadAtJoint(*args, **kwargs)
    
    def add_point_load_at_frame(self, frame, *args, **kwargs):
        """
        Add a point load at frame.

        Parameters
        ----------
        frame : Frame
            Frame
        """
        self.point_loads_at_frames[frame] = PointLoadAtFrame(*args, **kwargs)

    def add_distributed_load(self, frame, *args, **kwargs):
        """
        Add a distributed load at frame.

        Parameters
        ----------
        frame : Joint
            Frame.
        """
        self.distributed_loads[frame] = DistributedLoad(*args, **kwargs)

    def get_number_point_loads_at_joints(self):
        """Get number loads at joints"""
        return len(self.loads_at_joints)

    def get_number_point_loads_at_frames(self):
        """Get number point loads at frames"""
        return len(self.point_loads_at_frames)

    def get_number_distributed_loads(self):
        """Get number distributed loads"""
        return len(self.distributed_loads)

    def get_f(self, flag_displacements, indexes):
        """
        Get the load vector.

        Attributes
        ----------
        flag_displacements : array
            Flags active joint displacements.
        indexes : dict
            Key value pairs joints and indexes.
        
        Returns
        -------
        coo_matrix
            Load vector.
        """
        no = np.count_nonzero(flag_displacements)

        n = no * self.get_number_point_loads_at_joints()

        rows = np.empty(n, dtype=int)
        cols = np.zeros(n, dtype=int)
        data = np.empty(n)

        for i, (joint, point_load) in enumerate(self.loads_at_joints.items()):
            rows[i * no:(i + 1) * no] = indexes[joint]
            data[i * no:(i + 1) * no] = point_load.get_load(flag_displacements)

        return coo_matrix((data, (rows, cols)), (no * len(indexes), 1)) - self.get_f_fixed(flag_displacements, indexes)

    def get_f_fixed(self, flag_joint_displacements, indexes):
        """
        Get f fixed.

        Attributes
        ----------
        flag_joint_displacements : array
            Flags active joint displacements.
        indexes : dict
            Key value pairs joints and indexes.
        
        Returns
        -------
        coo_matrix
            f fixed.
        """
        no = np.count_nonzero(flag_joint_displacements)

        # point loads
        n = 2 * no * self.get_number_point_loads_at_frames()

        rows = np.empty(n, dtype=int)
        cols = np.zeros(n, dtype=int)
        data = np.empty(n)

        for i, (frame, point_load) in enumerate(self.point_loads_at_frames.items()):
            joint_j, joint_k = frame.joint_j, frame.joint_k
            
            rows[i * 2 * no:(i + 1) * 2 * no] = np.concatenate((indexes[joint_j], indexes[joint_k]))
            data[i * 2 * no:(i + 1) * 2 * no] = point_load.get_f_fixed(flag_joint_displacements, frame).flatten()

        point_loads = coo_matrix((data, (rows, cols)), (no * len(indexes), 1))
        
        n = self.get_number_distributed_loads()

        rows = np.empty(n * 2 * no, dtype=int)
        cols = np.zeros(n * 2 * no, dtype=int)
        data = np.empty(n * 2 * no)        
        
        for i, (frame, distributed_load) in enumerate(self.distributed_loads.items()):
            joint_j, joint_k = frame.joint_j, frame.joint_k

            rows[i * 2 * no:(i + 1) * 2 * no] = np.concatenate((indexes[joint_j], indexes[joint_k]))
            data[i * 2 * no:(i + 1) * 2 * no] = distributed_load.get_f_fixed(flag_joint_displacements, frame).flatten()

        distributed_loads = coo_matrix((data, (rows, cols)), (no * len(indexes), 1))

        return point_loads + distributed_loads


class PointLoadAtJoint(AttrDisplay):
    """
    Point load at joint.

    Attributes
    ----------
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
     get_load(flag_joint_displacements)
        Get the load vector.
    """
    __slots__ = ('fx', 'fy', 'fz', 'mx', 'my', 'mz')

    def __init__(self, fx=0, fy=0, fz=0, mx=0, my=0, mz=0):
        """
        Instantiate a PointLoadAtJoint object.

        Parameters
        ----------
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
        """
        self.fx = fx
        self.fy = fy
        self.fz = fz

        self.mx = mx
        self.my = my
        self.mz = mz

    def get_load(self, flag_joint_displacements):
        """
        Get load.

        Parameters
        ----------
        flag_joint_displacements : array
            Flags active joint's displacements.
        """
        return np.array([getattr(self, name) for name in self.__slots__])[flag_joint_displacements]

class PointLoadAtFrame(AttrDisplay):
    """
    Point load at frame.

    Attributes
    ----------
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
    __slots__ = ('fx', 'fy', 'fz', 'mx', 'my', 'mz')

    def __init__(self, fx=None, fy=None, fz=None, mx=None, my=None, mz=None):
        """
        Instantiate a PointLoadAtFrame object.

        Parameters
        ----------
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
        self.fx = fx if fx is not None else (0, 0)
        self.fy = fy if fy is not None else (0, 0)
        self.fz = fz if fz is not None else (0, 0)
        self.mx = mx if mx is not None else (0, 0)
        self.my = my if my is not None else (0, 0)
        self.mz = mz if mz is not None else (0, 0)

    def get_f_fixed(self, flag_joint_displacements, frame):
        """
        Get f fixed.

        Parameters
        ----------
        flag_joint_displacements : array
            Flags active joint displacements.
        frame : Frame
            Frame.
        """
        L = frame.get_length()

        f_local = np.empty((2 * 6, 1))

        # fx
        P = self.fx[0]
        a = self.fx[1] * L
        b = L - a

        f_local[0] = -P*b / L
        f_local[6] = -P*a / L

        # fy
        P = -self.fy[0]
        a = self.fy[1] * L
        b = L - a

        f_local[1] = P*b**2*(3*a+b) / L ** 3
        f_local[7] = P*a**2*(a+3*b) / L ** 3

        f_local[5] = P*a*b**2 / L**2
        f_local[11] = -P*a**2*b / L**2

        # fz
        P = -self.fz[0]
        a = self.fz[1] * L
        b = L - a

        f_local[2] = P*b**2*(3*a+b) / L ** 3
        f_local[8] = P*a**2*(a+3*b) / L ** 3

        f_local[4] = -P*a*b**2 / L**2
        f_local[10] = P*a**2*b / L**2

        # mx
        T = self.mx[0]
        a = self.mx[1] * L
        b = L - b

        f_local[3] = -T*b / L
        f_local[9] = -T*a / L

        # my
        M = self.my[0]
        a = self.my[1] * L
        b = L - a

        f_local[2] += -6*M*a*b / L ** 3
        f_local[8] += 6*M*a*b / L **3

        f_local[4] += M*b*(2*a-b) / L ** 2
        f_local[10] += M*a*(2*b-a) / L ** 2

        # mz
        M = self.mz[0]
        a = self.mz[1] * L
        b = L - a

        f_local[1] += 6*M*a*b / L ** 3
        f_local[7] += -6*M*a*b / L ** 3

        f_local[5] += M*b*(2*a-b) / L ** 2
        f_local[11] += M*a*(2*b-a) / L ** 2

        f_local = f_local[np.nonzero(np.tile(flag_joint_displacements, 2)[0])]

        return np.dot(frame.get_rotation_matrix(flag_joint_displacements), f_local)


class DistributedLoad(AttrDisplay):
    """
    Distributed load at frame.

    Attributes
    ----------
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
    __slots__ = ('system', 'fx', 'fy', 'fz')

    def __init__(self, fx=0, fy=0, fz=0):
        """
        Instantiate a DistributedLoad object.

        Parameters
        ----------
        fx : float
            Distributed force along x-axis.
        fy : float
            Distributed force along y-axis.
        fz : float
            Distributed force along z-axis.
        """
        self.system = 'local'

        self.fx = fx
        self.fy = fy
        self.fz = fz

    def get_f_fixed(self, flag_joint_displacements, frame):
        """
        Get f fixed.

        Parameters
        ----------
        flag_joint_displacements : array
            Flags active joint displacements.
        frame : Frame
            Frame.
        """
        length = frame.get_length()

        fx_2 = self.fx * length / 2
        fy_2 = self.fy * length / 2
        fz_2 = self.fz * length / 2

        fy_12 = self.fy * length ** 2 / 12
        fz_12 = self.fz * length ** 2 / 12
        
        f_local = np.array([[-fx_2, -fy_2, -fz_2, 0, fz_12, -fy_12, -fx_2, -fy_2, -fz_2, 0, -fz_12, fy_12]]).T
        
        return np.dot(frame.get_matrix_rotation(flag_joint_displacements), f_local[np.nonzero(np.tile(flag_joint_displacements, 2))[0]])


class Displacement(AttrDisplay):
    """
    Displacement.

    Attributes
    ----------
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
    __slots__ = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')

    def __init__(self, ux=0, uy=0, uz=0, rx=0, ry=0, rz=0):
        """
        Instantiate a Displacement.

        Parameters
        ----------
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
        """
        self.ux = ux
        self.uy = uy
        self.uz = uz

        self.rx = rx
        self.ry = ry
        self.rz = rz

    def get_displacement(self, flag_joint_displacements):
        """Get displacements"""
        return np.array([getattr(self, name) for name in self.__slots__])[flag_joint_displacements]


class Reaction(AttrDisplay):
    """
    Reaction.

    Attributes
    ----------
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
    __slots__ = ('fx', 'fy', 'fz', 'mx', 'my', 'mz')

    def __init__(self, fx=0, fy=0, fz=0, mx=0, my=0, mz=0):
        """
        Instantiate a Reaction.

        Parameters
        ----------
        fx : float
            Force along 'x' axis.
        fy : float
            Force along 'y' axis.
        fz : float
            Force along 'z' axis.
        mx : float
            Moment around 'x' axis.
        my : float
            Moment around 'y' axis.
        mz : float
            Moment around 'z' axis.
        """
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

    def get_reactions(self, flag_joint_displacements):
        """Get reactions"""
        return np.array([getattr(self, name) for name in self.__slots__])[flag_joint_displacements]


class EndActions(AttrDisplay):
    """
    End actions.

    Attributes
    ----------
    fx_j : float
        Force along x-axis at near joint.
    fy_j : float
        Force along y-axis at near joint.
    fz_j : float
        Force along z-axis at near joint.
    rx_j : float
        Moment around x-axis at near joint.
    ry_j : float
        Moment around y-axis at near joint.
    rz_j : float
        Moment around z-axis at near joint.
    fx_k : float
        Force along x-axis at far joint.
    fy_k : float
        Force along y-axis at far joint.
    fz_k : float
        Force along z-axis at far joint.
    rx_k : float
        Moment around x-axis at far joint.
    ry_k : float
        Moment around y-axis at far joint.
    rz_k : float
        Moment around z-axis at far joint.

    Methods
    -------
    get_end_actions()
        Get the end actions vector.
    """
    __slots__ = ("fx_j", "fy_j", "fz_j", "rx_j", "ry_j", "rz_j",
                 "fx_k", "fy_k", "fz_k", "rx_k", "ry_k", "rz_k")

    def __init__(self, fx_j=0, fy_j=0, fz_j=0, rx_j=0, ry_j=0, rz_j=0,
                       fx_k=0, fy_k=0, fz_k=0, rx_k=0, ry_k=0, rz_k=0):
        """
        Instantiate a EndActions object.

        Parameters
        ----------
        fx_j : float
            Force along 'x' axis at joint_j.
        fy_j : float
            Force along 'y' axis at joint_j.
        fz_j : float
            Force along 'z' axis at joint_j.
        rx_j : float
            Moment around 'x' axis at joint_j.
        ry_j : float
            Moment around 'y' axis at joint_j.
        rz_j : float
            Moment around 'z' axis at joint_j.
        fx_k : float
            Force along 'x' axis at joint_k.
        fy_k : float
            Force along 'y' axis at joint_k.
        fz_k : float
            Force along 'z' axis at joint_k.
        rx_k : float
            Moment around 'x' axis at joint_k.
        ry_k : float
            Moment around 'y' axis at joint_k.
        rz_k : float
            Moment around 'z' axis at joint_k.
        """
        self.fx_j = fx_j
        self.fy_j = fy_j
        self.fz_j = fz_j
        self.rx_j = rx_j
        self.ry_j = ry_j
        self.rz_j = rz_j

        self.fx_k = fx_k
        self.fy_k = fy_k
        self.fz_k = fz_k
        self.rx_k = rx_k
        self.ry_k = ry_k
        self.rz_k = rz_k

    def get_end_actions(self, flag_joint_displacements):
        """Get end actions"""  
        return np.array([getattr(self, name) for name in self.__slots__])[np.tile(flag_joint_displacements, 2)].reshape((-1, 1))

class InternalForce(AttrDisplay):
    """
    Internal force.

    Attributes
    ----------
    fx : list
        ...
    fy : list
        ...
    fz : list
        ...
    mx : list
        ...
    my : list
        ...
    mz : list
        ...
    """
    __slots__ = ('fx', 'fy', 'fz', 'mx', 'my', 'mz')

    def __init__(self, fx, fy, fz, mx, my, mz):
        """
        Instantiate a InternalForces object.

        Parameters
        ----------
        fx : list
            ...
        fy : list
            ...
        fz : list
            ...
        mx : list
            ...
        my : list
            ...
        mz : list
            ...
        """
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.mx = mx
        self.my = my
        self.mz = mz

if __name__ == "__main__":
    pass
