import numpy as np

from numpy import linalg

from scipy.spatial import distance
from scipy.spatial.transform import Rotation

from scipy.sparse import bsr_matrix, coo_matrix

from pyFEM.classtools import AttrDisplay, UniqueInstances


class Material(AttrDisplay):
    """
    Linear elastic material

    Attributes
    ----------
    E : float
        Young's modulus.
    G : float
        Shear modulus.
    """
    __slots__ = ('E', 'G')

    def __init__(self, modulus_elasticity=0, shearing_modulus_elasticity=0):
        """
        Instantiate a Material object

        Parameters
        ----------
        modulus_elasticity : float
            Young's modulus.
        shearing_modulus_elasticity : float
            Shear modulus.
        """
        self.E = modulus_elasticity
        self.G = shearing_modulus_elasticity


class Section(AttrDisplay):
    """
    Cross-sectional area

    Attributes
    ----------
    A : float
        Cross-sectional area.
    Ix : float
        Inertia around axis x-x.
    Iy : float
        Inertia around axis y-y.
    Iz : float
        Inertia around axis z-z.
    """
    __slots__ = ('A', 'Iy', 'Iz', 'Ix')

    def __init__(self, area=0, torsion_constant=0, moment_inertia_y=0, moment_inertia_z=0):
        """
        Instantiate a Section object

        Parameters
        ----------
        area : float
            Cross-sectional area.
        torsion_constant : float
            Inertia around axis x-x.
        moment_inertia_y : float
            Inertia around axis y-y.
        moment_inertia_z : float
            Inertia around axis z-z.
        """
        self.A = area
        self.Ix = torsion_constant
        self.Iy = moment_inertia_y
        self.Iz = moment_inertia_z


class RectangularSection(Section):
    """
    Rectangular cross-section

    Attributes
    ----------
    width : float
        Width rectangular cross section.
    height : float
        Height rectangular cross section.
    A : float
        Cross-sectional area.
    Ix : float
        Inertia around axis x-x.
    Iy : float
        Inertia around axis y-y.
    Iz : float
        Inertia around axis z-z.
    """
    __slots__ = ('width', 'height')

    def __init__(self, width, height):
        """
        Instantiate a rectangular section object

        Parameters
        ----------
        width : float
            Width rectangular cross section.
        height : float
            Height rectangular cross section.
        """
        self.width = width
        self.height = height

        a = min(width, height)
        b = max(width, height)

        area = width * height
        torsion_constant = (1/3 - 0.21 * (a / b) * (1 - (1/12) * (a/b)**4)) * b * a ** 3
        moment_inertia_y = (1 / 12) * width * height ** 3
        moment_inertia_z = (1 / 12) * height * width ** 3

        super().__init__(area, torsion_constant, moment_inertia_y, moment_inertia_z)


class Joint(AttrDisplay, metaclass=UniqueInstances):
    """
    End of members

    Attributes
    ----------
    x : float
        asd
    y : float
        asd
    z : float
        asd

    Methods
    -------
    get_coordinate()
        asd
    """
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0, y=0, z=0):
        """
        Instantiate a Joint object

        Parameters
        ----------
        x : float
            asd
        y : float
            asd
        z : float
            asd
        """
        self.x = x
        self.y = y
        self.z = z

    def get_coordinate(self):
        """Get coordinates"""
        return np.array([self.x, self.y, self.z])


class Frame(AttrDisplay, metaclass=UniqueInstances):
    """
    Long elements in comparison to their cross-sectional dimensions

    Attributes
    ----------
    joint_j : Joint
        asd
    joint_k : Joint
        asd
    material : Material
        asd
    section : Section
        asd

    Methods
    -------
    get_length()
        asd
    get_direction_cosines()
        asd
    get_rotation()
        asd
    get_rotation_matrix(active_joint_displacements)
        asd
    get_local_stiffness_matrix(active_joint_displacements)
        asd
    get_global_stiffness_matrix(active_joint_displacements)
        asd
    """
    __slots__ = ("joint_j", "joint_k", "material", "section")

    def __init__(self, joint_j=None, joint_k=None, material=None, section=None):
        """
        Instantiate a Frame object

        Parameters
        ----------
        joint_j : Joint
            asd
        joint_k : Joint
            asd
        material : Material
            asd
        section : Section
            asd
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
        """Get rotation"""
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

    def get_rotation_matrix(self, flag_active_joint_displacements):
        """
        Get rotation matrix

        Parameters
        ----------
        flag_active_joint_displacements : array
            asd
        """
        # rotation as direction cosine matrix
        indptr = np.array([0, 1, 2])
        indices = np.array([0, 1])
        data = np.tile(self.get_rotation().as_dcm(), (2, 1, 1))

        # matrix rotation for a joint
        t1 = bsr_matrix((data, indices, indptr), shape=(6, 6)).tolil()

        flag_active_joint_displacements = np.nonzero(flag_active_joint_displacements)[0]
        n = 2 * np.size(flag_active_joint_displacements)

        t1 = t1[flag_active_joint_displacements[:, None], flag_active_joint_displacements].toarray()
        data = np.tile(t1, (2, 1, 1))

        return bsr_matrix((data, indices, indptr), shape=(n, n)).toarray()

    def get_local_stiffness_matrix(self, active_joint_displacements):
        """
        Get local stiffness matrix


        Parameters
        ----------
        active_joint_displacements : array
            asd
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

        k = coo_matrix((data, (rows, cols)), shape=(12, 12)).tolil()

        active_frame_displacement = np.nonzero(np.tile(active_joint_displacements, 2))[0]

        return k[active_frame_displacement[:, None], active_frame_displacement].toarray()

    def get_global_stiffness_matrix(self, active_joint_displacements):
        """
        Get the global stiffness matrix

        Parameters
        ----------
        active_joint_displacements : array
            asd
        """
        k = self.get_local_stiffness_matrix(active_joint_displacements)
        t = self.get_rotation_matrix(active_joint_displacements)

        return np.dot(np.dot(t, k), np.transpose(t))

# class Truss(metaclass=UniqueInstances):
#     def get_forces(self, load_pattern):
#         displacements = np.append(self.node_i.displacements[load_pattern].displacement,
#                                   self.node_j.displacements[load_pattern].displacement).reshape(-1, 1)
#         return -np.dot(np.linalg.inv(self.get_matrix_transformation()), np.dot(self.get_global_stiff_matrix(),
#                                                                                displacements))[0, 0]


class Support(AttrDisplay):
    """
    Point of support

    Attributes
    ----------
    ux : bool
        asd
    uy : bool
        asd
    uz : bool
        asd
    rx : bool
        asd
    ry : bool
        asd
    rz : bool
        asd

    Methods
    -------
    get_restrains()
        asd
    """
    __slots__ = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')

    def __init__(self, ux=False, uy=False, uz=False, rx=False, ry=False, rz=False):
        """
        Instantiate a Support object

        Parameters
        ----------
        ux : bool
            asd
        uy : bool
            asd
        uz : bool
            asd
        rx : bool
            asd
        ry : bool
            asd
        rz : bool
            asd
        """
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def get_restrains(self, flag_joint_displacements):
        """
        Get restrains

        Attributes
        ----------
        flag_joint_displacements : array
            asd
        """
        return np.array([getattr(self, name) for name in self.__slots__])[flag_joint_displacements]


class LoadPattern(AttrDisplay):
    """
    Load pattern

    Attributes
    ----------
    loads_at_joints : dict
        asd
    distributed_loads : dict
        asd

    Methods
    -------
    add_load_at_joint
        asd
    add_distributed_load
        asd
    get_number_loads_at_joints
        asd
    get_load_vector
        asd
    """
    __slots__ = ("loads_at_joints", "distributed_loads")

    def __init__(self):
        """Instantiate a LoadPatter object"""
        self.loads_at_joints = {}
        # self.point_loads_at_frames = {}
        self.distributed_loads = {}

    def add_point_load_at_joint(self, joint, *args, **kwargs):
        """
        Add a point load at joint

        Parameters
        ----------
        joint : Joint
            asd
        args : list
            asd
        kwargs : dict
            asd
        """
        self.loads_at_joints[joint] = PointLoad(*args, **kwargs)

    def add_distributed_load(self, frame, *args, **kwargs):
        """
        Add a distributed load at frame

        Parameters
        ----------
        frame : Joint
            asd
        args : list
            asd
        kwargs : dict
            asd
        """
        self.distributed_loads[frame] = DistributedLoad(*args, **kwargs)

    def get_number_point_loads_at_joints(self):
        """Get number loads at joints"""
        return len(self.loads_at_joints)

    def get_number_distributed_loads(self):
        """Get number distributed loads"""
        return len(self.distributed_loads)

    def get_f(self, flag_displacements, indexes):
        """
        Get the load vector

        Attributes
        ----------
        flag_displacements : array
            asd
        indexes : dict
            asd
        """
        no = np.count_nonzero(flag_displacements)

        n = self.get_number_point_loads_at_joints()

        rows = np.empty(n * no, dtype=int)
        cols = np.zeros(n * no, dtype=int)
        data = np.empty(n * no)

        for i, (joint, point_load) in enumerate(self.loads_at_joints.items()):
            rows[i * no:(i + 1) * no] = indexes[joint]
            data[i * no:(i + 1) * no] = point_load.get_load(flag_displacements)

        return coo_matrix((data, (rows, cols)), (no * len(indexes), 1)) - self.get_f_fixed(flag_displacements, indexes)

    def get_f_fixed(self, flag_joint_displacements, indexes):
        """
        Get the f fixed.

        Attributes
        ----------
        flag_joint_displacements : array
            asd
        indexes : dict
            asd
        """
        no = np.count_nonzero(flag_joint_displacements)

        n = self.get_number_distributed_loads()

        rows = np.empty(2 * n * no, dtype=int)
        cols = np.zeros(2 * n * no, dtype=int)
        data = np.empty(2 * n * no)

        for i, (frame, distributed_load) in enumerate(self.distributed_loads.items()):
            joint_j = frame.joint_j
            joint_k = frame.joint_k

            rows[i * 2 * no:(i + 1) * 2 * no] = np.concatenate((indexes[joint_j], indexes[joint_k]))
            data[i * 2 * no:(i + 1) * 2 * no] = distributed_load.get_f_fixed(flag_joint_displacements, frame)

        return coo_matrix((data, (rows, cols)), (no * len(indexes), 1))


class PointLoad(AttrDisplay):
    """
    Point load

    Attributes
    ----------
    fx : float
        asd
    fy : float
        asd
    fz : float
        asd
    mx : float
        asd
    my : float
        asd
    mz : float
        asd

    Methods
    -------
    get_load
        asd
    """
    __slots__ = ('fx', 'fy', 'fz', 'mx', 'my', 'mz')

    def __init__(self, fx=0, fy=0, fz=0, mx=0, my=0, mz=0):
        """
        Instantiate a PointLoad object

        Parameters
        ----------
        fx : float
            asd
        fy : float
            asd
        fz : float
            asd
        mx : float
            asd
        my : float
            asd
        mz : float
            asd
        """
        self.fx = fx
        self.fy = fy
        self.fz = fz

        self.mx = mx
        self.my = my
        self.mz = mz

    def get_load(self, flag_joint_displacements):
        """
        Get load

        Parameters
        ----------
        flag_joint_displacements : array
            asd
        """

        return np.array([getattr(self, name) for name in self.__slots__])[flag_joint_displacements]


class DistributedLoad(AttrDisplay):
    """
    Distributed load

    Attributes
    ----------
    fx : float
        asd
    fy : float
        asd
    fz : float
        asd
    mx : float
        asd
    my : float
        asd
    mz : float
        asd

    Methods
    -------
    get_load
        asd
    """
    __slots__ = ('system', 'fx', 'fy', 'fz', 'mx', 'my', 'mz')

    def __init__(self, system='global', fx=0, fy=0, fz=0, mx=0, my=0, mz=0):
        """
        Instantiate a PointLoad object

        Parameters
        ----------
        system: str
            asd
        fx : float
            asd
        fy : float
            asd
        fz : float
            asd
        mx : float
            asd
        my : float
            asd
        mz : float
            asd
        """
        self.system = system

        self.fx = fx
        self.fy = fy
        self.fz = fz

        self.mx = mx
        self.my = my
        self.mz = mz

    def get_f_fixed(self, flag_joint_displacements, frame):
        """
        Get f fixed.

        Parameters
        ----------
        flag_joint_displacements : array
            asd
        frame : Frame
            asd
        """
        length = frame.get_length()

        # fx = self.fx
        fy = self.fy
        fz = self.fz
        # rx = self.rx
        # ry = self.ry
        # rz = self.rz

        f_local = [0, -fy * length / 2, -fz * length / 2, 0, fz * length ** 2 / 12, -fy * length ** 2 / 12]
        f_local += [0, -fy * length / 2, -fz * length / 2, 0, -fz * length ** 2 / 12, fy * length ** 2 / 12]

        return np.dot(frame.get_rotation_matrix(flag_joint_displacements), f_local)


class Displacement(AttrDisplay):
    """
    Displacement

    Attributes
    ----------
    ux : float
        asd
    uy : float
        asd
    uz : float
        asd
    rx : float
        asd
    ry : float
        asd
    rz : float
        asd

    Methods
    -------
    get_displacements
        asd
    """
    __slots__ = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')

    def __init__(self, ux=0, uy=0, uz=0, rx=0, ry=0, rz=0):
        """
        Instantiate a Displacement

        Parameters
        ----------
        ux : float
            asd
        uy : float
            asd
        uz : float
            asd
        rx : float
            asd
        ry : float
            asd
        rz : float
            asd
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
    Reaction

    Attributes
    ----------
    fx : float
        asd
    fy : float
        asd
    fz : float
        asd
    mx : float
        asd
    my : float
        asd
    mz : float
        asd

    Methods
    -------
    get_reactions
        asd
    """
    __slots__ = ('fx', 'fy', 'fz', 'mx', 'my', 'mz')

    def __init__(self, fx=0, fy=0, fz=0, mx=0, my=0, mz=0):
        """
        Instantiate a Reaction

        Parameters
        ----------
        fx : float
            asd
        fy : float
            asd
        fz : float
            asd
        mx : float
            asd
        my : float
            asd
        mz : float
            asd
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


if __name__ == "__main__":
    pass
