import json
import numpy as np

from pyFEM.primitives import *


class Structure:
    """Model and analyse framed structures.

    Attributes
    ----------
    ux : bool
        Flag translation along x-axis active.
    uy : bool
        Flag translation along y-axis active.
    uz : bool
        Flag translation along z-axis active.
    rx : bool
        Flag rotation around x-axis active.
    ry : bool
        Flag rotation around y-axis active.
    rz : bool
        Flag rotation around z-axis active.
    materials : dict
        Materials.
    sections : dict
        Sections.
    joints : dict
        Joints.
    frames : dict
        Frames.
    supports : dict
        Supports.
    load_patterns : dict
        Load patterns.
    displacements : dict
        Displacements.
    end_actions : dict
        End actions.
    reactions : dict
        Reactions.
    internal_forces: dict
        Internal forces.
    internal_displacements: dict
        Internal displacements.
    
    Methods
    -------
    add_material(name, *args, **kwargs)
        Add a material.
    add_section(name, *args, **kwargs)
        Add a section.
    add_rectangular_section(name, width, height)
        Add a rectangular section.
    add_box_section(name, width, depth, L3, t1, t2, t3, f1h, f1v, f2h, f2v, f3h, f3v, L1)
        Add a box section.
    add_joint(name, *args, **kwargs)
        Add a joint.
    add_frame(name, *args, **kwargs)
        Add a frame.
    add_support(joint, *args, **kwargs)
        Add a support.
    add_load_pattern(name)
        Add a load pattern.
    add_load_at_joint(load_pattern, joint, *args, **kwargs)
        Add a load at joint.
    add_point_load_at_frame(load_pattern, frame, *args, **kwargs)
        Add a point load at frame.
    add_distributed_load(load_pattern, frame, *args, **kwargs)
        Add a distributed load at frame.
    get_flags_active_joint_displacements()
        Get flags active joint displacements.
    set_flags_active_joint_displacements()
        Set flags active joint displacements.
    get_number_active_joint_displacements()
        Get number active joint displacements.
    get_indexes()
        Get joint indexes.
    set_indexes()
        Set joint indexes.
    get_stiffness_matrix()
        Get the stiffness matrix of the structure.
    get_stiffness_matrix_modified_by_supports()
        Get the stiffness matrix of the structure modified by supports.
    set_stiffness_matrix_modified_by_supports()
        Set the stiffness matrix of the structure modified by supports.
    solve_load_pattern(load_pattern)
        Solve a load pattern.
    solve()
        Solve the structure due to a load pattern.
    export(filename)
        Export the model.
    """
    def __init__(self, ux=None, uy=None, uz=None, rx=None, ry=None, rz=None):
        """
        Instantiate a Structure object.

        Parameters
        ----------
        ux : bool, optional
            Flag translation along x-axis active.
        uy : bool, optional
            Flag translation along y-axis active.
        uz : bool, optional
            Flag translation along z-axis active.
        rx : bool, optional
            Flag rotation around x-axis active.
        ry : bool, optional
            Flag rotation around y-axis active.
        rz : bool, optional
            Flag rotation around z-axis active.
        """
        self._active_joint_displacements = np.array([])
        self._indexes = {}
        self._ksupport = np.array([])
        
        # flags active joint displacements
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.rx = rx
        self.ry = ry
        self.rz = rz

        # dict materials and sections
        self.materials = {}
        self.sections = {}

        # dict joints and frames
        self.joints = {}
        self.frames = {}

        # dict supports and load patterns
        self.supports = {}
        self.load_patterns = {}

        # dict displacements, end actions and reactions
        self.displacements = {}
        self.end_actions = {}
        self.reactions = {}

        # dict internal forces and displacements
        self.internal_forces = {}
        self.internal_displacements = {}

    def add_material(self, name, *args, **kwargs):
        """
        Add a material.

        Parameters
        ----------
        name : str
            Material name.
        
        Returns
        -------
        material : Material
            Material.
        """
        material = self.materials[name] = Material(self, name, *args, **kwargs) 

        return material

    def add_section(self, name, *args, **kwargs):
        """
        Add a section.

        Parameters
        ----------
        name : str
            Section name.
        
        Returns
        -------
        section : Section
            Section.
        """
        section = self.sections[name] = Section(self, name, *args, **kwargs)

        return section

    def add_rectangular_section(self, name, width, height):
        """
        Add a rectangular section.

        Parameters
        ----------
        name : str
            Rectangular section name.
        width : float
            Width.
        height : float
            Height.
        
        Returns
        -------
        rect_sect : RectangularSection
            RectangularSection.
        """
        rect_sect = self.sections[name] = RectangularSection(self, name, width, height)

        return rect_sect

    def add_box_section(self, name, width, depth, L3, t1, t2, t3, f1h, f1v, f2h, f2v, f3h, f3v, L1):
        """
        Add a box section.

        Parameters
        ----------
        name : str
            Box section name.
        width : float
            Total width.
        depth : float
            Total depth.
        L3 : float
            Exterior girder bottom offset.
        t1 : float
            Top slab thickness.
        t2 : float
            Bottom slab thickness.
        t3 : float
            Exterior girder thickness.
        f1h : float
            f1 horizontal dimension.
        f1v : float
            f1 vertical dimension.
        f2h : float
            f2 horizontal dimension.
        f2v : float
            f2 vertical dimension.
        f3h : float
            f3 horizontal dimension.
        f3v : float
            f3 vertical dimension.
        L1 : float
            Overhang length.

        Returns
        -------
        box_sect : BoxSection
            BoxSection.
        """
        box_sect = self.sections[name] = BoxSection(self, name, width, depth, L3, t1, t2, t3, f1h, f1v, f2h, f2v, f3h, f3v, L1)

        return box_sect

    def add_joint(self, name, *args, **kwargs):
        """
        Add a joint.

        Parameters
        ----------
        name : str
            Joint name.

        Returns
        -------
        joint : Joint
            Joint.
        """
        joint = self.joints[name] = Joint(self, name, *args, **kwargs)

        return joint

    def add_frame(self, name, *args, **kwargs):
        """
        Add a frame.

        Parameters
        ----------
        name : str
            Frame name.

        Returns
        -------
        frame : Frame
            Frame.
        """
        frame = self.frames[name] = Frame(self, name, *args, **kwargs)

        return frame

    def add_support(self, joint, *args, **kwargs):
        """
        Add a support.

        Parameters
        ----------
        joint : str
            Joint name.

        Returns
        -------
        support : Support
            Support.
        """
        support = self.supports[joint] = Support(self, joint, *args, **kwargs)

        return support
        

    def add_load_pattern(self, name):
        """
        Add a load pattern.

        Parameters
        ----------
        name : str
            Load pattern name.

        Returns
        -------
        loadPattern : LoadPattern.
            Load pattern.
        """
        loadPattern = self.load_patterns[name] = LoadPattern(self, name)

        return loadPattern

    def add_load_at_joint(self, load_pattern, joint, *args, **kwargs):
        """
        Add a point load at joint.

        Parameters
        ----------
        load_pattern : str
            Load pattern name.
        joint : str
            Joint name.

        Returns
        -------
        pointLoad : PointLoadAtJoint
            Point load at joint.
        """
        pointLoad = self.load_patterns[load_pattern].add_point_load_at_joint(joint, *args, **kwargs)

        return pointLoad

    def add_point_load_at_frame(self, load_pattern, frame, *args, **kwargs):
        """
        Add a point load at frame.

        Parameters
        ----------
        load_pattern : str
            Load pattern name.
        frame : str
            Frame name.

        Returns
        -------
        pointLoad : PointLoadAtFrame
            Point load at frame.
        """
        pointLoad = self.load_patterns[load_pattern].add_point_load_frame(frame, *args, **kwargs)

        return pointLoad
        
    def add_distributed_load(self, load_pattern, frame, *args, **kwargs):
        """
        Add a distributed load at frame.

        Parameters
        ----------
        load_pattern : str
            Load pattern name.
        frame : str
            Frame name.
        
        Returns
        -------
        distributedLoad : DistributedLoad
            Distributed loads.
        """
        distributedLoad = self.load_patterns[load_pattern].add_distributed_load(frame, *args, **kwargs)

    def get_flags_active_joint_displacements(self):
        """Get flags active joint displacements.

        Returns
        -------
        array
            Flags active joint displacements.
        """
        return self._active_joint_displacements
        
    def set_flags_active_joint_displacements(self):
        """Set flags active joint displacements"""
        
        ux = self.ux if self.ux is not None else False
        uy = self.uy if self.uy is not None else False
        uz = self.uz if self.uz is not None else False
        rx = self.rx if self.rx is not None else False
        ry = self.ry if self.ry is not None else False
        rz = self.rz if self.rz is not None else False
        
        self._active_joint_displacements = np.array([ux, uy, uz, rx, ry, rz])

    def get_number_active_joint_displacements(self):
        """
        Get number active joint displacements.
        
        Returns
        -------
        int
            Number of active joint displacements.
        """
        return np.count_nonzero(self.get_flags_active_joint_displacements())

    def get_indexes(self):
        """Get joint indexes.

        Returns
        -------
        dict
            Key value pairs joints and indexes.
        """
        return self._indexes
        
    def set_indexes(self):
        """Set joint indexes"""
        n = self.get_number_active_joint_displacements()

        self._indexes = {joint: np.arange(n * i, n * (i + 1)) for i, joint in enumerate(self.joints)}

    def get_stiffness_matrix(self):
        """
        Get stiffness matrix of the structure.
        
        Returns
        -------
        array
            Stiffness matrix of the structure.
        """
        flags_active_joint_displacements = self.get_flags_active_joint_displacements()
        no_active_joint_displacements = self.get_number_active_joint_displacements()
        indexes = self.get_indexes()

        no_joints = len(self.joints)
        no_frames = len(self.frames)

        n = 2 * no_active_joint_displacements
        n_2 = n ** 2

        rows = np.empty(no_frames * n_2, dtype=int)
        cols = np.empty(no_frames * n_2, dtype=int)
        data = np.empty(no_frames * n_2)

        for i, frame in enumerate(self.frames.values()):
            k_element = frame.get_global_stiffness_matrix()
            indexes_element = np.concatenate((indexes[frame.joint_j], indexes[frame.joint_k]))
            indexes_element = np.broadcast_to(indexes_element, (n, n))

            rows[i * n_2:(i + 1) * n_2] = indexes_element.flatten('F')
            cols[i * n_2:(i + 1) * n_2] = indexes_element.flatten()
            data[i * n_2:(i + 1) * n_2] = k_element.flatten()
        
        return coo_matrix((data, (rows, cols)), 2 * (no_active_joint_displacements * no_joints,)).toarray()
    
    def get_stiffness_matrix_modified_by_supports(self):
        """Get stiffness matrix of the structure modified by supports.

        Returns
        -------
        stiffness_matrix_with_supports : array
            Stiffness matrix of the structure modified by supports.
        """
        return self._ksupport
    
    def set_stiffness_matrix_modified_by_supports(self):
        """Set stiffness matrix of the structure modified by supports"""
        
        flags_active_joint_displacements = self.get_flags_active_joint_displacements()
        indexes = self.get_indexes()

        stiffness_matrix = self.get_stiffness_matrix()
        n = np.shape(stiffness_matrix)[0]
        
        for joint, support in self.supports.items():
            joint_indexes = indexes[joint]
            restraints = support.get_restraints()

            for index in joint_indexes[restraints]:
                stiffness_matrix[index] = stiffness_matrix[:, index] = np.zeros(n)
                stiffness_matrix[index, index] = 1
        
        self._ksupport = stiffness_matrix

    def solve_load_pattern(self, load_pattern):
        """
        Solve the structure due to a load pattern.

        Parameters
        ----------
        load_pattern : str
            Load pattern name.
        """
        flags_joint_displacements = self.get_flags_active_joint_displacements()
        indexes = self.get_indexes()
        loadPattern = self.load_patterns[load_pattern]

        f = loadPattern.get_f().toarray()
        f_support = np.copy(f)

        for joint, support in self.supports.items():
            joint_indexes = indexes[joint]
            restrains = support.get_restraints()

            for index in joint_indexes[restrains]:
                f_support[index, 0] = 0
        
        # find displacements
        u = np.linalg.solve(self.get_stiffness_matrix_modified_by_supports(), f_support)
        
        # store displacements
        load_pattern_displacements = {}

        for joint in self.joints:
            joint_indexes = indexes[joint]
            displacements = np.array(6 * [None])
            displacements[flags_joint_displacements] = u[joint_indexes, 0]
            load_pattern_displacements[joint] = Displacements(self, load_pattern, joint, *displacements)

        self.displacements[load_pattern] = load_pattern_displacements

        # store frame end actions
        flags_frame_displacements = np.tile(flags_joint_displacements, 2)
        n = np.count_nonzero(flags_frame_displacements)

        rows = []
        cols = []
        data = []

        load_pattern_end_actions = {}

        for frame, _frame in self.frames.items():
            t = _frame.get_matrix_rotation()
            indexes_element = np.concatenate((indexes[_frame.joint_j], indexes[_frame.joint_k]))

            k_element = _frame.get_local_stiffness_matrix()
            u_element = np.dot(np.transpose(t), u[indexes_element])

            f_fixed = np.zeros((n, 1))

            if frame in loadPattern.point_loads_at_frames:
                for point_load in loadPattern.point_loads_at_frames[frame]:
                    f_fixed += point_load.get_f_fixed()

            if frame in loadPattern.distributed_loads:
                for distributed_load in loadPattern.distributed_loads[frame]:
                    f_fixed += distributed_load.get_f_fixed()

            f_end_actions = np.array(12*[None])
            f_end_actions[flags_frame_displacements] = np.dot(k_element, u_element).flatten() + np.dot(np.transpose(t), f_fixed).flatten()
            load_pattern_end_actions[frame] = EndActions(self, load_pattern, frame, *f_end_actions)

            # reactions
            if any(joint in self.supports for joint in (_frame.joint_k, _frame.joint_j)):
                rows.extend(list(indexes_element))
                cols.extend(n * [0])
                data.extend(list(np.dot(t, load_pattern_end_actions[frame].get_end_actions()).flatten()))

        self.end_actions[load_pattern] = load_pattern_end_actions

        # store reactions
        number_joints = len(self.joints)
        n = np.count_nonzero(flags_frame_displacements)

        f += loadPattern.get_f_fixed().toarray()
        f_end_actions = coo_matrix((data, (rows, cols)), (number_joints * n, 1)).toarray()
        
        load_pattern_reactions = {}

        for joint in self.supports:
            joint_indexes = indexes[joint]
            _restrains = self.supports[joint].get_restraints()
            restrains = np.array(6 * [False])
            restrains[flags_joint_displacements] = _restrains
            reactions = np.array(6 * [None])
            reactions[flags_joint_displacements & restrains] = f_end_actions[joint_indexes, 0][_restrains] - f[joint_indexes, 0][_restrains]
            load_pattern_reactions[joint] = Reaction(self, load_pattern, joint, *reactions)

        self.reactions[load_pattern] = load_pattern_reactions

        # store internal forces
        load_pattern_internal_forces = {}

        for frame, _frame in self.frames.items():
            load_pattern_internal_forces[frame] = InternalForces(self, load_pattern, frame, **_frame.get_internal_forces(load_pattern))
            
        self.internal_forces[load_pattern] = load_pattern_internal_forces

        # # store internal displacements
        # load_pattern_internal_displacements = {}
        # for frame in self.frames.values():
        #     j = load_pattern_displacements[frame.joint_j]
        #     k = load_pattern_displacements[frame.joint_k]
        #     load_pattern_internal_displacements[frame] = InternalDisplacement(**frame.get_internal_displacement(j, k))

        # self.internal_displacements[load_pattern] = load_pattern_internal_displacements

    def solve(self):
        """Solve the structure"""
        self.set_flags_active_joint_displacements()
        self.set_indexes()
        self.set_stiffness_matrix_modified_by_supports()
        
        for load_pattern in self.load_patterns:
            self.solve_load_pattern(load_pattern)


    # def load(self, data):
    #     """
    #     Create a Structure from dict.
    #     """
    #     # restart
    #     # self.__init__(ux, uy, uz, rx, ry, rz)
        
    #     # materials
    #     for name, material in data['materials'].items():
    #         E = material['E'] if 'E' in material else None
    #         G = material['G'] if 'G' in material else None            
            
    #         self.add_material(name, E, G)
            
    
    # def export(self, filename=None):
    #     """
    #     Save the structure to a file in json format.

    #     Parameters
    #     ----------
    #     filename : string
    #         Filename
    #     """
    #     data = {}

    #     # save the materials
    #     if self.materials:
    #         data['materials'] = {}
    #         for key, material in self.materials.items():
    #             data['materials'][key] = {'E': material.E, 'G': material.G}

    #     # save sections
    #     if self.sections:
    #         data['sections'] = {}
    #         for key, section in self.sections.items():
    #             data['sections'][key] = {'area': section.A, 'Ix': section.Ix, 'Iy': section.Iy, 'Iz': section.Iz, 'type': section.__class__.__name__}
    #             if section.__class__.__name__ == "RectangularSection":
    #                 data['sections'][key]['width'] = section.width
    #                 data['sections'][key]['height'] = section.height

    #     # save the joints
    #     if self.joints:
    #         data['joints'] = {}
    #         for key, joint in self.joints.items():
    #             data['joints'][key] = {'x': joint.x, 'y': joint.y, 'z': joint.z}

    #     # save the frames
    #     material_keys = {value: key for key, value in self.materials.items()}
    #     section_keys = {value: key for key, value in self.sections.items()}
    #     joint_keys = {value: key for key, value in self.joints.items()}
    #     frame_keys = {value: key for key, value in self.frames.items()}

    #     if self.frames:
    #         data['frames'] = {}
    #         for key, frame in self.frames.items():
    #             data['frames'][key] = {'j': joint_keys[frame.joint_j],
    #                                    'k': joint_keys[frame.joint_k],
    #                                    'material': material_keys[frame.material],
    #                                    'section': section_keys[frame.section]}

    #     # save the supports
    #     if self.supports:
    #         data['supports'] = {}
    #         for joint, support in self.supports.items():
    #             data['supports'][joint_keys[joint]] = {'ux': support.ux, 'uy': support.uy, 'uz': support.uz, 'rx': support.rx, 'ry': support.ry, 'rz': support.rz}

    #     # save the loads
    #     if self.load_patterns:
    #         data['load_patterns'] = {}
    #         for key, load_pattern in self.load_patterns.items():
    #             data['load_patterns'][key] = {}

    #             if load_pattern.loads_at_joints:
    #                 data['load_patterns'][key]['joints'] = {}
    #                 for joint, point_load in load_pattern.loads_at_joints.items():
    #                     data['load_patterns'][key]['joints'][joint_keys[joint]] = []
    #                     data['load_patterns'][key]['joints'][joint_keys[joint]].append({  # TODO: manage many 'point load' at the same joint for the same load pattern 
    #                         'fx': point_load.fx, 
    #                         'fy': point_load.fy,
    #                         'fz': point_load.fy,
    #                         'mx': point_load.mx,
    #                         'my': point_load.my,
    #                         'mz': point_load.mz
    #                     })
                
    #             if load_pattern.distributed_loads:
    #                 data['load_patterns'][key]['frames'] = {}
    #                 if load_pattern.distributed_loads:
    #                     for frame, distributed_load in load_pattern.distributed_loads.items():
    #                         if not frame_keys[frame] in data['load_patterns'][key]['frames']:
    #                             data['load_patterns'][key]['frames'][frame_keys[frame]] = {}

    #                         data['load_patterns'][key]['frames'][frame_keys[frame]]['distributed'] = {}

    #                         if distributed_load.system == 'local':
    #                             if not 'local' in data['load_patterns'][key]['frames'][frame_keys[frame]]['distributed']:
    #                                 data['load_patterns'][key]['frames'][frame_keys[frame]]['distributed']['local'] = []
                                
    #                             data['load_patterns'][key]['frames'][frame_keys[frame]]['distributed']['local'].append({
    #                                 'fx': distributed_load.fx,  
    #                                 'fy': distributed_load.fy,
    #                                 'fz': distributed_load.fz
    #                         })

    #     # with open(filename, 'w') as outfile:
    #     #     json.dump(data, outfile, indent=4)

    #     return data

    # def __str__(self):
    #     report = "Flag joint displacements\n" \
    #              "------------------------\n"

    #     report += "ux: {}\nuy: {}\nuz: {}" \
    #               "\nrx: {}\nry: {}\nrz: {}\n".format(*self.get_flags_active_joint_displacements())
    #     report += "\n"

    #     report += "Materials\n" \
    #               "---------\n"
    #     report += '\t'.join(("label", "E", "\t\tG")) + '\n'
    #     for label, material in self.materials.items():
    #         report += "{}\t\t{}\n".format(label,
    #                                       ',\t'.join([str(getattr(material, key)) for key in material.__dict__]))
    #     report += "\n"
    #     materials = {v: k for k, v in self.materials.items()}

    #     report += "Sections\n" \
    #               "--------\n"
    #     # really ugly, see Learning Python 5th ed. by Mark Lutz, pag 1015
    #     report_sections = ''
    #     report_rectangular_sections = ''
    #     for label, section in self.sections.items():
    #         if section.__class__.__name__ == "Section":
    #             report_sections += "{}\t\t{}\n".format(label,
    #                                             ',\t'.join([str(getattr(section, name)) for name in (x for x in dir(section) if not x.startswith('__'))]))
    #         elif section.__class__.__name__ == "RectangularSection":
    #             report_rectangular_sections += "{}\t\t{}\n".format(label,
    #                                             ',\t'.join([str(getattr(section, name)) for name in (x for x in dir(section) if not x.startswith('__'))]))


    #     if report_sections != '':
    #         report += '\t'.join(("label", "A", "\tIx", "Iy", "Iz")) + '\n'
    #         report += report_sections

    #     if report_rectangular_sections != '':
    #         report += '\t'.join(("label", "A", "\tIx", "Iy", "Iz", "height", "width")) + '\n'
    #         report += report_rectangular_sections
        
    #     report += "\n"
    #     sections = {v: k for k, v in self.sections.items()}

    #     report += "Joints\n" \
    #               "------\n"
    #     report += '\t'.join(("label", "x", "y", "z")) + '\n'
    #     for label, joint in self.joints.items():
    #         report += "{}\t\t{}\n".format(label,
    #                                       ',\t'.join([str(getattr(joint, key)) for key in joint.__dict__]))
    #     report += "\n"
    #     joints = {v: k for k, v in self.joints.items()}
    #     frames = {v: k for k, v in self.frames.items()}

    #     report += "Frames\n" \
    #               "------\n"
    #     report += '\t'.join(("label", "Joint j", "Joint k", "material", "section")) + '\n'
    #     for label, frame in self.frames.items():
    #         report += "{}\t\t{}\t\t{}\t\t{}\t\t\t{}\n".format(label, frame.joint_j, frame.joint_k, frame.material, frame.section)
    #     report += "\n"

    #     report += "Supports\n" \
    #               "--------\n"
    #     report += '\t'.join(("label", '\t\t'.join(("ux", "uy", "uz", "rx", "ry", "rz")))) + '\n'
    #     for label, support in self.supports.items():
    #         report += "{}\t\t{}\n".format(label,
    #                                       ',\t'.join([str(getattr(support, key)) for key in support.__dict__]))
    #     report += "\n"
    #     load_patterns = {v: k for k, v in self.load_patterns.items()}

    #     report += "Load patterns\n" \
    #               "-------------\n"
    #     for label, load_pattern in self.load_patterns.items():
    #         report += "{}:\n".format(label)

    #         if load_pattern.loads_at_joints:
    #             row_format = "{:>10}" * 6
    #             report += "label" + row_format.format("fx", "fy", "fz", "mx", "my", "mz") + '\n'

    #             for joint, point_load in load_pattern.loads_at_joints.items():
    #                 report += str(joints[joint]) + '\t' + \
    #                           row_format.format(*[str(getattr(point_load, name)) for name in point_load.__slots__]) + '\n'
            
    #         if load_pattern.distributed_loads:
    #             row_format = "{:>10}" * 4
    #             report += "label" + row_format.format("system", "fx", "fy", "fz") + '\n'

    #             for frame, distributed_load in load_pattern.distributed_loads.items():
    #                 report += str(frame) + ' \t' + \
    #                           row_format.format(*[str(getattr(distributed_load, key)) for key in distributed_load.__dict__])

    #     report += "\n"

    #     report += "Displacements\n" \
    #               "-------------\n"
    #     for load_pattern, displacements in self.displacements.items():
    #         report += "{}:\n".format(load_patterns[load_pattern])
    #         row_format = "{:>11}" * 6
    #         report += "label" + row_format.format("ux", "uy", "uz", "rx", "ry", "rz") + '\n'
    #         for joint, displacement in displacements.items():
    #             report += "{}\t\t{}\n".format(joints[joint],
    #                                           ',\t'.join(["{:+.5e}".format(getattr(displacement, name))
    #                                                       for name in displacement.__slots__]))
    #     report += "\n"

    #     report += "Frame end actions\n" \
    #               "------------------\n"
    #     for load_pattern, frame_end_actions in self.end_actions.items():
    #         report += "{}:\n".format(load_patterns[load_pattern])
    #         row_format = "{:>11}" * 6
    #         report += "label" + row_format.format("fxj", "fyj", "fzj", "rxj", "ryj", "rzj", "fxk", "fyk", "fzk", "rxk", "ryk", "rzk") + '\n'
    #         for frame, end_actions in frame_end_actions.items():
    #             report += "{}\t\t{}\n".format(frames[frame],
    #                                           ',\t'.join(["{:+.5f}".format(getattr(end_actions, name))
    #                                                       for name in end_actions.__slots__]))
    #     report += "\n"

    #     report += "Reactions\n" \
    #               "---------\n"
    #     for load_pattern, reactions in self.reactions.items():
    #         report += "{}:\n".format(load_patterns[load_pattern])
    #         row_format = "{:>11}" * 6
    #         report += "label" + row_format.format("fx", "fy", "fz", "mx", "my", "mz") + '\n'
    #         for joint, reaction in reactions.items():
    #             report += "{}\t\t{}\n".format(joints[joint],
    #                                           ',\t'.join(["{:+.5f}".format(getattr(reaction, name))
    #                                                       for name in reaction.__slots__]))

    #     report += "Internal forces\n" \
    #               "---------------\n"
    #     for load_pattern, internal_forces in self.internal_forces.items():
    #         report += "{}:\n".format(load_patterns[load_pattern])

    #         for frame, internal_force in internal_forces.items():
    #             report += str(frames[frame]) + '\n'

    #             for force in internal_force.__slots__:
    #                 report += force + '\n'
    #                 report += getattr(internal_force, force).__repr__() + '\n'

    #     return report


if __name__ == '__main__':
    pass
    # def example_1():
    #     """Solution to problem 7.1 from 'Microcomputadores en Ingeniería Estructural'"""
    #     # create the model
    #     model = Structure(ux=True, uy=True)

    #     # add materials
    #     model.add_material(key='1', E=2040e4)

    #     # add sections
    #     model.add_section(key='1', A=030e-4)
    #     model.add_section(key='2', A=040e-4)
    #     model.add_section(key='3', A=100e-4)
    #     model.add_section(key='4', A=150e-4)

    #     # add joints
    #     model.add_joint(key=1, x=0, y=0)
    #     model.add_joint(key=2, x=8, y=0)
    #     model.add_joint(key=3, x=4, y=3)
    #     model.add_joint(key=4, x=4, y=0)

    #     # add frames
    #     model.add_frame(key='1-3', joint_j=1, joint_k=3, material='1', section='3')
    #     model.add_frame(key='1-4', joint_j=1, joint_k=4, material='1', section='2')
    #     model.add_frame(key='3-2', joint_j=3, joint_k=2, material='1', section='4')
    #     model.add_frame(key='4-2', joint_j=4, joint_k=2, material='1', section='2')
    #     model.add_frame(key='4-3', joint_j=4, joint_k=3, material='1', section='1')

    #     # add supports
    #     model.add_support(joint=1, ux=True, uy=True)
    #     model.add_support(joint=2, ux=False, uy=True)

    #     # add load patterns
    #     model.add_load_pattern(key="point loads")

    #     # add point loads
    #     model.add_load_at_joint(load_pattern="point loads", joint=3, fx=5 * 0.8, fy=5 * 0.6)
    #     model.add_load_at_joint(load_pattern="point loads", joint=4, fy=-20)

    #     # solve the problem
    #     model.solve()

    #     print(model)
    #     # indexes = model.set_indexes()
    #     # print(np.array_str(model.load_patterns['point loads'].get_f(model.get_flag_active_joint_displacements(), indexes).toarray(), precision=2, suppress_small=True))
    #     # print(np.array_str(model.get_stiffness_matrix_with_support(model.get_stiffness_matrix(indexes).tolil(), indexes).toarray(), precision=2, suppress_small=True))

    #     # export the model
    #     model.export('example_1.json')

    # def example_2():
    #     """Solution to problem 7.2 from 'Microcomputadores en Ingeniería Estructural'"""
    #     # create the model
    #     model = Structure(ux=True, uy=True, uz=True)

    #     # add material
    #     model.add_material("2100 t/cm2", 2100e4)

    #     # add sections
    #     model.add_section("10 cm2", 10e-4)
    #     model.add_section("20 cm2", 20e-4)
    #     model.add_section("40 cm2", 40e-4)
    #     model.add_section("50 cm2", 50e-4)

    #     # add joints
    #     model.add_joint('1', 2.25, 6, 4.8)
    #     model.add_joint('2', 3.75, 6, 2.4)
    #     model.add_joint('3', 5.25, 6, 4.8)
    #     model.add_joint('4', 0.00, 0, 6.0)
    #     model.add_joint('5', 3.75, 0, 0.0)
    #     model.add_joint('6', 7.50, 0, 6.0)

    #     # add frames
    #     model.add_frame('1-2', '1', '2', "2100 t/cm2", '20 cm2')
    #     model.add_frame('1-3', '1', '3', "2100 t/cm2", '20 cm2')
    #     model.add_frame('1-4', '1', '4', "2100 t/cm2", '40 cm2')
    #     model.add_frame('1-6', '1', '6', "2100 t/cm2", '50 cm2')
    #     model.add_frame('2-3', '2', '3', "2100 t/cm2", '20 cm2')
    #     model.add_frame('2-4', '2', '4', "2100 t/cm2", '50 cm2')
    #     model.add_frame('2-5', '2', '5', "2100 t/cm2", '40 cm2')
    #     model.add_frame('3-5', '3', '5', "2100 t/cm2", '50 cm2')
    #     model.add_frame('3-6', '3', '6', "2100 t/cm2", '40 cm2')
    #     model.add_frame('4-5', '4', '5', "2100 t/cm2", '10 cm2')
    #     model.add_frame('4-6', '4', '6', "2100 t/cm2", '10 cm2')
    #     model.add_frame('5-6', '5', '6', "2100 t/cm2", '10 cm2')

    #     # add supports
    #     model.add_support('4', True, True, True)
    #     model.add_support('5', True, True, True)
    #     model.add_support('6', True, True, True)

    #     # add load pattern
    #     model.add_load_pattern("point loads")

    #     # add point loads
    #     model.add_load_at_joint("point loads", '1', 10, 15, -12)
    #     model.add_load_at_joint("point loads", '2',  5, -3, -10)
    #     model.add_load_at_joint("point loads", '3', -4, -2,  -6)

    #     # solve
    #     model.solve()

    #     print(model)

    #     model.export("example_2.json")

    # def example_3():
    #     """"Solution to problem 7.6 from 'Microcomputadores en Ingeniería Estructural'"""
    #     # create the model
    #     model = Structure(ux=True, uy=True, uz=True, rx=True, ry=True, rz=True)

    #     # add material
    #     model.add_material('material1', 220e4, 85e4)

    #     # add rectangular sections
    #     model.add_rectangular_section('section1', 0.4, 0.3)
    #     model.add_rectangular_section('section2', 0.25, 0.4)

    #     # add joints
    #     model.add_joint('1', 0, 3, 3)
    #     model.add_joint('2', 5, 3, 3)
    #     model.add_joint('3', 0, 0, 3)
    #     model.add_joint('4', 0, 3, 0)

    #     # add frames
    #     model.add_frame('1-2', '1', '2', 'material1', 'section1')
    #     model.add_frame('4-1', '4', '1', 'material1', 'section2')
    #     model.add_frame('3-1', '3', '1', 'material1', 'section1')

    #     # add supports
    #     model.add_support('2', *6 * (True,))
    #     model.add_support('3', *6 * (True,))
    #     model.add_support('4', *6 * (True,))

    #     # add load pattern
    #     model.add_load_pattern("distributed loads")

    #     # add distributed loads
    #     model.add_distributed_load("distributed loads", '1-2', 0, -2.4, 0)
    #     model.add_distributed_load("distributed loads", '4-1', 0, -3.5, 0)

    #     # solve
    #     model.solve()

    #     print(model)

    #     model.export("example_3.json")

    # example_1()
    # example_2()
    # example_3()
