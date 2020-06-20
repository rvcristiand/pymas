from pyFEM.primitives import *

import sys


import numpy as np
import json


class Structure:
    """Model and analysis a frame structure

    Attributes
    ----------
    ux : bool
        Flag active x-axis displacement.
    uy : bool
        Flag active y-axis displacement.
    uz : bool
        Flag active z-axis displacement.
    rx : bool
        Flag active x-axis rotation.
    ry : bool
        Flag active y-axis rotation.
    rz : bool
        Flag active z-axis rotation.
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
    reactions : dict
        Reactions.

    Methods
    -------
    add_material(key, *args, **kwargs)
        Add a material.
    add_section(key, *args, **kwargs)
        Add a section.
    add_joint(key, *args, **kwargs)
        Add a joint.
    add_frame(key, key_joint_j, key_joint_k, key_material, key_section)
        Add a frame.
    add_support(key_joint, *args, **kwargs)
        Add a support.
    add_load_pattern(key)
        Add a load pattern.
    add_load_at_joint(key_load_pattern, key_joint, *args, **kwargs)
        Add a point load.
    get_flag_active_joint_displacements()
        Get flag active joint displacements.
    get_number_active_joint_displacements()
        Get number active joint displacements.
    get_number_joints()
        Get number joints.
    get_number_frames()
        Get number frames.
    set_indexes()
        Set joint's indexes.
    get_stiffness_matrix()
        Get stiffness matrix.
    get_stiffness_matrix_with_supports(indexes)...
        Get stiffness matrix modified by boundaries...
    solve_load_pattern(load_pattern, indexes, k, k_support)
        Solve frame structure for load pattern...
    set_load_pattern_displacements(load_pattern, indexes, u)
        Set load pattern displacements...
    set_load_pattern_reactions(load_pattern, indexes, f)
        Set load pattern reactions...
    solve()
        Solve structure...
    export()
        Export the model.
    """
    def __init__(self, ux=False, uy=False, uz=False, rx=False, ry=False, rz=False):
        """
        Instantiate a Structure object

        Parameters
        ----------
        ux : bool
            asd...
        uy : bool
            asd...
        uz : bool
            asd...
        rx : bool
            asd...
        ry : bool
            asd...
        rz : bool
            asd...
        """
        # flag active joint displacements
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

        # dict supports
        self.supports = {}

        # dict load patterns
        self.load_patterns = {}

        # dict displacements
        self.displacements = {}

        # dict reactions
        self.reactions = {}

    def add_material(self, key, *args, **kwargs):
        """
        Add a material

        Parameters
        ----------
        key : immutable
            asd
        """
        self.materials[key] = Material(*args, **kwargs)

    def add_section(self, key, *args, **kwargs):
        """
        Add a section

        Parameters
        ----------
        key : immutable
            asd
        """
        self.sections[key] = Section(*args, **kwargs)

    def add_rectangular_section(self, key, *args, **kwargs):
        """
        Add a section

        Parameters
        ----------
        key : inmutable
            asd
        """
        self.sections[key] = RectangularSection(*args, **kwargs)

    def add_joint(self, key, *args, **kwargs):
        """
        Add a joint

        Parameters
        ----------
        key : immutable
            asd
        """
        self.joints[key] = Joint(*args, **kwargs)

    def add_frame(self, key, key_joint_j, key_joint_k, key_material, key_section):
        """
        Add a frame

        Parameters
        ----------
        key : immutable
            asd
        key_joint_j : immutable
            asd
        key_joint_k : immutable
            asd
        key_material : immutable
            asd
        key_section : immutable
            asd
        """
        self.frames[key] = Frame(self.joints[key_joint_j], self.joints[key_joint_k],
                                 self.materials[key_material], self.sections[key_section])

    def add_support(self, key_joint, *args, **kwargs):
        """
        Add a support

        Parameters
        ----------
        key_joint : immutable
            asd
        """
        self.supports[self.joints[key_joint]] = Support(*args, **kwargs)

    def add_load_pattern(self, key):
        """
        Add a load pattern

        Parameters
        ----------
        key : immutable
            asd
        """
        self.load_patterns[key] = LoadPattern()

    def add_load_at_joint(self, key_load_pattern, key_joint, *args, **kwargs):
        """
        Add a point load at joint

        Parameters
        ----------
        key_load_pattern : immutable
            asd
        key_joint : immutable
            asd
        """
        self.load_patterns[key_load_pattern].add_point_load_at_joint(self.joints[key_joint], *args, **kwargs)

    def add_distributed_load(self, key_load_pattern, key_frame, *args, **kwargs):
        """
        Add a distributed load

        Parameters
        ----------
        key_load_pattern : immutable
            asd
        key_frame : immutable
            asd
        """
        self.load_patterns[key_load_pattern].add_distributed_load(self.frames[key_frame], *args, **kwargs)

    def get_flag_active_joint_displacements(self):
        """Get active joint displacements"""
        return np.array([self.ux, self.uy, self.uz, self.rx, self.ry, self.rz])

    def get_number_active_joint_displacements(self):
        """Get number of active joint displacements"""
        return np.count_nonzero(self.get_flag_active_joint_displacements())

    def get_number_joints(self):
        """Get number of joints"""
        return len(self.joints)

    def get_number_frames(self):
        """Get number of frames"""
        return len(self.frames)

    def set_indexes(self):  # Alternative to save dof in joints
        """Set the indexes"""
        n = self.get_number_active_joint_displacements()

        return {joint: np.arange(n * i, n * (i + 1)) for i, joint in enumerate(self.joints.values())}

    def get_stiffness_matrix(self, indexes):
        """
        Get the stiffness matrix of the structure

        Parameters
        ----------
        indexes : dict
            asd
        """
        flag_joint_displacements = self.get_flag_active_joint_displacements()
        number_active_joint_displacements = np.count_nonzero(flag_joint_displacements)

        number_joints = self.get_number_joints()
        number_frames = self.get_number_frames()

        # just for elements with two joints
        n = 2 * number_active_joint_displacements  # change function element type
        n_2 = n ** 2

        rows = np.empty(number_frames * n_2, dtype=int)
        cols = np.empty(number_frames * n_2, dtype=int)
        data = np.empty(number_frames * n_2)

        for i, frame in enumerate(self.frames.values()):
            k_element = frame.get_global_stiffness_matrix(flag_joint_displacements)
            indexes_element = np.concatenate((indexes[frame.joint_j], indexes[frame.joint_k]))
            indexes_element = np.broadcast_to(indexes_element, (n, n))

            rows[i * n_2:(i + 1) * n_2] = indexes_element.flatten('F')
            cols[i * n_2:(i + 1) * n_2] = indexes_element.flatten()
            data[i * n_2:(i + 1) * n_2] = k_element.flatten()

        return coo_matrix((data, (rows, cols)), 2 * (number_active_joint_displacements * number_joints,))

    def get_stiffness_matrix_with_support(self, stiffness_matrix, indexes):
        """
        Get stiffness matrix with support

        Parameters
        ----------
        stiffness_matrix : array
            asd
        indexes : dict
            asd

        Returns
        -------
        stiffness_matrix_with_supports : lil_matrix
            asd
        """
        flag_joint_displacements = self.get_flag_active_joint_displacements()
        n = np.shape(stiffness_matrix)[0]

        for joint, support in self.supports.items():
            joint_indexes = indexes[joint]
            restrains = support.get_restrains(flag_joint_displacements)

            for index in joint_indexes[restrains]:
                stiffness_matrix[index] = np.zeros(n)
                stiffness_matrix[:, index] = np.zeros((n, 1))
                stiffness_matrix[index, index] = 1

        return stiffness_matrix

    def solve_load_pattern(self, load_pattern, indexes, k, k_support):
        """
        Solve load pattern

        Parameters
        ----------
        load_pattern : LoadPattern
            asd
        indexes : dict
            asd
        k : coo_matrix
            asd
        k_support : lil_matrix
            asd

        Returns
        -------
        u : ndarray
            asd
        f : ndarray
            asd
        """
        flag_joint_displacements = self.get_flag_active_joint_displacements()

        f = load_pattern.get_f(flag_joint_displacements, indexes).tolil()

        for joint, support in self.supports.items():
            joint_indexes = indexes[joint]
            restrains = support.get_restrains(flag_joint_displacements)
            for index in joint_indexes[restrains]:
                f[index, 0] = 0

        u = np.linalg.solve(k_support.toarray(), f.toarray())
        f = np.dot(k.toarray(), u) + load_pattern.get_f_fixed(flag_joint_displacements, indexes).toarray()

        return u, f

    def set_load_pattern_displacements(self, load_pattern, indexes, u):
        """
        Set load pattern displacement

        Parameters
        ----------
        load_pattern : LoadPattern
            asd
        indexes : dict
            asd
        u : ndarray
            ads
        """
        flag_joint_displacements = self.get_flag_active_joint_displacements()

        load_pattern_displacements = {}

        for joint in self.joints.values():
            joint_indexes = indexes[joint]
            displacements = flag_joint_displacements.astype(float)
            displacements[flag_joint_displacements] = u[joint_indexes, 0]
            load_pattern_displacements[joint] = Displacement(*displacements)

        self.displacements[load_pattern] = load_pattern_displacements

    def set_load_pattern_reactions(self, load_pattern, indexes, f):
        """
        Set load pattern reactions

        Parameters
        ----------
        load_pattern : LoadPattern
            asd
        indexes : dict
            asd
        f : ndarray
            ads
        """
        flag_joint_displacements = self.get_flag_active_joint_displacements()

        load_pattern_reactions = {}

        for joint, support in self.supports.items():
            joint_indexes = indexes[joint]
            reactions = flag_joint_displacements.astype(float)
            reactions[flag_joint_displacements] = f[joint_indexes, 0]
            load_pattern_reactions[joint] = Reaction(*reactions)

        self.reactions[load_pattern] = load_pattern_reactions

    def solve(self):
        """Solve the structure"""
        indexes = self.set_indexes()

        k = self.get_stiffness_matrix(indexes)
        k_support = self.get_stiffness_matrix_with_support(k.tolil(), indexes)

        for label, load_pattern in self.load_patterns.items():
            u, f = self.solve_load_pattern(load_pattern, indexes, k, k_support)
            self.set_load_pattern_displacements(load_pattern, indexes, u)
            self.set_load_pattern_reactions(load_pattern, indexes, f)
    
    def export(self, filename):
        """
        Save the structure to a file in json format.

        Parameters
        ----------
        filename : string
            Filename
        """
        data = {
            'joints': {}, 
            'materials': {},
            'sections': {},
            'frames': {},
            'supports': {},
            'load_patterns': {}
        }

        # save the materials
        for key, material in self.materials.items():
            data['materials'][key] = {'E': material.E, 'G': material.G}

        # save sections
        for key, section in self.sections.items():
            data['sections'][key] = {'area': section.A, 'Ix': section.Ix, 'Iy': section.Iy, 'Iz': section.Iz, 'type': section.__class__.__name__}
            
            if section.__class__.__name__ == "RectangularSection":
                data['sections'][key]['width'] = section.width
                data['sections'][key]['height'] = section.height

        # save the joints
        for key, joint in self.joints.items():
            data['joints'][key] = {'x': joint.x, 'y': joint.y, 'z': joint.z}

        # save the frames
        joint_key_list = list(self.joints.keys())
        joint_val_list = list(self.joints.values())

        material_key_list = list(self.materials.keys())
        material_val_list = list(self.materials.values())

        section_key_list = list(self.sections.keys())
        section_val_list = list(self.sections.values())

        for key, frame in self.frames.items():
            data['frames'][key] = {'j': joint_key_list[joint_val_list.index(frame.joint_j)],
                                   'k': joint_key_list[joint_val_list.index(frame.joint_k)],
                                   'material': material_key_list[material_val_list.index(frame.material)],
                                   'section': section_key_list[section_val_list.index(frame.section)]}

        # save the supports
        for key, support in self.supports.items():
            data['supports'][joint_key_list[joint_val_list.index(key)]] = {'ux': support.ux, 'uy': support.uy, 'uz': support.uz, 'rx': support.rx, 'ry': support.ry, 'rz': support.rz}

        # save the loads
        for key, load_pattern in self.load_patterns.items():
            if load_pattern.loads_at_joints:
                data['load_patterns'][key] = {'joints': {}}  # , 'frames': {}
                for joint, point_load in load_pattern.loads_at_joints.items():
                    data['load_patterns'][key]['joints'][joint_key_list[joint_val_list.index(joint)]] = []
                    data['load_patterns'][key]['joints'][joint_key_list[joint_val_list.index(joint)]].append({
                        'fx': point_load.fx, 
                        'fy': point_load.fy,
                        'fz': point_load.fy,
                        'mx': point_load.mx,
                        'my': point_load.my,
                        'mz': point_load.mz
                    })

        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def __str__(self):
        report = "Flag joint displacements\n" \
                 "------------------------\n"

        report += "ux: {}\nuy: {}\nuz: {}" \
                  "\nrx: {}\nry: {}\nrz: {}\n".format(*self.get_flag_active_joint_displacements())
        report += "\n"

        report += "Materials\n" \
                  "---------\n"
        report += '\t'.join(("label", "E", "\t\tG")) + '\n'
        for label, material in self.materials.items():
            report += "{}\t\t{}\n".format(label,
                                          ',\t'.join([str(getattr(material, name)) for name in material.__slots__]))
        report += "\n"
        materials = {v: k for k, v in self.materials.items()}

        report += "Sections\n" \
                  "--------\n"
        # really ugly, see Learning Python 5th ed. by Mark Lutz, pag 1015
        report_sections = ''
        report_rectangular_sections = ''
        for label, section in self.sections.items():
            if section.__class__.__name__ == "Section":
                report_sections += "{}\t\t{}\n".format(label,
                                                ',\t'.join([str(getattr(section, name)) for name in (x for x in dir(section) if not x.startswith('__'))]))
            elif section.__class__.__name__ == "RectangularSection":
                report_rectangular_sections += "{}\t\t{}\n".format(label,
                                                ',\t'.join([str(getattr(section, name)) for name in (x for x in dir(section) if not x.startswith('__'))]))


        if report_sections != '':
            report += '\t'.join(("label", "A", "\tIx", "Iy", "Iz")) + '\n'
            report += report_sections

        if report_rectangular_sections != '':
            report += '\t'.join(("label", "A", "\tIx", "Iy", "Iz", "height", "width")) + '\n'
            report += report_rectangular_sections
        
        report += "\n"
        sections = {v: k for k, v in self.sections.items()}

        report += "Joints\n" \
                  "------\n"
        report += '\t'.join(("label", "x", "y", "z")) + '\n'
        for label, joint in self.joints.items():
            report += "{}\t\t{}\n".format(label,
                                          ',\t'.join([str(getattr(joint, name)) for name in joint.__slots__]))
        report += "\n"
        joints = {v: k for k, v in self.joints.items()}

        report += "Frames\n" \
                  "------\n"
        report += '\t'.join(("label", "Joint j", "Joint k", "material", "section")) + '\n'
        for label, frame in self.frames.items():
            report += "{}\t\t{}\t\t{}\t\t{}\t\t\t{}\n".format(label, joints[frame.joint_j],
                                                              joints[frame.joint_k],
                                                              materials[frame.material],
                                                              sections[frame.section])
        report += "\n"

        report += "Supports\n" \
                  "--------\n"
        report += '\t'.join(("label", '\t\t'.join(("ux", "uy", "uz", "rx", "ry", "rz")))) + '\n'
        for label, support in self.supports.items():
            report += "{}\t\t{}\n".format(joints[label],
                                          ',\t'.join([str(getattr(support, name)) for name in support.__slots__]))
        report += "\n"
        load_patterns = {v: k for k, v in self.load_patterns.items()}

        report += "Load patterns\n" \
                  "-------------\n"
        for label, load_pattern in self.load_patterns.items():
            report += "{}:\n".format(label)
            row_format = "{:>10}" * 6
            report += "label" + row_format.format("fx", "fy", "fz", "mx", "my", "mz") + '\n'
            for joint, point_load in load_pattern.loads_at_joints.items():
                report += str(joints[joint]) + '\t' + \
                          row_format.format(*[str(getattr(point_load, name)) for name in point_load.__slots__]) + '\n'
        report += "\n"

        report += "Displacements\n" \
                  "-------------\n"
        for load_pattern, displacements in self.displacements.items():
            report += "{}:\n".format(load_patterns[load_pattern])
            row_format = "{:>11}" * 6
            report += "label" + row_format.format("ux", "uy", "uz", "rx", "ry", "rz") + '\n'
            for joint, displacement in displacements.items():
                report += "{}\t\t{}\n".format(joints[joint],
                                              ',\t'.join(["{:+.5f}".format(getattr(displacement, name))
                                                          for name in displacement.__slots__]))
        report += "\n"

        report += "Reactions\n" \
                  "---------\n"
        for load_pattern, reactions in self.reactions.items():
            report += "{}:\n".format(load_patterns[load_pattern])
            row_format = "{:>11}" * 6
            report += "label" + row_format.format("fx", "fy", "fz", "mx", "my", "mz") + '\n'
            for joint, reaction in reactions.items():
                report += "{}\t\t{}\n".format(joints[joint],
                                              ',\t'.join(["{:+.5f}".format(getattr(reaction, name))
                                                          for name in reaction.__slots__]))

        return report


if __name__ == '__main__':
    def example_1():
        """Solution to problem 7.1 from 'Microcomputadores en Ingeniería Estructural'"""
        # create the model
        model = Structure(ux=True, uy=True)

        # add materials
        model.add_material(key='1', modulus_elasticity=2040e4)

        # add sections
        model.add_section(key='1', area=030e-4)
        model.add_section(key='2', area=040e-4)
        model.add_section(key='3', area=100e-4)
        model.add_section(key='4', area=150e-4)

        # add joints
        model.add_joint(key=1, x=0, y=0)
        model.add_joint(key=2, x=8, y=0)
        model.add_joint(key=3, x=4, y=3)
        model.add_joint(key=4, x=4, y=0)

        # add frames
        model.add_frame(key='1-3', key_joint_j=1, key_joint_k=3, key_material='1', key_section='3')
        model.add_frame(key='1-4', key_joint_j=1, key_joint_k=4, key_material='1', key_section='2')
        model.add_frame(key='3-2', key_joint_j=3, key_joint_k=2, key_material='1', key_section='4')
        model.add_frame(key='4-2', key_joint_j=4, key_joint_k=2, key_material='1', key_section='2')
        model.add_frame(key='4-3', key_joint_j=4, key_joint_k=3, key_material='1', key_section='1')

        # add supports
        model.add_support(key_joint=1, ux=True, uy=True)
        model.add_support(key_joint=2, ux=False, uy=True)

        # add load patterns
        model.add_load_pattern(key="point loads")

        # add point loads
        model.add_load_at_joint(key_load_pattern="point loads", key_joint=3, fx=5 * 0.8, fy=5 * 0.6)
        model.add_load_at_joint(key_load_pattern="point loads", key_joint=4, fy=-20)

        # solve the problem
        model.solve()

        print(model)

        # export the model
        model.export('example_1.json')

    def example_2():
        """Solution to problem 7.2 from 'Microcomputadores en Ingeniería Estructural'"""
        # create the model
        model = Structure(ux=True, uy=True, uz=True)

        # add material
        model.add_material("2100 t/cm2", 2100e4)

        # add sections
        model.add_section("10 cm2", 10e-4)
        model.add_section("20 cm2", 20e-4)
        model.add_section("40 cm2", 40e-4)
        model.add_section("50 cm2", 50e-4)

        # add joints
        model.add_joint('1', 2.25, 6, 4.8)
        model.add_joint('2', 3.75, 6, 2.4)
        model.add_joint('3', 5.25, 6, 4.8)
        model.add_joint('4', 0.00, 0, 6.0)
        model.add_joint('5', 3.75, 0, 0.0)
        model.add_joint('6', 7.50, 0, 6.0)

        # add frames
        model.add_frame('1-2', '1', '2', "2100 t/cm2", '20 cm2')
        model.add_frame('1-3', '1', '3', "2100 t/cm2", '20 cm2')
        model.add_frame('1-4', '1', '4', "2100 t/cm2", '40 cm2')
        model.add_frame('1-6', '1', '6', "2100 t/cm2", '50 cm2')
        model.add_frame('2-3', '2', '3', "2100 t/cm2", '20 cm2')
        model.add_frame('2-4', '2', '4', "2100 t/cm2", '50 cm2')
        model.add_frame('2-5', '2', '5', "2100 t/cm2", '40 cm2')
        model.add_frame('3-5', '3', '5', "2100 t/cm2", '50 cm2')
        model.add_frame('3-6', '3', '6', "2100 t/cm2", '40 cm2')
        model.add_frame('4-5', '4', '5', "2100 t/cm2", '10 cm2')
        model.add_frame('4-6', '4', '6', "2100 t/cm2", '10 cm2')
        model.add_frame('5-6', '5', '6', "2100 t/cm2", '10 cm2')

        # add supports
        model.add_support('4', True, True, True)
        model.add_support('5', True, True, True)
        model.add_support('6', True, True, True)

        # add load pattern
        model.add_load_pattern("point loads")

        # add point loads
        model.add_load_at_joint("point loads", '1', 10, 15, -12)
        model.add_load_at_joint("point loads", '2',  5, -3, -10)
        model.add_load_at_joint("point loads", '3', -4, -2,  -6)

        # solve
        model.solve()

        print(model)

        model.export("example_2.json")

    def example_3():
        """"Solution to problem 7.6 from 'Microcomputadores en Ingeniería Estructural'"""
        # create the model
        model = Structure(ux=True, uy=True, uz=True, rx=True, ry=True, rz=True)

        # add material
        model.add_material('material1', 220e4, 85e4)

        # add rectangular sections
        model.add_rectangular_section('section1', 0.4, 0.3)
        model.add_rectangular_section('section2', 0.25, 0.4)

        # add joints
        model.add_joint('1', 0, 3, 3)
        model.add_joint('2', 5, 3, 3)
        model.add_joint('3', 0, 0, 3)
        model.add_joint('4', 0, 3, 0)

        # add frames
        model.add_frame('1-2', '1', '2', 'material1', 'section1')
        model.add_frame('4-1', '4', '1', 'material1', 'section2')
        model.add_frame('3-1', '3', '1', 'material1', 'section1')

        # add supports
        model.add_support('2', *6 * (True,))
        model.add_support('3', *6 * (True,))
        model.add_support('4', *6 * (True,))

        # add load pattern
        model.add_load_pattern("distributed loads")

        # add distributed loads
        model.add_distributed_load("distributed loads", '1-2', 0, -2.4, 0)
        model.add_distributed_load("distributed loads", '4-1', 0, -3.5, 0)

        # solve
        model.solve()

        print(model)

        model.export("example_3.json")

    example_1()
    example_2()
    example_3()
