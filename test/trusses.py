#!/bin/python
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from pyFEM.core import *


def pratt(a_input, h_input):
    structures = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    a_h = []
    label_tension = []
    tension = []
    label_compression = []
    compression = []

    for a in a_input:
        for h in h_input:
            structure = Structure()
            structures.append(structure)

            # add material
            structure.materials.add('mat', 1)

            # add section
            structure.sections.add('sec', 'mat', 1)

            # add nodes
            # lower cord
            for i, label in enumerate(letters):
                structure.nodes.add(label + '0', i * a, 0, 0)
            # top cord
            for i, label in enumerate(letters[1:-1]):
                structure.nodes.add(label + '1', (i + 1) * a, h, 0)

            # add trusses
            # lower cord
            for i, label in enumerate(letters[0:-1]):
                structure.trusses.add(label + '0', letters[i] + '0', letters[i + 1] + '0', 'sec')
            # top cord
            for i, label in enumerate(letters[1:-2]):
                structure.trusses.add(label + '1', letters[i + 1] + '1', letters[i + 2] + '1', 'sec')
            # vertical
            for label in letters[1:-1]:
                structure.trusses.add(label + '2', label + '0', label + '1', 'sec')
            # left to right
            for i in range(1, 3):
                structure.trusses.add(letters[i] + letters[i + 1], letters[i] + '1', letters[i + 1] + '0', 'sec')
            # right to left
            for i in range(3, 5):
                structure.trusses.add(letters[i] + letters[i + 1], letters[i] + '0', letters[i + 1] + '1', 'sec')
            # border
            structure.trusses.add('ab', 'a0', 'b1', 'sec')
            structure.trusses.add('fg', 'f1', 'g0', 'sec')

            # add supports
            for label in [node.label for node in structure.nodes]:
                if label == 'a0':
                    structure.supports.add(label, True, True, True)
                elif label == 'g0':
                    structure.supports.add(label, False, True, True)
                else:
                    structure.supports.add(label, False, False, True)

            # add load pattern
            structure.load_patterns.add("point load")

            # add point load
            structure.load_patterns["point load"].point_loads.add('d1', 0, -1, 0)

            # solve the problem
            structure.solve()

            # save maximum values
            label_max_tension = ''
            max_tension = 0
            for truss in structure.trusses:
                if truss.get_forces("point load") > max_tension:
                    label_max_tension = truss.label
                    max_tension = truss.get_forces("point load")

            label_max_compression = ''
            max_compression = 0
            for truss in structure.trusses:
                if truss.get_forces("point load") < max_compression:
                    label_max_compression = truss.label
                    max_compression = truss.get_forces("point load")

            a_h.append(a / h)
            label_tension.append(label_max_tension)
            tension.append(max_tension)
            label_compression.append(label_max_compression)
            compression.append(max_compression)

    data = np.array([a_h, label_tension, tension, label_compression, compression]).transpose()
    data = data[data[:, 0].argsort()]

    a_h = data[:, 0].astype(float)
    label_tension = data[:, 1]
    tension = data[:, 2].astype(float)
    label_compression = data[:, 3]
    compression = data[:, 4].astype(float)

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].set_title('Pratt', fontsize=16)

    axs[0].plot(a_h, tension, 'b-o')
    axs[0].set_ylabel('Tension')
    axs[0].grid(True)

    axs[1].plot(a_h, compression, 'r-o')
    axs[1].set_xlabel('a/h')
    axs[1].set_ylabel('Compression')
    axs[1].grid(True)

    plt.show()

    print("Pratt")
    print("{}: \t{}({}) \t{}({})".format("a/h", "label tension", "tension", "label compression", "compression"))
    for i in range(len(a_h)):
        print("{:.3f}: \t{}({:.3f}) \t\t\t\t{}({:.3f})".format(a_h[i], label_tension[i], tension[i],
                                                               label_compression[i], compression[i]))


def howe(a_input, h_input):
    structures = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    a_h = []
    label_tension = []
    tension = []
    label_compression = []
    compression = []

    for a in a_input:
        for h in h_input:
            structure = Structure()
            structures.append(structure)

            # add material
            structure.materials.add('mat', 1)

            # add section
            structure.sections.add('sec', 'mat', 1)

            # add nodes
            # lower cord
            for i, label in enumerate(letters):
                structure.nodes.add(label + '0', i * a, 0, 0)
            # top cord
            for i, label in enumerate(letters[1:-1]):
                structure.nodes.add(label + '1', (i + 1) * a, h, 0)

            # add trusses
            # lower cord
            for i, label in enumerate(letters[0:-1]):
                structure.trusses.add(label + '0', letters[i] + '0', letters[i + 1] + '0', 'sec')
            # top cord
            for i, label in enumerate(letters[1:-2]):
                structure.trusses.add(label + '1', letters[i + 1] + '1', letters[i + 2] + '1', 'sec')
            # vertical
            for label in letters[1:-1]:
                structure.trusses.add(label + '2', label + '0', label + '1', 'sec')
            # left to right
            for i in range(1, 3):
                structure.trusses.add(letters[i] + letters[i + 1], letters[i] + '0', letters[i + 1] + '1', 'sec')
            # right to left
            for i in range(3, 5):
                structure.trusses.add(letters[i] + letters[i + 1], letters[i] + '1', letters[i + 1] + '0', 'sec')
            # border
            structure.trusses.add('ab', 'a0', 'b1', 'sec')
            structure.trusses.add('fg', 'f1', 'g0', 'sec')

            # add supports
            for label in [node.label for node in structure.nodes]:
                if label == 'a0':
                    structure.supports.add(label, True, True, True)
                elif label == 'g0':
                    structure.supports.add(label, False, True, True)
                else:
                    structure.supports.add(label, False, False, True)

            # add load pattern
            structure.load_patterns.add("point load")

            # add point load
            structure.load_patterns["point load"].point_loads.add('d1', 0, -1, 0)

            # solve the problem
            structure.solve()

            # save maximum values
            label_max_tension = ''
            max_tension = 0
            for truss in structure.trusses:
                if truss.get_forces("point load") > max_tension:
                    label_max_tension = truss.label
                    max_tension = truss.get_forces("point load")

            label_max_compression = ''
            max_compression = 0
            for truss in structure.trusses:
                if truss.get_forces("point load") < max_compression:
                    label_max_compression = truss.label
                    max_compression = truss.get_forces("point load")

            a_h.append(a / h)
            label_tension.append(label_max_tension)
            tension.append(max_tension)
            label_compression.append(label_max_compression)
            compression.append(max_compression)

    data = np.array([a_h, label_tension, tension, label_compression, compression]).transpose()
    data = data[data[:, 0].argsort()]

    a_h = data[:, 0].astype(float)
    label_tension = data[:, 1]
    tension = data[:, 2].astype(float)
    label_compression = data[:, 3]
    compression = data[:, 4].astype(float)

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].set_title('Howe', fontsize=16)

    axs[0].plot(a_h, tension, 'b-o')
    axs[0].set_ylabel('Tension')
    axs[0].grid(True)

    axs[1].plot(a_h, compression, 'r-o')
    axs[1].set_xlabel('a/h')
    axs[1].set_ylabel('Compression')
    axs[1].grid(True)

    plt.show()

    print("Pratt")
    print("{}: \t{}({}) \t{}({})".format("a/h", "label tension", "tension", "label compression", "compression"))
    for i in range(len(a_h)):
        print("{:.3f}: \t{}({:.3f}) \t\t\t\t{}({:.3f})".format(a_h[i], label_tension[i], tension[i],
                                                               label_compression[i], compression[i]))


def warren(a_input, h_input):
    structures = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    a_h = []
    label_tension = []
    tension = []
    label_compression = []
    compression = []

    for a in a_input:
        for h in h_input:
            structure = Structure()
            structures.append(structure)

            # add material
            structure.materials.add('mat', 1)

            # add section
            structure.sections.add('sec', 'mat', 1)

            # add nodes
            # lower cord
            for i, label in enumerate(letters):
                if i % 2 == 0:
                    structure.nodes.add(label + '0', i * a, 0, 0)
            # top cord
            for i, label in enumerate(letters[1:-1]):
                if i % 2 == 0:
                    structure.nodes.add(label + '1', (i + 1) * a, h, 0)

            print(structure.nodes)

            # add trusses
            # lower cord
            for i, label in enumerate(letters[0:-1]):
                if i % 2 == 0:
                    structure.trusses.add(label + '0', letters[i] + '0', letters[i + 2] + '0', 'sec')
            # top cord
            for i, label in enumerate(letters[1:-2]):
                if i % 2 == 0:
                    structure.trusses.add(label + '1', letters[i + 1] + '1', letters[i + 3] + '1', 'sec')
            # vertical
            # for label in letters[1:-1]:
            #     structure.trusses.add(label + '2', label + '0', label + '1', 'sec')
            # left to right
            for i in range(1, 5, 2):
                structure.trusses.add(letters[i] + letters[i + 1], letters[i] + '1', letters[i + 1] + '0', 'sec')
            # right to left
            for i in range(2, 6, 2):
                structure.trusses.add(letters[i] + letters[i + 1], letters[i] + '0', letters[i + 1] + '1', 'sec')
            # border
            structure.trusses.add('ab', 'a0', 'b1', 'sec')
            structure.trusses.add('fg', 'f1', 'g0', 'sec')

            # add supports
            for label in [node.label for node in structure.nodes]:
                if label == 'a0':
                    structure.supports.add(label, True, True, True)
                elif label == 'g0':
                    structure.supports.add(label, False, True, True)
                else:
                    structure.supports.add(label, False, False, True)

            # add load pattern
            structure.load_patterns.add("point load")

            # add point load
            structure.load_patterns["point load"].point_loads.add('d1', 0, -1, 0)

            # solve the problem
            structure.solve()

            # save maximum values
            label_max_tension = ''
            max_tension = 0
            for truss in structure.trusses:
                if truss.get_forces("point load") > max_tension:
                    label_max_tension = truss.label
                    max_tension = truss.get_forces("point load")

            label_max_compression = ''
            max_compression = 0
            for truss in structure.trusses:
                if truss.get_forces("point load") < max_compression:
                    label_max_compression = truss.label
                    max_compression = truss.get_forces("point load")

            a_h.append(a / h)
            label_tension.append(label_max_tension)
            tension.append(max_tension)
            label_compression.append(label_max_compression)
            compression.append(max_compression)

    data = np.array([a_h, label_tension, tension, label_compression, compression]).transpose()
    data = data[data[:, 0].argsort()]

    a_h = data[:, 0].astype(float)
    label_tension = data[:, 1]
    tension = data[:, 2].astype(float)
    label_compression = data[:, 3]
    compression = data[:, 4].astype(float)

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].set_title('Warren', fontsize=16)

    axs[0].plot(a_h, tension, 'b-o')
    axs[0].set_ylabel('Tension')
    axs[0].grid(True)

    axs[1].plot(a_h, compression, 'r-o')
    axs[1].set_xlabel('a/h')
    axs[1].set_ylabel('Compression')
    axs[1].grid(True)

    plt.show()

    print("Pratt")
    print("{}: \t{}({}) \t{}({})".format("a/h", "label tension", "tension", "label compression", "compression"))
    for i in range(len(a_h)):
        print("{:.3f}: \t{}({:.3f}) \t\t\t\t{}({:.3f})".format(a_h[i], label_tension[i], tension[i],
                                                               label_compression[i], compression[i]))


def k(a_input, h_input):
    structures = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    a_h = []
    label_tension = []
    tension = []
    label_compression = []
    compression = []

    for a in a_input:
        for h in h_input:
            structure = Structure()
            structures.append(structure)

            # add material
            structure.materials.add('mat', 1)

            # add section
            structure.sections.add('sec', 'mat', 1)

            # add nodes
            # lower cord
            for i, label in enumerate(letters):
                structure.nodes.add(label + '0', i * a, 0, 0)
            # top cord
            for i, label in enumerate(letters[1:-1]):
                structure.nodes.add(label + '1', (i + 1) * a, h, 0)

            # add trusses
            # lower cord
            for i, label in enumerate(letters[0:-1]):
                structure.trusses.add(label + '0', letters[i] + '0', letters[i + 1] + '0', 'sec')
            # top cord
            for i, label in enumerate(letters[1:-2]):
                structure.trusses.add(label + '1', letters[i + 1] + '1', letters[i + 2] + '1', 'sec')
            # vertical
            for label in letters[1:-1]:
                structure.trusses.add(label + '2', label + '0', label + '1', 'sec')
            # left to right
            for i in range(1, 3):
                structure.trusses.add(letters[i] + letters[i + 1], letters[i] + '1', letters[i + 1] + '0', 'sec')
            # right to left
            for i in range(3, 5):
                structure.trusses.add(letters[i] + letters[i + 1], letters[i] + '0', letters[i + 1] + '1', 'sec')
            # border
            structure.trusses.add('ab', 'a0', 'b1', 'sec')
            structure.trusses.add('fg', 'f1', 'g0', 'sec')

            # add supports
            for label in [node.label for node in structure.nodes]:
                if label == 'a0':
                    structure.supports.add(label, True, True, True)
                elif label == 'g0':
                    structure.supports.add(label, False, True, True)
                else:
                    structure.supports.add(label, False, False, True)

            # add load pattern
            structure.load_patterns.add("point load")

            # add point load
            structure.load_patterns["point load"].point_loads.add('d1', 0, -1, 0)

            # solve the problem
            structure.solve()

            # save maximum values
            label_max_tension = ''
            max_tension = 0
            for truss in structure.trusses:
                if truss.get_forces("point load") > max_tension:
                    label_max_tension = truss.label
                    max_tension = truss.get_forces("point load")

            label_max_compression = ''
            max_compression = 0
            for truss in structure.trusses:
                if truss.get_forces("point load") < max_compression:
                    label_max_compression = truss.label
                    max_compression = truss.get_forces("point load")

            a_h.append(a / h)
            label_tension.append(label_max_tension)
            tension.append(max_tension)
            label_compression.append(label_max_compression)
            compression.append(max_compression)

    data = np.array([a_h, label_tension, tension, label_compression, compression]).transpose()
    data = data[data[:, 0].argsort()]

    a_h = data[:, 0].astype(float)
    label_tension = data[:, 1]
    tension = data[:, 2].astype(float)
    label_compression = data[:, 3]
    compression = data[:, 4].astype(float)

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].set_title('Pratt', fontsize=16)

    axs[0].plot(a_h, tension, 'b-o')
    axs[0].set_ylabel('Tension')
    axs[0].grid(True)

    axs[1].plot(a_h, compression, 'r-o')
    axs[1].set_xlabel('a/h')
    axs[1].set_ylabel('Compression')
    axs[1].grid(True)

    plt.show()

    print("Pratt")
    print("{}: \t{}({}) \t{}({})".format("a/h", "label tension", "tension", "label compression", "compression"))
    for i in range(len(a_h)):
        print("{:.3f}: \t{}({:.3f}) \t\t\t\t{}({:.3f})".format(a_h[i], label_tension[i], tension[i],
                                                               label_compression[i], compression[i]))


pratt(a_input=np.linspace(1, 1, 1), h_input=np.linspace(0.2, 5, 50))
howe(a_input=np.linspace(1, 1, 1), h_input=np.linspace(0.2, 5, 50))
warren(a_input=np.linspace(1, 1, 1), h_input=np.linspace(0.2, 5, 50))
