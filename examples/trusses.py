import examples.makepath

import matplotlib.pyplot as plt
import numpy as np

from pyFEM import Structure


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
            structure = Structure(ux=True, uy=True)
            structures.append(structure)

            # add material
            structure.add_material('mat', 1)

            # add section
            structure.add_section('sec', 1)

            # add nodes
            # lower cord
            for i, label in enumerate(letters):
                structure.add_joint(label + '0', i * a, 0)
            # top cord
            for i, label in enumerate(letters[1:-1]):
                structure.add_joint(label + '1', (i + 1) * a, h)

            # add trusses
            # lower cord
            for i, label in enumerate(letters[0:-1]):
                structure.add_frame(label + '0', letters[i] + '0', letters[i + 1] + '0', 'mat', 'sec')
            # top cord
            for i, label in enumerate(letters[1:-2]):
                structure.add_frame(label + '1', letters[i + 1] + '1', letters[i + 2] + '1', 'mat', 'sec')
            # vertical
            for label in letters[1:-1]:
                structure.add_frame(label + '2', label + '0', label + '1', 'mat', 'sec')
            # left to right
            for i in range(1, 3):
                structure.add_frame(letters[i] + letters[i + 1], letters[i] + '1', letters[i + 1] + '0', 'mat', 'sec')
            # right to left
            for i in range(3, 5):
                structure.add_frame(letters[i] + letters[i + 1], letters[i] + '0', letters[i + 1] + '1', 'mat', 'sec')
            # border
            structure.add_frame('ab', 'a0', 'b1', 'mat', 'sec')
            structure.add_frame('fg', 'f1', 'g0', 'mat', 'sec')

            # add supports
            for label in structure.joints:
                if label == 'a0':
                    structure.add_support(label, True, True)
                elif label == 'g0':
                    structure.add_support(label, False, True)

            # add load pattern
            structure.add_load_pattern("point load")

            # add point load
            structure.load_patterns["point load"].add_point_load_at_joint('d1', fy= -1)

            # solve the problem
            structure.solve()

            # save maximum values
            label_max_tension = ''
            max_tension = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k > max_tension:
                    label_max_tension = end_actions.frame
                    max_tension = end_actions.fx_k

            label_max_compression = ''
            max_compression = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k < max_compression:
                    label_max_compression = end_actions.frame
                    max_compression = end_actions.fx_k

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

    plt.savefig('pratt.jpg')

    print("Pratt")
    print("{}: \t{}({}) \t{}({})".format("a/h", "label tension", "tension", "label compression", "compression"))
    for i in range(len(a_h)):
        print("{:.3f}: \t{}({:.3f}) \t\t\t\t{}({:.3f})".format(a_h[i], label_tension[i], tension[i],
                                                               label_compression[i], compression[i]))

    return structures


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
            structure = Structure(ux=True, uy=True)
            structures.append(structure)

            # add material
            structure.add_material('mat', 1)

            # add section
            structure.add_section('sec', 1)

            # add nodes
            # lower cord
            for i, label in enumerate(letters):
                structure.add_joint(label + '0', i * a, 0)
            # top cord
            for i, label in enumerate(letters[1:-1]):
                structure.add_joint(label + '1', (i + 1) * a, h)

            # add trusses
            # lower cord
            for i, label in enumerate(letters[0:-1]):
                structure.add_frame(label + '0', letters[i] + '0', letters[i + 1] + '0', 'mat', 'sec')
            # top cord
            for i, label in enumerate(letters[1:-2]):
                structure.add_frame(label + '1', letters[i + 1] + '1', letters[i + 2] + '1', 'mat', 'sec')
            # vertical
            for label in letters[1:-1]:
                structure.add_frame(label + '2', label + '0', label + '1', 'mat', 'sec')
            # left to right
            for i in range(1, 3):
                structure.add_frame(letters[i] + letters[i + 1], letters[i] + '0', letters[i + 1] + '1', 'mat', 'sec')
            # right to left
            for i in range(3, 5):
                structure.add_frame(letters[i] + letters[i + 1], letters[i] + '1', letters[i + 1] + '0', 'mat', 'sec')
            # border
            structure.add_frame('ab', 'a0', 'b1', 'mat', 'sec')
            structure.add_frame('fg', 'f1', 'g0', 'mat', 'sec')

            # add supports
            for label in structure.joints:
                if label == 'a0':
                    structure.add_support(label, True, True)
                elif label == 'g0':
                    structure.add_support(label, False, True)
                    
            # add load pattern
            structure.add_load_pattern("point load")

            # add point load
            structure.load_patterns["point load"].add_point_load_at_joint('d1', fy= -1)

            # solve the problem
            structure.solve()

            # save maximum values
            label_max_tension = ''
            max_tension = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k > max_tension:
                    label_max_tension = end_actions.frame
                    max_tension = end_actions.fx_k

            label_max_compression = ''
            max_compression = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k < max_compression:
                    label_max_compression = end_actions.frame
                    max_compression = end_actions.fx_k

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

    plt.savefig('howe.jpg')

    print("Howe")
    print("{}: \t{}({}) \t{}({})".format("a/h", "label tension", "tension", "label compression", "compression"))
    for i in range(len(a_h)):
        print("{:.3f}: \t{}({:.3f}) \t\t\t\t{}({:.3f})".format(a_h[i], label_tension[i], tension[i],
                                                               label_compression[i], compression[i]))

    return structures


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
            structure = Structure(ux=True, uy=True)
            structures.append(structure)

            # add material
            structure.add_material('mat', 1)

            # add section
            structure.add_section('sec', 1)

            # add nodes
            # lower cord
            for i, label in enumerate(letters):
                if i % 2 == 0:
                    structure.add_joint(label + '0', i * a, 0)
            # top cord
            for i, label in enumerate(letters[1:-1]):
                if i % 2 == 0:
                    structure.add_joint(label + '1', (i + 1) * a, h)

            # add trusses
            # lower cord
            for i, label in enumerate(letters[0:-1]):
                if i % 2 == 0:
                    structure.add_frame(label + '0', letters[i] + '0', letters[i + 2] + '0', 'mat', 'sec')
            # top cord
            for i, label in enumerate(letters[1:-2]):
                if i % 2 == 0:
                    structure.add_frame(label + '1', letters[i + 1] + '1', letters[i + 3] + '1', 'mat', 'sec')
            # vertical
            # for label in letters[1:-1]:
            #     structure.trusses.add(label + '2', label + '0', label + '1', 'sec')
            # left to right
            for i in range(1, 5, 2):
                structure.add_frame(letters[i] + letters[i + 1], letters[i] + '1', letters[i + 1] + '0', 'mat', 'sec')
            # right to left
            for i in range(2, 6, 2):
                structure.add_frame(letters[i] + letters[i + 1], letters[i] + '0', letters[i + 1] + '1', 'mat', 'sec')
            # border
            structure.add_frame('ab', 'a0', 'b1', 'mat', 'sec')
            structure.add_frame('fg', 'f1', 'g0', 'mat', 'sec')

            # add supports
            for label in structure.joints:
                if label == 'a0':
                    structure.add_support(label, True, True)
                elif label == 'g0':
                    structure.add_support(label, False, True)

            # add load pattern
            structure.add_load_pattern("point load")

            # add point load
            structure.add_load_at_joint('point load', 'd1', fy=-1)

            # solve the problem
            structure.solve()

            # save maximum values
            label_max_tension = ''
            max_tension = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k > max_tension:
                    label_max_tension = end_actions.frame
                    max_tension = end_actions.fx_k

            label_max_compression = ''
            max_compression = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k < max_compression:
                    label_max_compression = end_actions.frame
                    max_compression = end_actions.fx_k

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

    plt.savefig('warren.jpg')

    print("Pratt")
    print("{}: \t{}({}) \t{}({})".format("a/h", "label tension", "tension", "label compression", "compression"))
    for i in range(len(a_h)):
        print("{:.3f}: \t{}({:.3f}) \t\t\t\t{}({:.3f})".format(a_h[i], label_tension[i], tension[i],
                                                               label_compression[i], compression[i]))

    return structures


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
            structure = Structure(ux=True, uy=True)
            structures.append(structure)

            # add material
            structure.add_material('mat', 1)

            # add section
            structure.add_section('sec', 1)

            # add nodes
            # lower cord
            for i, label in enumerate(letters):
                structure.add_joint(label + '0', i * a, 0)
            # top cord
            for i, label in enumerate(letters[1:-1]):
                structure.add_joint(label + '1', (i + 1) * a, h)
            # middle
            for i, label in enumerate(letters[1:3]):
                structure.add_joint(label + '2', (i + 1) * a, h / 2)
            for i, label in enumerate(letters[4:6]):
                structure.add_joint(label + '2', (i + 4) * a, h / 2)

            # add trusses
            # lower cord
            for i, label in enumerate(letters[0:-1]):
                structure.add_frame(label + '0', letters[i] + '0', letters[i + 1] + '0', 'mat', 'sec')
            # top cord
            for i, label in enumerate(letters[1:-2]):
                structure.add_frame(label + '1', letters[i + 1] + '1', letters[i + 2] + '1', 'mat', 'sec')
            # vertical
            for i in range(2):
                for label in letters[1:3] + letters[4:6]:
                    structure.add_frame(label + '2' + str(i), label + str(2 * i), label + str(2 - i), 'mat', 'sec')

                structure.add_frame('d2', 'd0', 'd1', 'mat', 'sec')
            # diagonals
            for i in range(2):
                # left
                for j in range(2):
                    structure.add_frame(letters[j + 1] + letters[j + 2] + str(i), letters[j + 1] + '2',
                                        letters[j + 2] + str(i), 'mat', 'sec')
                # right
                for j in range(2):
                    structure.add_frame(letters[j + 3] + letters[j + 4] + str(i), letters[j + 3] + str(i),
                                        letters[j + 4] + '2', 'mat', 'sec')
            # border
            structure.add_frame('ab', 'a0', 'b1', 'mat', 'sec')
            structure.add_frame('fg', 'f1', 'g0', 'mat', 'sec')

            # add supports
            for label in structure.joints:
                if label == 'a0':
                    structure.add_support(label, True, True)
                elif label == 'g0':
                    structure.add_support(label, False, True)

            # add load pattern
            structure.add_load_pattern("point load")

            # Add Point load
            structure.add_load_at_joint('point load', 'd1', 0, -1)

            # solve the problem
            structure.solve()

            # save maximum values
            label_max_tension = ''
            max_tension = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k > max_tension:
                    label_max_tension = end_actions.frame
                    max_tension = end_actions.fx_k

            label_max_compression = ''
            max_compression = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k < max_compression:
                    label_max_compression = end_actions.frame
                    max_compression = end_actions.fx_k

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

    axs[0].set_title('k', fontsize=16)

    axs[0].plot(a_h, tension, 'b-o')
    axs[0].set_ylabel('Tension')
    axs[0].grid(True)

    axs[1].plot(a_h, compression, 'r-o')
    axs[1].set_xlabel('a/h')
    axs[1].set_ylabel('Compression')
    axs[1].grid(True)

    plt.savefig('k.jpg')

    print("k")
    print("{}: \t{}({}) \t{}({})".format("a/h", "label tension", "tension", "label compression", "compression"))
    for i in range(len(a_h)):
        print("{:.3f}: \t{}({:.3f}) \t\t\t\t{}({:.3f})".format(a_h[i], label_tension[i], tension[i],
                                                               label_compression[i], compression[i]))

    return structures


def baltimore(a_input, h_input):
    structures = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
    a_h = []
    label_tension = []
    tension = []
    label_compression = []
    compression = []

    for a in a_input:
        for h in h_input:
            structure = Structure(ux=True, uy=True)
            structures.append(structure)

            # add material
            structure.add_material('mat', 1)

            # add section
            structure.add_section('sec', 1)

            # add nodes
            # lower cord
            for i, label in enumerate(letters):
                structure.add_joint(label + '0', i * a / 2, 0)
            # top cord
            for i, label in enumerate(letters[2:-2:2]):
                structure.add_joint(label + '1', (2 * i + 2) * a / 2, h)
            # middle
            for i, label in enumerate(letters[1::2]):
                structure.add_joint(label + '2', (2 * i + 1) * a / 2, h / 2)

            # add trusses
            # lower cord
            for i, label in enumerate(letters[0:-1]):
                structure.add_frame(label + '0', letters[i] + '0', letters[i + 1] + '0', 'mat', 'sec')
            # top cord
            for i, label in enumerate(letters[2:-4:2]):
                structure.add_frame(label + '1', letters[2 * i + 2] + '1', letters[2 * i + 4] + '1', 'mat', 'sec')
            # vertical
            # longest
            for label in letters[2:-2:2]:
                structure.add_frame(label + '2', label + '0', label + '1', 'mat', 'sec')
            # shortest
            for label in letters[1:-1:2]:
                structure.add_frame(label + '2', label + '0', label + '2', 'mat', 'sec')
            # left to right first level
            for i, label in enumerate(letters[:-1:2]):
                structure.add_frame(letters[2 * i] + letters[2 * i + 1] + '0',
                                    letters[2 * i] + '0', letters[2 * i + 1] + '2', 'mat', 'sec')
            # right to left first level
            for i, label in enumerate(letters[1:-1:2]):
                structure.add_frame(letters[2 * i + 1] + letters[2 * i + 2] + '0',
                                    letters[2 * i + 1] + '2', letters[2 * i + 2] + '0', 'mat', 'sec')
            # left to right second level
            for i, label in enumerate(letters[7:-2:2]):
                structure.add_frame(letters[2 * i + 7] + letters[2 * i + 8] + '1',
                                    letters[2 * i + 7] + '2', letters[2 * i + 8] + '1', 'mat', 'sec')
            # right to left second level
            for i, label in enumerate(letters[2:6:2]):
                structure.add_frame(letters[2 * i + 2] + letters[2 * i + 3] + '1',
                                    letters[2 * i + 2] + '1', letters[2 * i + 3] + '2', 'mat', 'sec')
            # border
            structure.add_frame('bc1', 'b2', 'c1', 'mat', 'sec')
            structure.add_frame('kl1', 'k1', 'l2', 'mat', 'sec')

            # add supports
            for label in structure.joints:
                if label == 'a0':
                    structure.add_support(label, True, True)
                elif label == 'm0':
                    structure.add_support(label, False, True)

            # add load pattern
            structure.add_load_pattern("point load")

            # add point load
            structure.add_load_at_joint('point load', 'g1', fy=-1)

            # solve the problem
            structure.solve()

            # save maximum values
            label_max_tension = ''
            max_tension = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k > max_tension:
                    label_max_tension = end_actions.frame
                    max_tension = end_actions.fx_k

            label_max_compression = ''
            max_compression = 0
            for end_actions in structure.end_actions['point load'].values():
                if end_actions.fx_k < max_compression:
                    label_max_compression = end_actions.frame
                    max_compression = end_actions.fx_k

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

    axs[0].set_title('Baltimore', fontsize=16)

    axs[0].plot(a_h, tension, 'b-o')
    axs[0].set_ylabel('Tension')
    axs[0].grid(True)

    axs[1].plot(a_h, compression, 'r-o')
    axs[1].set_xlabel('a/h')
    axs[1].set_ylabel('Compression')
    axs[1].grid(True)

    plt.savefig('baltimore.jpg')

    print("Baltimore")
    print("{}: \t{}({}) \t{}({})".format("a/h", "label tension", "tension", "label compression", "compression"))
    for i in range(len(a_h)):
        print("{:.3f}: \t{}({:.3f}) \t\t\t\t{}({:.3f})".format(a_h[i], label_tension[i], tension[i],
                                                               label_compression[i], compression[i]))

    # return structures


pratt(a_input=np.linspace(1, 5, 5), h_input=np.linspace(1, 6, 6))
howe(a_input=np.linspace(1, 5, 5), h_input=np.linspace(1, 6, 6))
warren(a_input=np.linspace(1, 5, 5), h_input=np.linspace(1, 6, 6))
k(a_input=np.linspace(1, 5, 5), h_input=np.linspace(1, 6, 6))
baltimore(a_input=np.linspace(1, 5, 5), h_input=np.linspace(1, 6, 6))
