# PyMAS: Python-based Direct Stiffness Method

![GitHub licence](https://img.shields.io/github/license/rvcristiand/pymas)
![GitHub Release](https://img.shields.io/github/v/release/rvcristiand/pymas)
![GitHub contributors](https://img.shields.io/github/contributors-anon/rvcristiand/pymas)
![GitHub top language](https://img.shields.io/github/languages/top/rvcristiand/pymas)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/rvcristiand/pymas)
![GitHub commits since latest release](https://img.shields.io/github/commits-since/rvcristiand/pymas/latest)
![GitHub last commit](https://img.shields.io/github/last-commit/rvcristiand/pymas)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/rvcristiand/pymas/total)
![GitHub forks](https://img.shields.io/github/forks/rvcristiand/pymas)

---

## Project Overview

PyMAS is a versatile [Python](https://www.python.org/) library designed for **structural analysis** using the **[Direct Stiffness Method](https://en.wikipedia.org/wiki/Direct_stiffness_method)**. It offers a clear and intuitive object-oriented interface for defining structural elements, materials, loads, and supports. This makes it an excellent resource for civil engineering students, researchers, and professionals.

### Key Features
* **Intuitive API:** Define structures, materials, and sections with a straightforward and clear syntax.
* **Element Support:** Easily define and work with truss, beam and frame 2D or 3D elements.
* **Extensible Design:** The architecture allows for future expansion to include more complex elements and analysis types.
* **Open-Source:** Freely available for use, modification, and distribution under the MIT License.

---

## Installation

The easiest way to install PyMAS is directly from PyPI using `pip`. It is recommended to use a **virtual environment** for your projects.

```bash
pip install pymas
````

-----

## Quick Start: Usage Example

Here is a practical example demonstrating how to model a beam, apply loads, and perform a structural analysis.

```python
from pymas import Structure

# dimensions of the rectangular cross section
b = 0.5  # cross section width, m
h = 1.0  # cross section heigh, m

# length and stiffness modulus
L = 10                 # beam length, m
E = 4700*28**0.5*1000  # Youn's modulus, kN/m2

# cross-section area and self weight
A = b*h  # cross-sectional area, m2
w = 24*A # self weight per length, kN/m

# create the beam-type model
model = Structure(type='beam')

# add materials
model.add_material('concrete 28 MPa', E)

# add cross sections
model.add_rectangular_section('0.5x1.0', base=b, height=h)

# add start and end beam joints
model.add_joint('a', x=0)
model.add_joint('b', x=L)

# add a frame for the beam
model.add_frame('beam', 'a', 'b', 'concrete 28 MPa', '0.5x1.0')

# add supports
model.add_support('a', r_uy=True)
model.add_support('b', r_uy=True)

# add load patterns
model.add_load_pattern('self weight')

# add distributed loads
model.add_distributed_load('self weight', 'beam', fy=-w)

# analyze the model
model.run_analysis()
model.export('simple_beam.json')

# show results
print(f'Θa: {model.displacements['self weight']['a'].rz:+.3e} rad')
print(f'Θb: {model.displacements['self weight']['b'].rz:+.3e} rad')
print(f'Ra: {model.reactions['self weight']['a'].fy:+.1f} kN')
print(f'Rb: {model.reactions['self weight']['b'].fy:+.1f} kN')
print(f'Mmax: {max(model.internal_forces['self weight']['beam'].mz):.1f} kN m')
print(f'νmax: {min(model.internal_displacements['self weight']['beam'].uy):.3e} m')
```

-----

### Contributing

Contributions are welcome\! If you would like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add new feature'`).
4.  Push your changes to the branch (`git push origin feature/your-feature-name`).
5.  Submit a [Pull Request](https://github.com/rvcristiand/pymas/pulls).

You can find [here](https://www.dataschool.io/how-to-contribute-on-github/) a good gide to this workflow.

-----

### License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/mestradam/pymas/blob/main/LICENSE) file for complete details.

-----

### Contact

For support, questions, or to report bugs, please utilize the [GitHub Issues](https://github.com/rvcristiand/pymas/issues) page.

