# pymas
![GitHub licence](https://img.shields.io/github/license/rvcristiand/pymas)
![GitHub Release](https://img.shields.io/github/v/release/rvcristiand/pymas) <!-- ![GitHub contributors](https://img.shields.io/github/contributors-anon/rvcristiand/pymas) -->
![GitHub top language](https://img.shields.io/github/languages/top/rvcristiand/pymas) <!-- ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/rvcristiand/pymas) -->
![GitHub commits since latest release](https://img.shields.io/github/commits-since/rvcristiand/pymas/latest)
![GitHub last commit](https://img.shields.io/github/last-commit/rvcristiand/pymas)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/rvcristiand/pymas/total)
[![GitHub stars](https://img.shields.io/github/stars/rvcristiand/pymas)]()
<!-- ![GitHub forks](https://img.shields.io/github/forks/rvcristiand/pymas) -->

Model and analyze framed structures with [Python](https://www.python.org/).

## Table of Contents 
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Background
pymas is a Python package that implements the [direct stiffness method](https://en.wikipedia.org/wiki/Direct_stiffness_method) to help you model and analyze linear elastic framed structures under static loads. 

<!-- It offers a clear and intuitive object-oriented interface for defining structural elements, materials, loads, and supports. -->

<!-- ### Key Features -->
<!-- * **Intuitive API:** Define structures, materials, and sections with a straightforward and clear syntax. -->
<!-- * **Element Support:** Easily define and work with truss, beam and frame 2D or 3D elements. -->
<!-- * **Extensible Design:** The architecture allows for future expansion to include more complex elements and analysis types. -->
<!-- * **Open-Source:** Freely available for use, modification, and distribution under the MIT License. -->

## Install
You can install pymas using [pip](https://pip.pypa.io/en/stable/):

```
pip install git+https://github.com/rvcristiand/pymas.git
```

<details>
	<summary><h3>Manual Installation</h3></summary>

You can obtain a copy of pymas from [its repository](https://github.com/rvcristiand/pymas) by downloading a ZIP archive, or by cloning it using [Git](https://git-scm.com/):
	
```bash
git clone https://github.com/rvcristiand/pymas.git
```

To install pymas, navigate to the project directory in your terminal and run:

```bash
pip install .
```

This will install pymas and any required dependencies.
</details>

## Usage 
You can model and analyze linear elastic framed structures using the [Structure](https://github.com/rvcristiand/pymas/blob/74305d1df22b4b534f352d23f9316267b7b17998/src/pymas/core.py#L8) class.

### Examples
<details>
	<summary><h4>Simple beam</h4></summary>
The following code models a simple concrete beam subjected to its self weight and outputs key results:

```python
from pymas import Structure

# model simple beam

# model and analyze a simple concrete beam subjected to its self weight

# dimensions of the rectangular cross section
b = 0.5  # width, m
h = 1    # heigh, m

# length and stiffness modulus
L = 10                 # length, m
E = 4700*28**0.5*1000  # stiffness module, kN/m2

# cross-sectional area and self weight
A = b*h  # cross-sectional area, m2
w = 24*A   # self weight per length, kN/m

# create the model
model = Structure(type='beam')

# add materials
model.add_material('concrete 28 MPa', E)

# add sections
model.add_rectangular_section('0.5x1.0', base=b, height=h)

# add joints
model.add_joint('a', x=0)
model.add_joint('b', x=L)

# add frame
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

print(f'Θa: {model.displacements['self weight']['a'].rz:+.3e} rad')
print(f'Θb: {model.displacements['self weight']['b'].rz:+.3e} rad')
print(f'Ra: {model.reactions['self weight']['a'].fy:+.1f} kN')
print(f'Rb: {model.reactions['self weight']['b'].fy:+.1f} kN')
print(f'Mmax: {max(model.internal_forces['self weight']['beam'].mz):.1f} kN m')
print(f'νmax: {min(model.internal_displacements['self weight']['beam'].uy):.3e} m')
```

**Output:**

```
Θa: -4.825e-04 rad
Θb: +4.825e-04 rad
Ra: +60.0 kN
Rb: +60.0 kN
Mmax: 150.0 kN m
νmax: -1.508e-03 m
```
</details>

## Contributing

You can contribute to this project creating a new [issue](https://github.com/rvcristiand/pymas/issues/new) or creating [pull requests](https://github.com/rvcristiand/pymas/pulls).

<!-- Contributions are welcome\! If you would like to contribute, please follow these steps: -->

<!-- 1.  Fork the repository. -->
<!-- 2.  Create a new branch (`git checkout -b feature/your-feature-name`). -->
<!-- 3.  Make your changes and commit them (`git commit -m 'Add new feature'`). -->
<!-- 4.  Push your changes to the branch (`git push origin feature/your-feature-name`). -->
<!-- 5.  Submit a [Pull Request](https://github.com/rvcristiand/pymas/pulls). -->

<!-- You can find [here](https://www.dataschool.io/how-to-contribute-on-github/) a good gide to this workflow. -->

## License
[MIT](LICENSE)
