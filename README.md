# [pymas](https://github.com/rvcristiand/pymas)
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
	<summary><b>Manual Installation</b></summary>
	<p>You can obtain a copy of pymas from <a href="https://github.com/rvcristiand/pymas">its repository</a> by downloading a ZIP archive, or by cloning it using <a href="https://git-scm.com/">Git</a>:</p>
	<pre><code class="language-bash">git clone https://github.com/rvcristiand/pymas.git</code></pre>
	<p>To install pymas, navigate to the project directory in your terminal and run:</p>
	<pre><code class="language-bash">pip install .</code></pre>
	<p>This will install pymas and any required dependencies.</p>
</details>

## Usage 
You can model and analyze linear elastic framed structures using the [Structure](https://github.com/rvcristiand/pymas/blob/74305d1df22b4b534f352d23f9316267b7b17998/src/pymas/core.py#L8) class.

### Examples

<details markdown="1">
<summary><b>Simple beam</b></summary>
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
model.add_frame('beam', 'a', 'b', 'concrete 28 MPa', '0.5x1.0', bending_z=True)

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

<b>Output</b>

```python
Θa: -4.825e-04 rad
Θb: +4.825e-04 rad
Ra: +60.0 kN
Rb: +60.0 kN
Mmax: 150.0 kN m
νmax: -1.508e-03 m
```
</details>

<details markdown="1">
<summary><b>Space truss</b></summary>
The following code models a space truss subject to points loads at their joints and outputs key results:

```python
from pymas import Structure

"""
Solution to problem 7.2 from 'Microcomputadores en Ingeniería Estructural'
"""
# create the model
model = Structure(type='space_truss')

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
model.add_frame('1-2', '1', '2', "2100 t/cm2", '20 cm2', axial=True)
model.add_frame('1-3', '1', '3', "2100 t/cm2", '20 cm2', axial=True)
model.add_frame('1-4', '1', '4', "2100 t/cm2", '40 cm2', axial=True)
model.add_frame('1-6', '1', '6', "2100 t/cm2", '50 cm2', axial=True)
model.add_frame('2-3', '2', '3', "2100 t/cm2", '20 cm2', axial=True)
model.add_frame('2-4', '2', '4', "2100 t/cm2", '50 cm2', axial=True)
model.add_frame('2-5', '2', '5', "2100 t/cm2", '40 cm2', axial=True)
model.add_frame('3-5', '3', '5', "2100 t/cm2", '50 cm2', axial=True)
model.add_frame('3-6', '3', '6', "2100 t/cm2", '40 cm2', axial=True)
model.add_frame('4-5', '4', '5', "2100 t/cm2", '10 cm2', axial=True)
model.add_frame('4-6', '4', '6', "2100 t/cm2", '10 cm2', axial=True)
model.add_frame('5-6', '5', '6', "2100 t/cm2", '10 cm2', axial=True)

# add supports
model.add_support('4', True, True, True)
model.add_support('5', True, True, True)
model.add_support('6', True, True, True)

# add load pattern
model.add_load_pattern("point loads")

# add point loads
model.add_joint_point_load("point loads", '1', 10, 15, -12)
model.add_joint_point_load("point loads", '2',  5, -3, -10)
model.add_joint_point_load("point loads", '3', -4, -2,  -6)

# analyze the model
model.run_analysis()
model.export('space_truss.json')

for joint in ['1', '2', '3']:
    print(f'joint: {joint}')
    print(f'\tux: {model.displacements["point loads"][joint].ux:.3e} m')
    print(f'\tux: {model.displacements["point loads"][joint].uy:.3e} m')
    print(f'\tux: {model.displacements["point loads"][joint].uz:.3e} m')
```

<b>Output</b>

```python
joint: 1
	ux: 8.048e-04 m
	ux: 3.326e-05 m
	ux: -4.464e-03 m
joint: 2
	ux: 2.226e-03 m
	ux: -7.277e-04 m
	ux: -2.732e-03 m
joint: 3
	ux: 7.512e-04 m
	ux: 3.670e-04 m
	ux: -1.773e-03 m
```
</details>

<details markdown="1">
<summary><b>Space frame</b></summary>
The following code models a space frame subject to distributed loads and outputs key results:

```python
from pymas import Structure

"""
Solution to problem 7.6 from 'Microcomputadores en Ingeniería Estructural'
"""

# structure
model = Structure('space_frame')

# add material
model.add_material('material1', 220e4, 85e4)

# add sections
model.add_section('section1', 0.12, 1.944e-3, 9e-4, 1.6e-3)
model.add_section('section2', 0.10, 1.2734e-3, 1.333e-3, 5.208e-4)

# add joints
model.add_joint('1', 0, 3, 3)
model.add_joint('2', 5, 3, 3)
model.add_joint('3', 0, 0, 3)
model.add_joint('4', 0, 3, 0)

# add frames
deformations = {
    'axial': True,
    'torsional': True,
    'bending_y': True,
    'bending_z': True
}

model.add_frame('1-2', '1', '2', 'material1', 'section1', **deformations)
model.add_frame('4-1', '4', '1', 'material1', 'section2', **deformations)
model.add_frame('3-1', '3', '1', 'material1', 'section1', **deformations)

# add supports
model.add_support('2', *6 * (True,))
model.add_support('3', *6 * (True,))
model.add_support('4', *6 * (True,))

# add load pattern
model.add_load_pattern("distributed loads")

# add distributed loads
model.add_distributed_load('distributed loads', '1-2', fy=-2.4)
model.add_distributed_load('distributed loads', '4-1', fy=-3.5)

# solve
model.run_analysis()

# export
model.export('space_frame.json')

print(model.displacements['distributed loads']['1'])
```

<b>Output</b>

```python
{'load_pattern': 'distributed loads', 'joint': '1', 'ux': 2.687315391431904e-05, 'uy': -0.00011574968996290452, 'uz': -1.000611510850756e-05, 'rx': -0.0005668526305782732, 'ry': 7.904785530558646e-06, 'rz': -0.0006309015221398215}
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
