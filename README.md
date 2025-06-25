# pymas

Model and analyze framed structures with [Python](https://www.python.org/).

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Background

pymas is a Python package implementing the [direct stiffness method](https://en.wikipedia.org/wiki/Direct_stiffness_method) that helps you model and analyze linear elastic framed structures under static loads.

## Install

```
pip install pymas git+https://github.com/rvcristiand/pymas.git
```

### Manual Installation
You can obtain a copy of pymas from [its reporsitory](https://github.com/rvcristiand/pymas) or you can clone it using [git](https://git-scm.com/).

```
git clone https://github.com/rvcristiand/pymas.git
```

## Usage

You can model and analyze linear elastic framed structures using the [Structure](https://github.com/rvcristiand/pymas/blob/74305d1df22b4b534f352d23f9316267b7b17998/src/pymas/core.py#L8) class.

```python
from pymas import Structure

# model and analyze a simple concrete beam subjected to its self weight

# dimensions of the rectangular cross section
b = 0.5  # width, m
h = 1    # heigh, m

# length and stiffness modulus
L = 10                 # length, m
E = 4700*28**0.5*1000  # stiffness module, kN/m2

# cross-sectional area and self weight
A = b * h  # cross-sectional area, m2
w = 24*A   # self weight per length, kN/m

# create the model
model = Structure(type='beam')

# add materials
model.add_material('concrete 28 MPa', E)

# add sections
model.add_rectangular_section('0.5x1.0', width=b, height=h)

# add joints
model.add_joint('a', x=0)
model.add_joint('b', x=L)

# add frame
model.add_frame('beam', 'a', 'b', 'concrete 28 MPa', '0.5x1.0')

# add supports
model.add_support('a', uy=True)
model.add_support('b', uy=True)

# add load patterns
model.add_load_pattern('self weight')

# add distributed loads
model.add_distributed_load('self weight', 'beam', fy=-w)

# analyze the model
model.run_analysis()
model.export('simple_beam.json')

print(model.reactions['self weight']['a'].fy)  # 60 kN
print(max(model.internal_forces['self weight']['beam'].mz)) # 150 kN m
```

## Contributing
You can contribute to this project creating a new [issue](https://github.com/rvcristiand/pymas/issues/new) or creating [pull requests](https://github.com/rvcristiand/pymas/pulls).

## License
[MIT](LICENSE)
