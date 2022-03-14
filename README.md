# pymas

Model and analyse framed structures with [Python](https://www.python.org/).

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Background

Implementación del [método directo de rigideces](https://en.wikipedia.org/wiki/Direct_stiffness_method) en [Python](https://www.python.org).

## Install

pymas usa [Python](python.org) y varias librerias científicas (véase [Scipy](https://scipy.org/)). La recomendación es instalar [Ananconda](https://www.anaconda.com/distribution/).

### Manual
Puede obtener una copia de pymas descargándola de la [página del repositorio](https://github.com/rvcristiand/pymas), o puede clonar este repositorio con [git](https://git-scm.com/).

```
git clone https://github.com/rvcristiand/pymas.git
```

Para importar pymas en otros proyectos, incluya la ruta del código fuente de la librería en la lista [`sys.path`](https://docs.python.org/3/tutorial/modules.html#the-module-search-path).

```
sys.path.append('.../pymas/src/')
```

## Usage

Puede analizar estructuras con la clase [Structure](https://github.com/rvcristiand/pymas/blob/00ee3aae72fa10c34b17381532ab87b5d1169c67/src/pymas/core.py#L7).

```python
from pymas import Structure

# model simplest beam

b = 0.5  # m
h = 1  # m

L = 10  # m
E = 4700*28**0.5*1000  # kN /m2

A = b * h  # m2
w = 24*A  # kN/m

# create the model
model = Structure(uy=True, rz=True)

# add materials
model.add_material('concrete', E, E / (2*(1+0.2)))

# add sections
rect_sect = model.add_rectangular_section('V0.5x1.0', width=b, height=h)

# add joints
model.add_joint('a', x=0)
model.add_joint('b', x=L)

# add frame
model.add_frame('1', 'a', 'b', 'concrete', 'V0.5x1.0')

# add supports
model.add_support('a', ux=True)
model.add_support('b', ux=True)

# add load patterns
loadPattern = model.add_load_pattern('self weight')

# add distributed loads
model.add_distributed_load('self weight', '1', fy=-w)

# solve the model
model.solve()
model.export('simplest_beam.json')

print(model.reactions['self weight']['a'].fy)  # 60 kN
print(max(model.internal_forces['self weight']['1'].mz)) # 150 kN m
```

## Contributing
Puede contribuir en este proyecto creando un [issue](https://github.com/rvcristiand/pymas/issues/new) o haciendo [pull requests](https://github.com/rvcristiand/pymas/pulls).

## License
[MIT](LICENSE)
