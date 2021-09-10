# pyFEM

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

pyFEM usa [Python](python.org) y varias librerias científicas (véase [Scipy](https://scipy.org/)). La recomendación es instalar [Ananconda](https://www.anaconda.com/distribution/).

Puede instalar esta librería con [pip](https://pip.pypa.io/en/stable/).
```
python -m pip install pyFEM
```

### Manual
Puede obtener una copia de pyFEM descargándola de la [página del repositorio](https://github.com/rvcristiand/pyFEM), o puede clonar este repositorio con [git](https://git-scm.com/).

```
git clone https://github.com/rvcristiand/pyFEM.git
```

Para importar pyFEM en otros proyectos, incluya la ruta del código fuente de la librería en la lista [`sys.path`](https://docs.python.org/3/tutorial/modules.html#the-module-search-path).

```
sys.path.append('.../pyFEM/src/')
```

## Usage

Puede analizar estructuras con la clase [Structure](https://github.com/rvcristiand/pyFEM/blob/b394a88a30b09d5cd7351a5ac35b69fa1c419b93/src/pyFEM/core.py#L7).

```python
import makepath
from pyFEM import Structure

# model simplest beam
model = Structure(uy=True, rz=True)

model.add_material('concrete', E=4700*28**0.5*1000)
rect_sect = model.add_rectangular_section('V0.5x1.0', width=0.5, height=1)

model.add_joint('a', x=0)
model.add_joint('b', x=10)
model.add_frame('1', 'a', 'b', 'concrete', 'V0.5x1.0')

model.add_support('a', uy=True)
model.add_support('b', uy=True)

model.add_load_pattern('self weight')
model.add_distributed_load('self weight', '1', fy=-24*rect_sect.A)

model.solve()

print(model.reactions['self weight']['a'].fy)  # 60 kN
print(max(model.internal_forces['self weight']['1'].mz)) # 150 kN m
```



## Contributing
Puede contribuir en este proyecto creando un [issue](https://github.com/rvcristiand/pyFEM/issues/new) o haciendo [pull requests](https://github.com/rvcristiand/pyFEM/pulls).

## License
[MIT](LICENSE)
