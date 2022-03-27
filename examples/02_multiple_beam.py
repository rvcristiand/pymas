"""
Multiple span beam
"""

# import path to the library and the library itself
import sys
sys.path.append('../src/')
from pymas import Structure

# create the structure and stablish the degrees of freedom
beam = Structure(uy=True, rz=True)

# add material (E = 200e6 kPa, G = 11.5e6 kPa)
beam.add_material('mat1', 200e6, 11.5e6)

# add section (HEB 200)
beam.add_section('sec1', 78.1e-4, 59.28e-8, 5696e-8, 2003e-8)

# add nodes, elements, supports, and loads
L = 5
beam.add_load_pattern('muerta')
beam.add_joint('1', 0, 0, 0)
beam.add_support('1', uy=True)
for i in range (2, 10):
    beam.add_joint(str(i), (i-1)*L, 0, 0)
    beam.add_support(str(i), uy=True)
    beam.add_frame(str(i-1), str(i-1), str(i), 'mat1', 'sec1')
    beam.add_distributed_load('muerta', str(i-1), fy=-25)

# solve the structure
beam.solve()

# export strutural model and results to JSON file
beam.export('02_multiple_beam.json')
