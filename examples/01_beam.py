"""
Two span beam
"""

# import path to the library and the library itself
import sys
sys.path.append('../src/')
from pymas import Structure

# create the structure and stablish the degrees of freedom
beam = Structure(uy=True, rz=True)

# add material (E = 200e6 kPa, G = 11.5e6 kPa)
beam.add_material("acero", 200e6, 11.5e6)

# add section (HEB 200)
beam.add_section('sec1', 78.1e-4, 59.28e-8, 5696e-8, 2003e-8)

# add nodes
beam.add_joint('n1', 0, 0, 0)
beam.add_joint('n2', 5, 0, 0)
beam.add_joint('n3', 8, 0, 0)

# add beam elements
beam.add_frame('el1', 'n1', 'n2', 'acero', 'sec1')
beam.add_frame('el2', 'n2', 'n3', 'acero', 'sec1')

# add supports
beam.add_support('n1', uy=True, rz=True)
beam.add_support('n2', uy=True)
beam.add_support('n3', uy=True)

# add external loads (kN, kN/m)
beam.add_load_pattern('ac externas')
beam.add_distributed_load('ac externas', 'el1', fy=-30)

# solve the structure
beam.solve()

# export strutural model and results to JSON file
beam.export('01_beam.json')
