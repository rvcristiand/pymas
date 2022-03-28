"""
Simple frame in 2D
"""

# import path to the library and the library itself
import sys
sys.path.append('../src/')
from pymas import Structure

# create the structure and stablish the degrees of freedom
frame = Structure(ux=True, uy=True, rz=True)

# add material (E = 21 GPa)
E = 21e6
frame.add_material('mat1', E, E/(2*(1+0.15)))

# add sections (sec1 b=0.3 h=0.45, sec2 b=0.3 h=0.3)
frame.add_section('sec1', 0.3, 0.45)
frame.add_section('sec2', 0.3, 0.3)

# add nodes, elements, supports, and loads
L = 6
H = 3
frame.add_load_pattern('muerta')

for i in range (1, 6): # base nodes
    frame.add_joint(i, (i-1)*L, 0, 0)
    frame.add_support(i, uy=True, ux=True, rz=True)

for j in range (1, 4): # higher nodes and columns
    for i in range (1, 6):
        frame.add_joint(j*5+i, (i-1)*L, j*H, 0)
        frame.add_frame((j-1)*5+i, (j-1)*5+i, j*5+i, 'mat1', 'sec2')

for j in range (1, 4): # beams
    for i in range (1, 5):
        frame.add_frame(15+(j-1)*5+i, j*5+i, j*5+1+i, 'mat1', 'sec1')
        frame.add_distributed_load('muerta', 15+(j-1)*5+i, fy=-5)

# solve the structure
frame.solve()

# export strutural model and results to JSON file
frame.export('03_frame.json')
