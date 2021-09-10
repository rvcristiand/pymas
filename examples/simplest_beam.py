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