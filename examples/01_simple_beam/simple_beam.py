import makepath
from pymas import Structure

# model simple beam

b = 0.5  # m
h = 1  # m

L = 10  # m
E = 4700*28**0.5*1000  # kN /m2

A = b * h  # m2
w = 24*A  # kN/m

# create the model
model = Structure(type='beam')

# add materials
model.add_material('concrete', E)

# add sections
rect_sect = model.add_rectangular_section('V0.5x1.0', base=b, height=h)

# add joints
model.add_joint('a', x=0)
model.add_joint('b', x=L)

# add frame
model.add_frame('1', 'a', 'b', 'concrete', 'V0.5x1.0')

# add supports
model.add_support('a', r_uy=True)
model.add_support('b', r_uy=True)

# add load patterns
loadPattern = model.add_load_pattern('self weight')

# add distributed loads
model.add_distributed_load('self weight', '1', fy=-w)

# solve the model
model.run_analysis()
model.export('simplest_beam.json')

print(model.reactions['self weight']['a'].fy)  # 60 kN
print(max(model.internal_forces['self weight']['1'].mz)) # 150 kN m
