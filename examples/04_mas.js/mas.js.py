import makepath
from pymas import Structure

# create a model
model = Structure(True, True, True)

# open model
model.open('space_truss.json')

# solve model
model.solve()

# for frame in model.frames.values():
#     print(frame.name)
#     print(frame.get_global_stiffness_matrix())
#     input()

# export model
model.export('space_truss-solved.json')
