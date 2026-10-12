# Engineering Mechanics: Statics, 4th Edition
# Bedford and Fowler
# Problem 6.64
# Unidades: metros y kilonewtons

from pymas import Structure

# Crear el modelo
model = Structure(type='space_truss')

# Material: no se dieron propiedades, se asume un módulo muy grande
# para que los elementos actúen como rígidos (en cerchas pymas solo requiere E)
E = 99999999
model.add_material('Rigid', E)

# Sección común para todos los elementos (A = 100)
model.add_section('TrussSection', 100)

# Nodos
model.add_joint('A', 1.1, -0.4, 0)
model.add_joint('B', 1, 0, 0)
model.add_joint('C', 0, 0, 0.6)
model.add_joint('D', 0, 0, -0.4)
model.add_joint('E', 0, 0.8, 0)

# Elementos (axial=True equivale a las rótulas de momento en Pynite)
members = {
    'AB': ('A', 'B'),
    'AC': ('A', 'C'),
    'AD': ('A', 'D'),
    'BC': ('B', 'C'),
    'BD': ('B', 'D'),
    'BE': ('B', 'E'),
}
for name, (j, k) in members.items():
    model.add_frame(name, j, k, 'Rigid', 'TrussSection', axial=True)

# Apoyos (restringen ux, uy, uz)
model.add_support('C', True, True, True)
model.add_support('D', True, True, True)
model.add_support('E', True, True, True)

# Patrón de carga y cargas nodales
model.add_load_pattern('Case 1')
model.add_joint_point_load('Case 1', 'A', 10, 60, 20)  # fx, fy, fz

# Análisis
model.run_analysis()
model.export('space_truss_nodal_load.json')

# Resultados
def axial(frame):
    # fuerza axial constante a lo largo del elemento
    return model.internal_forces['Case 1'][frame].fx[0]

print(f'Member BC calculated axial force: {axial("BC"):.1f}')
print('Member BC expected axial force: 32.7 Tension')
print(f'Member BD calculated axial force: {axial("BD"):.1f}')
print('Member BD expected axial force: 45.2 Tension')
print(f'Member BE calculated axial force: {axial("BE"):.1f}')
print('Member BE expected axial force: 112.1 Compression')
