'''
sections

Calculate polygon area properties.

Functions
-------
get_A(points): 
    Calculate area.
get_Qy(points):
    Calculate first moment area around z-axis.
get_Qz(points)
    Calculate first moment area around y-axis.
get_y(points)
    Calculate centroide y.
get_z(points)
    Calculate centroide z.
get_Iyy(points)
    Calculate second moment area around z-axis.
get_Izz(points)
    Calculate second moment area around y-axis.
'''


import numpy as np


def A(points):
    'Calculate area'
    A = 0
    num_points = np.shape(points)[0]

    for i in range(num_points):
        Ai = points[i, 0] * points[(i + 1) % num_points, 1] - points[i, 1] * points[(i + 1) % num_points, 0]
        Ai*=0.5

        A+=Ai

    return A

def Qy(points):
    'Calculate first moment area'
    Qy = 0
    num_points = np.shape(points)[0]
    
    for i in range(num_points):
        Qyi = points[i, 0] * points[(i + 1) % num_points, 1] - points[i, 1] * points[(i + 1) % num_points, 0]
        Qyi*=(points[i, 1] + points[(i + 1) % num_points, 1]) / 6
        
        Qy+=Qyi
        
    return Qy

def Qz(points):
    'Calculate first moment area'
    Qz = 0
    num_points = np.shape(points)[0]
    
    for i in range(num_points):
        Qzi = points[i, 0] * points[(i + 1) % num_points, 1] - points[i, 1] * points[(i + 1) % num_points, 0]  
        Qzi*=(points[i, 0] + points[(i + 1) % num_points, 0]) / 6
        
        Qz+=Qzi
        
    return Qz

def y(points):
    'Calculate centroide'
    return Qz(points) / A(points)

def z(points):
    'Calculate centroide'
    return Qy(points) / A(points)

def Iyy(points):
    'Calculate second moment area'
    Iyy = 0
    num_points = np.shape(points)[0]
    _z = z(points)

    for i in range(num_points):
        Ai = points[i, 0] * points[(i + 1) % num_points, 1] - points[i, 1] * points[(i + 1) % num_points, 0]
        Ai*=0.5
        
        zi = (points[i, 1] + points[(i + 1) % num_points, 1]) / 3

        Iyyi = (Ai / 6) * (points[i, 1] ** 2 + points[i, 1] * points[(i + 1) % num_points, 1] + points[(i + 1) % num_points, 1] ** 2)
        Iyy+=(Iyyi + Ai * _z * (_z - 2 * zi))

    return Iyy

def Izz(points):
    'Calculate second moment area'
    Izz = 0
    num_points = np.shape(points)[0]
    _y = y(points)

    for i in range(num_points):
        Ai = points[i, 0] * points[(i + 1) % num_points, 1] - points[i, 1] * points[(i + 1) % num_points, 0]
        Ai*=0.5

        yi = (points[i, 0] + points[(i + 1) % num_points, 0]) / 3

        Izzi = (Ai / 6) * (points[i, 0] ** 2 + points[i, 0] * points[(i + 1) % num_points, 0] + points[(i + 1) % num_points, 0] ** 2)
        Izz+=(Izzi + Ai * _y * (_y - 2 * yi))

    return Izz
