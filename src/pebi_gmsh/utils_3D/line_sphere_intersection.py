import numpy as np

def line_sphere_intersection(center, radius, point, direction):

    a = np.sum(direction**2, axis=-1)
    b = (np.sum(direction * (point - center),axis=-1))
    c = np.sum((point -  center) ** 2, axis=-1) - radius ** 2

    delta = b**2 - a*c

    return (-b - np.sqrt(delta))/(a), (-b + np.sqrt(delta))/(a) 

def sphere_line_reduction(center, radius, min_radius, point, direction):
    phi = radius - min_radius
    diff = center - point
    a = np.sum(direction**2, axis=-1) - phi**2
    b = (np.sum(diff*direction, axis=-1) + phi*radius)
    c = np.sum(diff**2, axis=-1) - radius**2
    return (-b + np.sqrt(b**2 - a*c))/a, (-b - np.sqrt(b**2 - a*c))/a 

