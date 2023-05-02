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


if __name__ == "__main__":

    center = np.array([[0,0,0]])
    radius = np.array([0.5])
    point = np.array([[1,0,0]])
    direction = np.array([[-1,.7,0]])

    a, b = line_sphere_intersection(center, radius, point, direction)

    p_a = point + direction*a
    p_b = point + direction*b

    print(p_a)
    print(p_b)
    print(np.sum((p_a-center)**2)**0.5)
    print(np.sum((p_b-center)**2)**0.5)


    center_2 = np.array([[1,0,0]])

    direction_2 = np.array([[0,1,0]])

    radius_2 = np.array([2])

    min_radius_2 = np.array([0])

    point_2 = np.array([[0,0,0]])

    a, b = sphere_line_reduction(center_2, radius_2, min_radius_2, point_2, direction_2)

    radius_a = radius_2-a*(radius_2-min_radius_2)
    radius_b = radius_2-b*(radius_2-min_radius_2)

    p_a = point_2 + direction_2*a
    p_b = point_2 + direction_2*b
    print(p_a)
    print(p_b)
    print(radius_a)
    print(radius_b)
    print(np.sum((p_a-center_2)**2)**0.5)
    print(np.sum((p_b-center_2)**2)**0.5)