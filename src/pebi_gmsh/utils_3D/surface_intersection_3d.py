from pebi_gmsh.triangulated_surface import TriangulatedSurface
import numpy as np


def _line_cuts(triangle, lines):
    
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])

    line_dirs = lines-triangle[0]

    opposite_sides = (np.sum(line_dirs*normal,axis=1))*(np.sum(np.roll(line_dirs,-1, axis=0)*normal,axis=1)) <= 0

    a = np.sum(line_dirs*normal, axis=1)
    b = np.sum(np.roll(line_dirs, -1, axis=0) * normal, axis=1)
    t = (a/(a-b)).reshape(-1,1)
    P = np.roll(lines, -1, axis=0)*t + (1-t)*lines

    opposite_sides = a*b <= 0
    tri_dirs = np.roll(triangle, -1, axis=0) - triangle
    
    inner_dirs = P.reshape(3,1,3)-triangle
    inside =  np.all(np.sum(np.cross(tri_dirs, inner_dirs)*normal, axis=2) >= 0, axis=1)
    nan_array = np.empty((3,3))
    nan_array[:,:] = np.nan
    return np.where(np.logical_and(inside, opposite_sides).reshape(1,3), P.T,nan_array, ).T
        

def triangle_intersection(a, b):
    pass

    
    #b_normal = np.cross(b[1] - b[0], b[2] - b[0]) 


if __name__ == "__main__":
    triangle = np.array([
        [0,0,0],
        [1,0,0],
        [0.5,0,1]
    ])
    lines = np.array([
        [0,-1,.5],
        [0.6,1,.5],
        [0,1,.5]
    ])
    hello = _line_cuts(triangle, lines)
    print(hello)