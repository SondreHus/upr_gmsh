import numpy as np
from  scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def flatten_sphere_centers(centers: np.ndarray):
    """Converts a set of 3d point triplets into sets where the first point is at origo, the second is at x=0,
    all three are at z=0 and the orientation of the triangle normal is [0,0,1].
    Additionally returns the necesssary rotation matrices and offsets to revert this change.

    Args:
        centers (np.ndarray): _description_
        radii (np.ndarray): _description_
    """
    centers = centers.reshape(-1,3,3)
    # TODO: Fix linear centers
    c_0 = centers[:, 0, :].reshape(-1,1,3).copy()
    flattened_centers = centers - c_0
    

    # Triangle normal
    a = np.cross(flattened_centers[:,1,:], flattened_centers[:,2,:])
    a = a/np.linalg.norm(a, axis=1).reshape(-1,1)

    
    b = np.array([0,0,1])
    v = np.cross(a,b)

    c = np.sum(a*b, axis=1)
    
    s = np.linalg.norm(v, axis=1)
    z = np.zeros(flattened_centers.shape[0])
    
    kmat = np.swapaxes(np.array([
        [z, -v[:, 2], v[:, 1]], 
        [v[:, 2], z, -v[:, 0]], 
        [-v[:, 1], v[:, 0], z]]
    ), 2, 0)
    
    k = np.divide((1 - c), (s ** 2), where = (np.isclose(s, 0)==False) , out = np.ones((s.shape))).reshape(-1,1,1)

    rotation_matrix = (np.eye(3) + kmat + kmat@kmat * k)
    
    # Resolving the cases where the face vector is either [0,0,1] or [0,0,-1]
    flip_matrix = np.tile(np.diag([1,-1,-1]),(s.shape[0], 1, 1))
    eye_matrix = np.tile(np.eye(3),(s.shape[0], 1, 1))
    singelton_matrix = np.where((a[:,2]>0).reshape(-1,1,1), eye_matrix, flip_matrix)    

    rotation_matrix = np.where(np.isclose(s,0).reshape(-1,1,1), singelton_matrix, rotation_matrix)

    kmat_inv = np.swapaxes(np.array([
        [z, v[:, 2], -v[:, 1]], 
        [-v[:, 2], z, v[:, 0]], 
        [v[:, 1], -v[:, 0], z]]
    ), 2, 0)

    rotation_matrix_inv = (np.eye(3) + kmat_inv + kmat_inv@kmat_inv * k)
    rotation_matrix_inv = np.where(np.isclose(s,0).reshape(-1,1,1), singelton_matrix, rotation_matrix_inv)

    flattened_centers =  flattened_centers @ rotation_matrix

    z_angles = -np.arctan2(flattened_centers[:,1,1], flattened_centers[:,1,0])

    z_rotation_matrix = np.swapaxes(np.array([
        [np.cos(z_angles), -np.sin(z_angles), z],
        [np.sin(z_angles), np.cos(z_angles), z],
        [z,z,z+1]
    ]),2,0)

    z_inv = np.swapaxes(np.array([
        [np.cos(-z_angles), -np.sin(-z_angles), z],
        [np.sin(-z_angles), np.cos(-z_angles), z],
        [z,z,z+1]
    ]),2,0)
    
    flattened_centers = flattened_centers @ z_rotation_matrix

    inverse_matrix = z_inv @ rotation_matrix_inv

    y_mirror_matrix = np.swapaxes(np.array([
        [z+1, z, z],
        [z,np.where(flattened_centers[:,2,1] < 0, -1, 1), z],
        [z, z, z+1]
    ]), 2, 0)

    flattened_centers = flattened_centers @ y_mirror_matrix

    inverse_matrix = inverse_matrix @ y_mirror_matrix

    return flattened_centers, c_0, inverse_matrix

# def rotation_matrix_from_vectors(vec1, vec2):
#     """ Find the rotation matrix that aligns vec1 to vec2
#     :param vec1: A 3d "source" vector
#     :param vec2: A 3d "destination" vector
#     :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
#     """
#     a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     s = np.linalg.norm(v)
#     kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

#     rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

#     return rotation_matrix

def sphere_intersections(centers, radii):
    """Returns the intersections on both sides of a set of point triplets and their radii

    Args:
        centers (_type_): _description_
        radii (_type_): _description_

    Returns:
        _type_: _description_
    """
    centers = centers.reshape(-1,3,3)
    radii = radii.reshape(-1,3)
    plane_centers, offset, rotation = flatten_sphere_centers(centers)

    # assert np.all(np.isclose(plane_centers[:,:,2], 0))

    # assert np.all(np.isclose(plane_centers[:,0,:], 0))

    # assert np.all(np.isclose(plane_centers[:,1,1], 0))


    d = plane_centers[:, 1, 0]

    i = plane_centers[:,2,0]
    j = plane_centers[:,2,1]

    x = (radii[:,0]**2 - radii[:,1]**2 + d**2)/(2*d)

    y = ((radii[:,0]**2 - radii[:,2]**2 + i**2 + j**2)/(2*j) - i*x/j)

    z = np.sqrt(radii[:,0]**2 - x**2 - y**2)

    return (np.c_[x,y,z].reshape((-1,1,3))@rotation + offset).reshape(-1,3), (np.c_[x,y,-z].reshape((-1,1,3))@rotation + offset).reshape(-1,3)



def flatten_planar_centers(centers, normal):
    
    origo = centers[0]
    centers = centers - origo
    up = np.array([0,0,1])

    if np.all(np.isclose(normal, -up)):
        matrix = np.diag([1,-1,-1])
        return centers@matrix.T, matrix, origo

    v = np.cross(normal, up)

    c = np.sum(normal*up)
    
    s = np.linalg.norm(v)

    kmat = np.array([
        [0, -v[2], v[1]], 
        [v[2], 0, -v[0]], 
        [-v[1], v[0], 0]]
    )

    k = 1 if np.isclose(s,0) else (1 - c) / (s ** 2)

    rotation_matrix = (np.eye(3) + kmat + kmat@kmat * k)
    rotation_matrix_inv = (np.eye(3) - kmat + kmat@kmat * k)
    return centers @ rotation_matrix.T, rotation_matrix_inv, origo


def plot_sphere(center, radius, ax):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)*radius + center[0]
    y = np.sin(u)*np.sin(v)*radius + center[1]
    z = np.cos(v)*radius + center[2]
    ax.plot_wireframe(x, y, z, color="r")
