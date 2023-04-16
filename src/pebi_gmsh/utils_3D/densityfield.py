import numpy as np
from numba import njit, jit
from numba.experimental import jitclass
from typing import List
from numba import float32
# Here we lay out the "Inscribed circle distances"


def planar_distance(origin_normal, target_point, target_normal, tol = 1e-5):
    
    factor = 1 - np.sum((origin_normal*target_normal), axis=-1)

    # if abs(factor) < tol:
    #     return np.inf, target_normal

    D_t = -np.sum(target_normal*target_point,axis=-1)
    D_t = np.divide(D_t, factor, np.zeros_like(D_t) + np.inf, where = factor > tol)
    target_normal = np.divide(target_normal, factor[:,None], np.zeros_like(target_normal), where = factor[:,None] > tol)
    return D_t, target_normal


def line_distance(p_o, n_o, p_t, n_t, tol = 1e-5):
    a = n_o[:, None, :] - np.sum(n_o[:, None, :]*n_t, axis=-1)[:, :, None] * n_t
    b = (p_o - p_t) - np.sum((p_o - p_t)*n_t, axis=-1)[:, :, None]*n_t
    a_mag = np.sum(a*a,axis=-1)-1
    # if abs(np.dot(a,a)-1) < 1e-5:
    h = np.sum(n_o[:, None, :]*(p_t-p_o), axis=-1)
    d = np.sum(np.cross(n_t, n_o[:,None,:])*(p_o-p_t),axis=-1)
    flat_dist = (d**2/h + h)/2
    dist = (-np.sum(a*b, axis=-1) - np.sqrt(np.sum(a*b, axis=-1)**2 - np.sum(b**2, axis=-1)*a_mag))/a_mag
    return np.where(abs(a_mag) > tol, dist, flat_dist)
    # return "Sqrt(((y-({}))*{} - (z-({}))*{})^2 + ((z-({}))*{} - (x-({}))*{})^2  + ((x-({}))*{} - (y-({}))*{})^2)"\
    #     .format(p[1], dir[2], p[2], dir[1], p[3], dir[0], p[0], dir[3], p[0], dir[1], p[1], dir[0])

# Safe for 2-sidige problemer

def point_distance(p_o, n_o, p_t):
    # assert not np.isclose(2*np.sum((p_o-p_t)*n_o),0)
    return -np.sum((p_o-p_t)**2, axis=-1)/(2*np.sum((p_o-p_t)*n_o, axis=-1))

    #return "0.66 * Abs(((x-({x0}))^2 + (y-({y0}))^2 + (z-({z0}))^2 )/( 2*( (x-({x0}))*{nx} + (y-({y0}))*{ny}) + (z-({z0}))*{nz}))".format(x0 = p[0], y0 = p[1], z0 = p[2], nx = n[0], ny = n[1], nz = n[2])

def line_distance_comp(p, p_t, n_t):
    return np.sqrt(np.sum(((p-p_t) - (np.dot((p-p_t), n_t))*n_t)**2))


def projected_edge_plane(origin_normal, radial_distance_normal, edge_normal):
    
    normal = np.sum(edge_normal*origin_normal, axis=-1).reshape(-1,3,1) * radial_distance_normal[:,None,:] + edge_normal
    return normal

spec = [
    ("origin_normal", float32[:]),
    ("tri_points", float32[:,:]),
    ("t_normal", float32[:]),
    ("t_d", float32),
    ("D_0", float32),
    ("N_0", float32[:]),
    ("D_1", float32),
    ("N_1", float32[:]),
    ("tri_dirs", float32[:,:]),
    ("tri_normals", float32[:,:]),
    ("p_tri_normals_0", float32[:,:]),
    ("p_tri_d_0", float32[:]),
    ("p_tri_normals_1", float32[:,:]),
    ("p_tri_d_1", float32[:]),
    ("edge_d_start", float32[:]),
    ("edge_d_end", float32[:]),
]
# @jitclass(spec)
class InscribedCircleField:

    def __init__(self, origin_normal, tri_points) -> None:
        
        # Normal vector of the plane using the density field
        self.origin_normal = origin_normal

        # N x 3 x 3
        self.tri_points = tri_points

        # Normal vectors of the targe plane for the inscribed circle
        self.t_normals = np.cross(tri_points[:,1] - tri_points[:,0], tri_points[:,2] - tri_points[:,0], axis=1)
        self.t_normals = self.t_normals / np.sqrt(np.sum(self.t_normals ** 2, axis=1))[:, None]
        self.t_ds = -np.sum(self.t_normals * tri_points[:,0], axis=1)

        # Linear inscribed distance formula between the place ends up being a linear formula f(xyz) = N*xyz + D
        # The formula varies based on the sign of the origin, since is assumes the inscribed sphere has its center in the positive origin normal direction
        self.D_0, self.N_0 = planar_distance(origin_normal, tri_points[:,0], self.t_normals)
        self.D_1, self.N_1 = planar_distance(-origin_normal, tri_points[:,0], self.t_normals)
        
        # Vectors along the edge directions of the triangle

        self.tri_dirs = np.roll(tri_points, -1, axis=1) - tri_points
        self.tri_dirs = self.tri_dirs/np.sqrt(np.sum(self.tri_dirs**2, axis=-1))[:, :, np.newaxis]

        # Vectors orthogonal to the target plane and the tri_dirs
        # Used to check whether a point projected onto the target plane is inside the given triangle

        self.tri_normals = np.cross(self.tri_dirs, self.t_normals[:, None, :], axis=-1)
        self.tri_normals = self.tri_normals / np.sqrt(np.sum(self.tri_normals ** 2, axis=-1))[:, :, np.newaxis]
        
        # Should be N x 3 x 3
        self.p_tri_normals_0 = projected_edge_plane(origin_normal, self.N_0, self.tri_normals)
        self.p_tri_d_0 = -np.sum(self.p_tri_normals_0 * self.tri_points, axis=-1)
        
        self.p_tri_normals_1 = projected_edge_plane(-origin_normal, self.N_1, self.tri_normals)
        self.p_tri_d_1 = -np.sum(self.p_tri_normals_1 * self.tri_points, axis=-1)

        self.edge_d_start = -np.sum(self.tri_dirs * tri_points, axis=-1)
        self.edge_d_end = -np.sum(self.tri_dirs * np.roll(tri_points, -1, axis=1), axis=-1)
        print("yo")
# triangle = np.array([[0.2, 0.2, 0.1],[0.7, 0.2, 0.1],[0.2, 0.7, 0.1]])
# normal = np.array([0,0,1])

    def distance(self, xyz):

        side = np.where(np.sum(self.t_normals*xyz, axis=-1) + self.t_ds >= 0 , 1, -1)

        inside_u = np.sum(self.p_tri_normals_0*xyz, axis=-1) < -self.p_tri_d_0

        line_distances_u = side[:, None] * line_distance(xyz, side[:, None]*self.origin_normal, self.tri_points, self.tri_dirs)
        edge_greater_u = np.sum((line_distances_u[:,:,None]*self.origin_normal + xyz)*self.tri_dirs, axis=-1) >= -self.edge_d_start
        edge_less_u =  np.sum((line_distances_u[:,:,None]*self.origin_normal + xyz)*self.tri_dirs, axis=-1) <= -self.edge_d_end

        inside_d = np.sum(self.p_tri_normals_1*xyz, axis=-1) < -self.p_tri_d_1

        line_distances_d = side[:, None] * line_distance(xyz, -side[:, None]*self.origin_normal, self.tri_points, self.tri_dirs)
        edge_greater_d = np.sum((line_distances_d[:,:,None]*-self.origin_normal + xyz)*self.tri_dirs, axis=-1) >= -self.edge_d_start
        edge_less_d =  np.sum((line_distances_d[:,:,None]*-self.origin_normal + xyz)*self.tri_dirs, axis=-1) <= -self.edge_d_end

        on_plane_u = np.all(inside_u, axis=-1)
        on_edge_u = np.any((inside_u==False) & edge_greater_u & edge_less_u, axis=-1)
        edge_u_idx = np.argmax((inside_u==False) & edge_greater_u & edge_less_u, axis=1)
        row_idx = np.arange(self.tri_points.shape[0])
        plane_u = abs(np.sum(self.N_0*xyz, axis=-1) + self.D_0)
        edge_u = abs(line_distance(xyz, self.origin_normal*side[:, None], self.tri_points[row_idx,None,edge_u_idx], self.tri_dirs[row_idx,None,edge_u_idx])).flatten()
        points = np.min(abs(point_distance(xyz, self.origin_normal, self.tri_points)), axis=-1)
        u = np.where(on_plane_u, plane_u, np.where(on_edge_u, edge_u, points))
        
        on_plane_d = np.all(inside_d, axis=1)
        on_edge_d = np.any((inside_d==False) & edge_greater_d & edge_less_d, axis=1)
        edge_d_idx = np.argmax((inside_d==False) & edge_greater_d & edge_less_d, axis=1)
        row_idx = np.arange(self.tri_points.shape[0])
        plane_d = abs(np.sum(self.N_1*xyz, axis=-1) + self.D_1)
        edge_d = abs(line_distance(xyz, -self.origin_normal*side[:, None], self.tri_points[row_idx,None,edge_d_idx], self.tri_dirs[row_idx,None,edge_d_idx])).flatten()
        points = np.min(abs(point_distance(xyz, -self.origin_normal, self.tri_points)), axis=-1)
        d = np.where(on_plane_d, plane_d, np.where(on_edge_d, edge_d, points))

        return min(u)# min(min(u), min(d))
        # print("yo")
        # if np.all(inside_u, axis=0):
        #     u = abs(np.dot(self.N_0, xyz) + self.D_0)
        # elif np.any((inside_u==False) & edge_greater_u & edge_less_u):
        #     i = np.where((inside_u==False) & edge_greater_u & edge_less_u)[0][0]
        #     u = abs(line_distance(xyz, self.origin_normal*side, self.tri_points[i], self.tri_dirs[i]))
        # else:
        #     i = (np.where(edge_less_u & np.roll(edge_greater_u, 1, axis=0))[0][0])
        #     u = abs(point_distance(xyz, self.origin_normal*side, self.tri_points[i]))

        # if np.all(inside_d):
        #     d = abs(np.dot(self.N_1, xyz) + self.D_1)
        # elif np.any((inside_d==False) & edge_greater_d & edge_less_d):
        #     i = np.where((inside_d==False) & edge_greater_d & edge_less_d)[0][0]
        #     d = abs(line_distance(xyz, -self.origin_normal*side, self.tri_points[i], self.tri_dirs[i]))
        # else:
        #     i = (np.where(edge_less_d & np.roll(edge_greater_d, 1, axis=0))[0][0])
        #     d = abs(point_distance(xyz, -self.origin_normal*side, self.tri_points[i]))
        
        # return min(u,d)
    
# @jitclass()
# class InscribedCircleField:
    # def __init__(self, origin_normal:np.ndarray, tris: np.ndarray) -> None:
    #     self.inscribed_triangles: List = []
    #     for i in range(tris.shape[0]):
    #         self.inscribed_triangles.append(
    #             InscribedCircleTriangle(origin_normal, tris[i])
    #         )
# @njit
# def distance(xyz, inscribed_triangles):
#     min_dist = np.inf
#     for i in range(len(inscribed_triangles)):
#         min_dist = min(min_dist, inscribed_triangles[i].distance(xyz))
#     return min_dist

if __name__ == "__main__":

    #point test
    for i in range(100):

        point = np.random.rand(3)
        normal = np.random.rand(3)
        normal = normal/np.sqrt(np.sum(normal**2))

        target = np.random.rand(3)

        radius = point_distance(point, normal, target)
        print("radius: {}, dist: {}".format(radius, np.sqrt(np.sum((point+normal*radius-target)**2))))
        assert np.isclose(np.sqrt(np.sum((normal*radius)**2)), np.sqrt(np.sum((point+normal*radius-target)**2))), "radius: {}, dist: {}".format(radius, np.sqrt(np.sum((point+normal*radius-target)**2)))

    # line test
    for i in range(100):
        # print(i)
        point_o = np.random.rand(3)
        point_t = np.random.rand(3)

        normal_o = np.random.rand(3)
        normal_o = normal_o/np.sqrt(np.sum(normal_o**2))

        normal_t = np.random.rand(3)
        normal_t = normal_t/np.sqrt(np.sum(normal_t**2))

        radius = line_distance(point_o, normal_o, point_t, normal_t)
        line_dist = line_distance_comp(point_o + radius*normal_o, point_t, normal_t)
        print("radius: {}, dist: {}".format(radius, line_dist))
        assert np.isclose(radius, line_dist) and radius >= 0



