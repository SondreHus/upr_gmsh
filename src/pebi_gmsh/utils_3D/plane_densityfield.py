import numpy as np
from numba import njit, jit
from numba.experimental import jitclass
from typing import List
from numba import float32
import os
import gmsh

def line_plane_distance(point, line_dir, plane_normal, plane_d):

    # TODO: Make sure point is on right side of plane
    inscribed_circle_dir = np.cross(line_dir, np.cross(line_dir, plane_normal))
    inscribed_circle_dir = inscribed_circle_dir/np.sqrt(np.sum(inscribed_circle_dir**2, axis=1)).reshape(-1,1)

    return (np.sum(point * plane_normal, axis=1) + plane_d)/(1-np.sum(inscribed_circle_dir*plane_normal, axis=1)), inscribed_circle_dir



def line_pair_center(p0, v0, p1, v1):

    
    v0 = np.repeat(v0.reshape(1,3), v1.shape[0], axis=0)
    p0 = np.repeat(p0.reshape(1,3), p1.shape[0], axis=0)
    aligned = np.logical_or(np.all(np.isclose(v0,v1), axis=1), np.all(np.isclose(v0,-v1), axis=1))
    centers = np.zeros(p1.shape)
    v2 = np.cross(v0, v1)
    v2 = np.where(aligned.reshape(-1,1), (p1-p0 - v0*np.sum((p1-p0)*v0, axis=1).reshape(-1,1)), v2)
    a = np.stack((v0,-v1,v2), axis=2)#np.vstack((v0,-v1,v2)).T
    m =  np.linalg.solve(a[~aligned], (p1-p0)[~aligned])
    centers[~aligned] = (p0[~aligned] + v0[~aligned]*m[:,0].reshape(-1,1) + p1[~aligned] + v1[~aligned]*m[:,1].reshape(-1,1))/2
    centers[aligned] = p0[aligned] + v1[aligned]/2
    return centers#(p0 + v0*t0 + p1 + v1*t1)/2


# def line_dir_intersection(origin_point, )
# @jit(cache=True)
def line_line_distance(origin_point: np.ndarray, origin_line_dir: np.ndarray, target_points: np.ndarray, target_line_dirs: np.ndarray, tol = 1e-4):
    
    target_points = target_points.reshape(-1,3)
    target_line_dirs = target_line_dirs.reshape(-1,3)
    
    dists = np.zeros(target_points.shape[0])
    dirs = np.zeros(target_line_dirs.shape)
    
    for i in range(target_points.shape[0]):

        target_point = target_points[i]
        target_line_dir = target_line_dirs[i]

        orthogonal_0 = np.cross(origin_line_dir, target_line_dir)

        # Lines are paralell
        if np.sum(orthogonal_0**2) < tol:
            direction = (target_point-origin_point) - origin_line_dir * (np.sum((target_point-origin_point)*origin_line_dir, axis=-1))
            dists[i] = np.linalg.norm(direction)
            dirs[i] = direction
        
        else:

            orthogonal_0 = orthogonal_0/np.linalg.norm(orthogonal_0)
            orthogonal_1 = np.cross(origin_line_dir, orthogonal_0)
            

            test_samples = np.linspace(0,2*np.pi, 36)

            test_dirs = np.sin(test_samples).reshape(-1,1)*orthogonal_0 + np.cos(test_samples).reshape(-1,1)*orthogonal_1

            sample_dists = line_distance(origin_point, test_dirs, target_point, target_line_dir)

            smallest = np.argmin(np.abs(sample_dists))

            dists[i] = np.abs(sample_dists[smallest])
            dirs[i] = test_dirs[smallest]*np.sign(sample_dists[smallest])

    return dists, dirs
    # Smallest inscribes sphere is always on the more acute angle
    # sign = np.where(np.sum(origin_line_dir.reshape(1,3)* target_line_dir, axis=-1) >= 0, 1, -1)

    # center = line_pair_center(origin_point, origin_line_dir, target_point, target_line_dir)
    
    # # Normal of the plane spanned by the closest point between the two lines, and the two intersecting points
    # plane_normal = np.cross(sign.reshape(-1,1)*target_line_dir + origin_line_dir, origin_point-center)
    # plane_normal = plane_normal/np.sqrt(np.sum(plane_normal**2, axis=-1)).reshape(-1,1)
    

    # sphere_dir = np.cross(origin_line_dir, plane_normal)
    # sphere_dir = sphere_dir/np.sqrt(np.sum(sphere_dir**2, axis=-1)).reshape(-1,1)
    
    # return line_distance(origin_point, sphere_dir.reshape(-1,3), target_point.reshape(-1,3), sign.reshape(-1,1)*target_line_dir.reshape(-1,3)), sphere_dir



def line_point_distance(origin_point, origin_line_dir, target_point):
    
    d = np.sum(origin_line_dir.reshape(-1,3)*(target_point-origin_point), axis=-1)
    h = np.sqrt(np.sum((target_point-origin_point - np.sum((target_point-origin_point)*origin_line_dir,axis=-1).reshape(-1,1)*origin_line_dir)**2, axis=-1))
    return (d**2 + h**2)/(2*h)

class LineInscribedField:
    
    def __init__(self, line_dir, triangle_coords, edge_coords, point_coords):
        

        self.line_dir = line_dir
        self.triangle_coords = triangle_coords
        self.edge_coords = edge_coords
        self.point_coords = point_coords

        self.edge_dirs = edge_coords[:,1] - edge_coords[:,0] 
        self.edge_dirs = self.edge_dirs/np.sqrt(np.sum(self.edge_dirs**2, axis=-1)).reshape(-1,1)

        self.edge_starts = np.sum(self.edge_dirs * edge_coords[:,0,:], axis=-1)
        self.edge_ends = np.sum(self.edge_dirs * edge_coords[:,1,:], axis=-1)

        self.plane_normals = np.cross(triangle_coords[:, 1, :] - triangle_coords[:, 0, :], triangle_coords[:, 2, :] - triangle_coords[:, 0, :])
        self.plane_normals = self.plane_normals/np.sqrt(np.sum(self.plane_normals**2, axis=-1)).reshape(-1,1)
        self.plane_ds = -np.sum(self.plane_normals*triangle_coords[:,0,:], axis=-1)
        
        tri_edge_dirs = np.roll(self.triangle_coords, -1, axis=1) - self.triangle_coords
        self.plane_boundaries = np.cross(np.repeat(self.plane_normals, 3, axis=0), tri_edge_dirs.reshape(-1,3)).reshape(-1,3,3)
        self.plane_boundary_ds = -np.sum(self.plane_boundaries*triangle_coords[:,:,:], axis=2)
        
        
    def distance(self, xyz):
        
        plane_dists, plane_dirs = line_plane_distance(xyz, self.line_dir, self.plane_normals, self.plane_ds)

        # NEED TO CHECK THIS
        inside = np.all(np.sum(np.repeat((xyz + plane_dirs*plane_dists.reshape(-1,1)), 3, axis=0).reshape(-1,3,3)*self.plane_boundaries, axis=-1) >= -self.plane_boundary_ds, axis=1)

        min_plane_dist = np.min(np.where(inside, np.abs(plane_dists), np.inf))

        line_dists, line_dirs = line_line_distance(xyz, self.line_dir, self.edge_coords[:,0,:].reshape(-1,3), self.edge_dirs)

        line_vals = np.sum((xyz.reshape(-1,3) + line_dirs * line_dists.reshape(-1,1)).reshape(-1,3) * self.edge_dirs, axis=-1)

        on_line = np.logical_and(line_vals >= self.edge_starts, line_vals <= self.edge_ends).flatten()

        min_line_dist = np.min(np.where(on_line, line_dists, np.inf))

        point_dists = line_point_distance(xyz, self.line_dir.reshape(-1,3), self.point_coords.reshape(-1,3))

        min_point_dist = np.min(point_dists)
        # print("\nplane : {}\nedge: {}\npoint: {}".format(min_plane_dist, min_line_dist, min_point_dist))
        return np.min((min_plane_dist, min_line_dist, min_point_dist))

        

@jit(cache=True)
def plane_plane_distance(origin_normal: np.ndarray, target_point: np.ndarray, target_normal: np.ndarray, tol = 1e-5):
    
    factor = 1 - np.sum((origin_normal*target_normal), axis=-1)


    D_t = -np.sum(target_normal * target_point, axis=-1)
    D_t = np.divide(D_t, factor)#, np.zeros(D_t.shape) + np.inf, where = factor > tol)
    D_t = np.where(factor > tol, D_t, np.zeros(D_t.shape) + np.inf)
    target_normal = np.divide(target_normal, np.expand_dims(factor,1), np.zeros(target_normal.shape))#, where = np.expand_dims(factor,1) > tol)
    target_normal = np.where(np.repeat(factor > tol, 3).reshape(-1,3), target_normal, np.zeros(target_normal.shape))
    return D_t, target_normal

@jit(cache=True)
def line_distance(p_o: np.ndarray, n_o: np.ndarray, p_t: np.ndarray, n_t: np.ndarray, tol = 1e-5):
    
    # o_size = n_o.reshape(-1,3).shape[0]
    # t_size = n_t.reshape(-1,3).shape[0]
    n_o = n_o.reshape(-1,3)#np.repeat(n_o.reshape(-1,3), t_size, axis=0)
    n_t = n_t.reshape(-1,3)#np.repeat(n_t.reshape(-1,3), o_size, axis=0)
    # p_o = p_o.reshape(-1,3)
    p_t = p_t.reshape(-1,3)#np.repeat(p_t.reshape(-1,3), o_size, axis=0)

    a = n_o - np.sum(n_o*n_t, axis=-1).reshape(-1,1) * n_t
    b = p_o - p_t - np.sum((p_o - p_t)*n_t, axis=-1).reshape(-1,1) * n_t
    a_mag = np.sum(a**2,axis=-1)-1
    # if abs(np.dot(a,a)-1) < 1e-5:
    h = np.sum(n_o*(p_t-p_o ), axis=-1)
    d = np.sum(np.cross(n_t, n_o)*(p_o - p_t),axis=-1)
    flat_dist = (d**2/h + h)/2
    
    dist = np.divide(-np.sum(a*b, axis=-1) - np.sqrt(np.sum(a*b, axis=-1)**2 - np.sum(b**2, axis=-1)*a_mag), a_mag)#, where=np.abs(a_mag) > tol, out = flat_dist)
    return np.where(np.abs(a_mag) > tol, dist, flat_dist)

# Safe for 2-sidige problemer
@jit(cache=True)
def point_distance(p_o: np.ndarray, n_o: np.ndarray, p_t: np.ndarray):
    # assert not np.isclose(2*np.sum((p_o-p_t)*n_o),0)
    return -np.sum((p_o-p_t)**2, axis=-1)/(2*np.sum((p_o-p_t)*n_o, axis=-1))


def line_distance_comp(p, p_t, n_t):
    return np.sqrt(np.sum(((p-p_t) - (np.dot((p-p_t), n_t))*n_t)**2))

@jit(cache=True)
def projected_edge_plane(origin_normal, radial_distance_normal, edge_normal):
    normal = np.sum(edge_normal*origin_normal, axis=-1).reshape(-1,3,1) * np.expand_dims(radial_distance_normal,1) + edge_normal
    return normal

spec = [
    ("origin_normal", float32[:]),
    ("tri_points", float32[:,:]),
    ("t_normal", float32[:]),
    ("D_0", float32[:]),
    ("N_0", float32[:,:]),
    ("D_1", float32[:]),
    ("N_1", float32[:,:]),
    ("tri_dirs", float32[:,:]),
    ("tri_normals", float32[:,:]),
    ("t_ds", float32[:]),
    ("t_normals", float32[:,:]),
    ("p_tri_normals_0", float32[:,:]),
    ("p_tri_d_0", float32[:]),
    ("p_tri_normals_1", float32[:,:]),
    ("p_tri_d_1", float32[:]),
    ("edge_d_start", float32[:]),
    ("edge_d_end", float32[:]),
]
# @jitclass(spec)
class InscribedSphereField:

    def __init__(self, origin_normal, tri_points) -> None:
        # Normal vector of the plane using the density field
        self.origin_normal = origin_normal

        # N x 3 x 3
        self.tri_points = tri_points

        # Normal vectors of the target plane for the inscribed circle
        self.t_normals = np.cross(tri_points[:,1] - tri_points[:,0], tri_points[:,2] - tri_points[:,0])
        self.t_normals = self.t_normals / np.expand_dims(np.sqrt(np.sum(self.t_normals ** 2, axis=1)), 1)
        self.t_ds = -np.sum(self.t_normals * tri_points[:,0], axis=1)

        # Linear inscribed distance formula between the planes ends up being a linear formula f(xyz) = N*xyz + D
        # The formula varies based on the sign of the origin, since is assumes the inscribed sphere has its center in the positive origin normal direction
        self.D_0, self.N_0 = plane_plane_distance(origin_normal, tri_points[:, 0], self.t_normals)
        self.D_1, self.N_1 = plane_plane_distance(-origin_normal, tri_points[:, 0], self.t_normals)
        
        # Vectors along the edge directions of the triangle

        self.tri_dirs = np.roll(tri_points, -1, axis=1) - tri_points
        self.tri_dirs = self.tri_dirs/np.expand_dims(np.sqrt(np.sum(self.tri_dirs**2, axis=-1)), 2)

        # Vectors orthogonal to the target plane and the tri_dirs
        # Used to check whether a point projected onto the target plane is inside the given triangle

        self.tri_normals = np.cross(self.tri_dirs, np.expand_dims(self.t_normals, 1), axis=-1)
        self.tri_normals = self.tri_normals / np.expand_dims(np.sqrt(np.sum(self.tri_normals ** 2, axis=-1)), 2)
        
        # Should be N x 3 x 3
        self.p_tri_normals_0 = projected_edge_plane(origin_normal, self.N_0, self.tri_normals)
        self.p_tri_d_0 = -np.sum(self.p_tri_normals_0 * self.tri_points, axis=-1)
        
        self.p_tri_normals_1 = projected_edge_plane(-origin_normal, self.N_1, self.tri_normals)
        self.p_tri_d_1 = -np.sum(self.p_tri_normals_1 * self.tri_points, axis=-1)

        self.edge_d_start = -np.sum(self.tri_dirs * tri_points, axis=-1)
        self.edge_d_end = -np.sum(self.tri_dirs * np.roll(tri_points, -1, axis=1), axis=-1)

# triangle = np.array([[0.2, 0.2, 0.1],[0.7, 0.2, 0.1],[0.2, 0.7, 0.1]])
# normal = np.array([0,0,1])
    def distance(self, xyz):
        # ones = np.ones(self.t_normals.shape[0])
        side = np.where(np.sum(self.t_normals*xyz, axis=-1) + self.t_ds >= 0 , 1, -1)

        inside_u = np.sum(self.p_tri_normals_0*xyz, axis=-1) <= -self.p_tri_d_0

        line_distances_u = (np.repeat(side, 3, axis=0) * line_distance(xyz, np.repeat(np.expand_dims(side, 1)*self.origin_normal, 3, axis=0), self.tri_points, self.tri_dirs)).reshape(-1,3)
        edge_greater_u = np.sum((np.expand_dims(line_distances_u, 2)*self.origin_normal + xyz)*self.tri_dirs, axis=-1) >= -self.edge_d_start
        edge_less_u =  np.sum((np.expand_dims(line_distances_u, 2)*self.origin_normal + xyz)*self.tri_dirs, axis=-1) <= -self.edge_d_end

        inside_d = np.sum(self.p_tri_normals_1*xyz, axis=-1) <= -self.p_tri_d_1

        line_distances_d = (np.repeat(side, 3, axis=0) * line_distance(xyz, -np.repeat(np.expand_dims(side, 1)*self.origin_normal, 3, axis=0), self.tri_points, self.tri_dirs)).reshape(-1,3)
        edge_greater_d = np.sum((np.expand_dims(line_distances_d, 2)*-self.origin_normal + xyz)*self.tri_dirs, axis=-1) >= -self.edge_d_start
        edge_less_d =  np.sum((np.expand_dims(line_distances_d, 2)*-self.origin_normal + xyz)*self.tri_dirs, axis=-1) <= -self.edge_d_end

        on_plane_u = np.all(inside_u, axis=-1)
        on_edge_u = np.any((inside_u==False) & edge_greater_u & edge_less_u, axis=-1)
        edge_u_idx = np.argmax((inside_u==False) & edge_greater_u & edge_less_u, axis=1)
        row_idx = np.arange(self.tri_points.shape[0])
        plane_u = np.abs(np.sum(self.N_0*xyz, axis=-1) + self.D_0)
        edge_u = np.abs(line_distance(xyz, self.origin_normal*np.expand_dims(side, 1), np.expand_dims(self.tri_points[row_idx, edge_u_idx],1), np.expand_dims(self.tri_dirs[row_idx, edge_u_idx],1))).flatten()
        points = np.min(np.abs(point_distance(xyz, self.origin_normal, self.tri_points)), axis=-1)
        u = np.where(on_plane_u, plane_u, np.where(on_edge_u, edge_u, points))
        
        on_plane_d = np.all(inside_d, axis=1)
        on_edge_d = np.any((inside_d==False) & edge_greater_d & edge_less_d, axis=1)
        edge_d_idx = np.argmax((inside_d==False) & edge_greater_d & edge_less_d, axis=1)

        row_idx = np.arange(self.tri_points.shape[0])
        plane_d = np.abs(np.sum(self.N_1*xyz, axis=-1) + self.D_1)
        edge_d = np.abs(line_distance(xyz, -self.origin_normal*np.expand_dims(side, 1), np.expand_dims(self.tri_points[row_idx, edge_d_idx],1), np.expand_dims(self.tri_dirs[row_idx, edge_d_idx],1))).flatten()
        points = np.min(np.abs(point_distance(xyz, -self.origin_normal, self.tri_points)), axis=-1)
        d = np.where(on_plane_d, plane_d, np.where(on_edge_d, edge_d, points))

        return np.min((np.min(u), np.min(d)))# min(min(u), min(d))



def triangle_inscribed_circle_field(triangles, origin_normal, field_coeff = 2, data_path = None):
    """constructs an inscribed triangle density field

    Args:
        points (_type_): vertices of the triangle
        origin_normal (_type_): normal of the plane utilizing the density field

    Returns:
        _type_: _description_
    """
    current = os.getcwd()
    if data_path is None:
        data_path = os.path.join(current, "src", "pebi_gmsh", "data", "constraint_array.npy")    

    np.save(data_path, np.r_[origin_normal.flatten(), triangles.flatten()])

    id = gmsh.model.mesh.field.add("ExternalProcess")
    gmsh.model.mesh.field.set_string(id, "CommandLine", 
                                    "python " + 
                                    "src/pebi_gmsh/utils_3D/plane_isd_field.py " + 
                                    str(field_coeff) + " " + 
                                    data_path
        )#os.path.join(current, "src", "pebi_gmsh", "utils_3D", "inscribed_circle_field.py"))
    
    #.join(map(str, np.vstack((origin_normal, vertices)).flatten()))
    return id




if __name__ == "__main__":

    dir_0 = np.array([1,0,0])

    dir_1 = np.random.rand(3)*2 - 1
    dir_1 = dir_1/np.linalg.norm(dir_1)


    p0 = np.zeros(3)
    p1 = np.random.rand(3)*2 - 1

    orthogonal_0 = np.array([0,0,1])
    orthogonal_1 = np.array([0,1,0])

    test_samples = np.linspace(0,2*np.pi, 180)
    test_dirs = np.sin(test_samples).reshape(-1,1)*orthogonal_0 + np.cos(test_samples).reshape(-1,1)*orthogonal_1

    smallest_dist = min([abs(line_distance(p0, test_dir, p1, dir_1)) for test_dir in test_dirs])

    dist = line_line_distance(p0, dir_0, p1, dir_1)

    print(smallest_dist, dist)