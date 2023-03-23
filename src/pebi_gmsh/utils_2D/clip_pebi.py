import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from pebi_gmsh.intersections_2d import polyline_intersections
from pebi_gmsh.polyline_interpolation import polyline_interpolation
import matplotlib.pyplot as plt


unit_square =np.array([[0,0],[1,0],[1,1], [0,1]])


def _append_vertices(voronoi: Voronoi, points):
    idx = list(range(voronoi.vertices.shape[0], voronoi.vertices.shape[0] + points.shape[0]))
    voronoi.vertices = np.r_[voronoi.vertices,points]
    return idx

def clip_pebi(voronoi: Voronoi, boundary = unit_square):

    # Setup usefull values
    area_max_dist = max(boundary[:,0]) - min(boundary[:,0]) + max(boundary[:,1]) - min(boundary[:,1])
    _, boundary_dist = polyline_interpolation(boundary)
    boundary_idx = _append_vertices(voronoi, boundary)
    boundary_loop = np.r_[boundary, [boundary[0]]]
    center = np.mean(voronoi.points, axis=0).reshape((1,-1))
    
    # Find all outside vertices
    outside = np.zeros(voronoi.vertices.shape[0], dtype=bool)
    for i in range(boundary_loop.shape[0]-1):
        dir_b = boundary_loop[i+1] - boundary_loop[i]
        dir_a = voronoi.vertices - boundary_loop[i]
        outside = np.logical_or(outside, (dir_a[:,0]*dir_b[1] - dir_a[:,1]*dir_b[0]) > 0)
    
    vertices_outside_idx = np.where(outside)[0]
    ridge_array = np.array(voronoi.ridge_vertices)
    outside_ridge = np.any(np.isin(ridge_array, vertices_outside_idx), axis=1)

    unbounded_ridge_idx = np.argwhere(np.logical_and(ridge_array[:,0] == -1, outside_ridge==False))[:,0]
    unbounded_ridge_site_idx = voronoi.ridge_points[unbounded_ridge_idx]
    unbounded_vertex_idx = ridge_array[unbounded_ridge_idx, 1]

    unbounded_medians = (voronoi.points[unbounded_ridge_site_idx[:, 1]] + voronoi.points[unbounded_ridge_site_idx[:, 0]])/2
    
    unbounded_dirs = voronoi.points[unbounded_ridge_site_idx[:, 1]] - voronoi.points[unbounded_ridge_site_idx[:, 0]]
    unbounded_dirs = np.c_[unbounded_dirs[:,1], -unbounded_dirs[:,0]]
    unbounded_dirs = unbounded_dirs*np.sign(np.sum((unbounded_medians - center) * unbounded_dirs,axis=1)).reshape((-1,1))/np.linalg.norm(unbounded_dirs, axis=1).reshape((-1,1))

    new_site_dict = {}
    for (site_a, site_b), vertex_id, dir in zip(unbounded_ridge_site_idx, unbounded_vertex_idx, unbounded_dirs):
        points, ii, _ = polyline_intersections(
            boundary_loop, 
            np.array([voronoi.vertices[vertex_id], voronoi.vertices[vertex_id] + dir * area_max_dist])
        )
        new_vert = _append_vertices(voronoi, points)[0]
        ridge_array = np.r_[ridge_array, [np.array([new_vert, vertex_id])]]
        voronoi.ridge_points = np.r_[voronoi.ridge_points, [np.array([site_a,site_b])]]
        
        dist = boundary_dist[ii[0]] + np.linalg.norm(points[0]-boundary_loop[ii[0]])
        new_site_dict[site_a, vertex_id] = new_site_dict.get((site_a, vertex_id), []) + [(new_vert, dist)]
        new_site_dict[site_b, vertex_id] = new_site_dict.get((site_b, vertex_id), []) + [(new_vert, dist)]


    ridge_vert_outside = np.isin(ridge_array, vertices_outside_idx)
    outside_ridge = np.logical_and(np.any(ridge_vert_outside, axis=1), np.all(ridge_vert_outside, axis=1) == False)
    outside_ridge_sites = voronoi.ridge_points[outside_ridge]
    ridge_vert_outside = ridge_vert_outside[outside_ridge]
    for i, ridge in enumerate(ridge_array[outside_ridge]):
        outside, inside = ridge[::(1 if ridge_vert_outside[i][0] else -1)]
        if inside == -1:
            continue
        site_a, site_b = outside_ridge_sites[i]

        points, ii, _ = polyline_intersections(
            boundary_loop, 
            np.array([voronoi.vertices[outside], voronoi.vertices[inside]])
        )
        new_vert = _append_vertices(voronoi, points)[0]
        ridge_array = np.r_[ridge_array, [np.array([new_vert, inside])]]
        voronoi.ridge_points = np.r_[voronoi.ridge_points, [np.array([site_a,site_b])]]
        dist = boundary_dist[ii[0]] + np.linalg.norm(points[0]-boundary_loop[ii[0]])
        new_site_dict[site_a, inside] = new_site_dict.get((site_a, inside), []) + [(new_vert, dist)]
        new_site_dict[site_b, inside] = new_site_dict.get((site_b, inside), []) + [(new_vert, dist)]


    vertices_outside_idx = np.r_[vertices_outside_idx,-1]
    outside_ridge = np.isin(ridge_array, vertices_outside_idx)
    voronoi.ridge_vertices = ridge_array[np.any(outside_ridge, axis = 1)==False].tolist()
    
    for point, region_idx in enumerate(voronoi.point_region):
        cell = voronoi.regions[region_idx]
        cell = np.array(cell)
        if not np.any(np.isin(cell, vertices_outside_idx)):
            continue
        if np.all(np.isin(cell, vertices_outside_idx)):
            continue
        vertex_outside = np.isin(cell, vertices_outside_idx)
        
        if vertex_outside[0]:
            roll = -np.where(vertex_outside==False)[0][0]
        else:
            roll = np.where(vertex_outside[::-1])[0][0]
        cell = np.roll(cell, roll)
        vertex_outside = np.roll(vertex_outside, roll)
        
        vi_idx_0 = cell[0]
        break_idx = np.where(vertex_outside)[0][0]
        vi_idx_1 = cell[break_idx-1]


        if vi_idx_1 != vi_idx_0:
            vertex_0_id, dist_0 = new_site_dict[point, vi_idx_0][0]
            vertex_1_id, dist_1 = new_site_dict[point, vi_idx_1][0]
        else:
            vertex_0_id, dist_0 = new_site_dict[point, vi_idx_0][0]
            vertex_1_id, dist_1 = new_site_dict[point, vi_idx_1][1]

        
        if abs(dist_0-dist_1) > boundary_dist[-1]/2:
            # cell includes start point of boundary
            dir = int(np.sign(dist_1-dist_0))
            corner_point_idx = np.argwhere((min(dist_0, dist_1) > boundary_dist) | (boundary_dist > max(dist_0, dist_1)))[::dir,0].tolist()
        else:
            dir = int(np.sign(-dist_1+dist_0))
            corner_point_idx = np.argwhere((min(dist_0, dist_1) < boundary_dist) & (boundary_dist < max(dist_0, dist_1)))[::dir,0].tolist()
        new_idx = [vertex_1_id] + [boundary_idx[corner_id] for corner_id in corner_point_idx] + [vertex_0_id]
        voronoi.ridge_vertices = voronoi.ridge_vertices + [[new_idx[i], new_idx[i+1]] for i in range(len(new_idx)-1)]
        voronoi.ridge_points = np.r_[voronoi.ridge_points, np.tile(np.array([site_a,site_b]), (len(new_idx)-1,1))]
        voronoi.regions[region_idx] = cell[:break_idx].tolist() + new_idx
        


 


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    angles = np.linspace(0,2*np.pi, 10, endpoint=False)

    # points = np.r_[np.c_[np.cos(angles), np.sin(angles)]*0.3 + 0.5]# np.array([[0.5,0.5]])]

    points = np.random.rand(10,2)
    voronoi = Voronoi(points)

    comparrison_vertices = np.array(voronoi.vertices)


    clip_pebi(voronoi)
    fig, ax = plt.subplots()
    voronoi_plot_2d(voronoi,ax = ax, show_vertices=False)
    plt.show()