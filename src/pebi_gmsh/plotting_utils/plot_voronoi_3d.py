import numpy as np
import plotly.figure_factory as ff
from scipy.spatial import Voronoi
def get_voronoi_edges(voronoi, points):
    edge_set = set()
    for j, ridge in enumerate(voronoi.ridge_vertices):
        for i in range(len(ridge)):
            a = min(ridge[i],ridge[i-1])
            b = max(ridge[i],ridge[i-1])
            if a == -1:
                continue
            if (voronoi.ridge_points[j,0] not in points) or (voronoi.ridge_points[j,1] not in points):
                continue
            edge_set.add((a,b))
    return np.array(list(edge_set))
            

def inside_box(points, bounding_box, padding = 0):
    
    return  (points[:, 0] + padding >= bounding_box[0]) & \
            (points[:, 1] + padding >= bounding_box[2]) & \
            (points[:, 2] + padding >= bounding_box[4]) & \
            (points[:, 0] - padding <= bounding_box[1]) & \
            (points[:, 1] - padding <= bounding_box[3]) & \
            (points[:, 2] - padding <= bounding_box[5])


bounding_box = [0,1,0,1,0,1]
def plot_voronoi_3d(voronoi: Voronoi, b_plane_normals, b_plane_d, padding = .2, sites = None):
   
    # edges = get_voronoi_edges(voronoi, points)
    vertices = voronoi.vertices
    inside = np.ones(voronoi.points.shape[0], dtype=bool)
    for i in range(b_plane_normals.shape[0]):
        inside = np.logical_and(inside, np.sum(voronoi.points * b_plane_normals[i], axis=1) < -b_plane_d[i])        
    # accepted_sites = (voronoi.points[:,1] < 0.5) & inside_box(voronoi.points, bounding_box)

    border_ridges = np.where(inside[voronoi.ridge_points[:,0]] != inside[voronoi.ridge_points[:,1]])[0]
    tris = []
    for ridge_id in border_ridges:
        ridge = voronoi.ridge_vertices[ridge_id]
        for i, j in zip(ridge[1:-1], ridge[2:]):
            tris.append([ridge[0], i, j])
    
    tris = np.array(tris)
    
    tris = tris[np.any(tris == -1, axis = 1) == False]

    # inside = np.all(inside_box(vertices[tris].reshape(-1,3), bounding_box, 0.03).reshape(-1,3), axis=1)
    
    # tris = tris[inside]
    
    fig = ff.create_trisurf(vertices[:,0], vertices[:,1], vertices[:,2],tris, "Portland")
    
    # fig.add_scatter3d(x=voronoi.points[:,0], y=voronoi.points[:,1], z=voronoi.points[:,2], mode="markers")
    
    fig.update_layout(scene = {
        "xaxis": {"range": [bounding_box[0] - padding, bounding_box[1] + padding]},
        "yaxis": {"range": [bounding_box[2] - padding, bounding_box[3] + padding]},
        "zaxis": {"range": [bounding_box[4] - padding, bounding_box[5] + padding]},
    })
    fig.show()

    # ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
    # plt.show()
    # for edge in edges:
    #     edge_coords = vertices[edge]
    #     ax.plot(edge_coords[:,0], edge_coords[:,1], edge_coords[:,2], color="C0")