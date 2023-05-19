import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.spatial import Voronoi
import trimesh 

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

def inside_mesh(points, mesh_verts, mesh_faces):
    mesh = trimesh.Trimesh((mesh_verts - np.array([0.5,0.5,0.5]))*1.01 + np.array([0.5,0.5,0.5]), faces=np.array(mesh_faces))
    return mesh.contains(points)





def plot_voronoi_3d(voronoi: Voronoi, mesh_verts, mesh_faces, padding = .2, data = None):# b_plane_normals, b_plane_d, padding = .2, sites = None):
   
    # edges = get_voronoi_edges(voronoi, points)
    vertices = voronoi.vertices
    # inside = inside_mesh(vertices, mesh_verts, mesh_faces)
    inside = inside_mesh(voronoi.points, mesh_verts, mesh_faces)#np.ones(voronoi.points.shape[0], dtype=bool)
    # for i in range(b_plane_normals.shape[0]):
    #     inside = np.logical_and(inside, np.sum(voronoi.points * b_plane_normals[i], axis=1) < -b_plane_d[i])        
    # accepted_sites = (voronoi.points[:,1] < 0.5) & inside_box(voronoi.points, bounding_box)

    # border_ridges = np.where(np.logical_and(inside[voronoi.ridge_points[:,0]], inside[voronoi.ridge_points[:,1]]))[0]
    tris = []
    for ridge_id in range(voronoi.ridge_points.shape[0]):
        points = voronoi.ridge_points[ridge_id]
        ridge = voronoi.ridge_vertices[ridge_id]
        if (inside[points[0]] != inside[points[1]]):
            for i, j in zip(ridge[1:-1], ridge[2:]):
                tris.append([ridge[0], i, j])
    
    tris = np.array(tris)
    
    tris = tris[np.any(tris == -1, axis = 1) == False]

    # inside = np.all(inside_box(vertices[tris].reshape(-1,3), bounding_box, 0.03).reshape(-1,3), axis=1)
    
    # tris = tris[inside]
    
    # fig = ff.create_trisurf(vertices[:,0], vertices[:,1], vertices[:,2],tris, "Portland")
    
    # # fig.add_scatter3d(x=voronoi.points[:,0], y=voronoi.points[:,1], z=voronoi.points[:,2], mode="markers")
    
    # fig.update_layout(scene = {
    #     "xaxis": {"range": [bounding_box[0] - padding, bounding_box[1] + padding]},
    #     "yaxis": {"range": [bounding_box[2] - padding, bounding_box[3] + padding]},
    #     "zaxis": {"range": [bounding_box[4] - padding, bounding_box[5] + padding]},
    # })
    # fig.layout.scene.camera.projection.type = "orthographic"
    # fig.show()
    plot_trimesh(vertices, tris, intensity=np.ones(vertices.shape[0]), data = data)
    # ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
    # plt.show()
    # for edge in edges:
    #     edge_coords = vertices[edge]
    #     ax.plot(edge_coords[:,0], edge_coords[:,1], edge_coords[:,2], color="C0")


def plot_3d_points(points, radii = None, color = None, data = None, return_data = False):
    if data is None:
        data = []
    if radii is None:
        radii = 5
    if color is None:
        color = points[:,2]
    
    
    trace = go.Scatter3d(
            x = points[:,0], y = points[:,1], z = points[:,2], mode = 'markers', marker = dict(
            size = radii,
            color = color, # set color to an array/list of desired values
            colorscale = 'Viridis',
        )
    )

    data.append(trace)
    if return_data:
        return data
    
    layout = go.Layout(title = '3D Scatter plot')
    fig = go.Figure(data = data, layout = layout)
    fig.layout.scene.camera.projection.type = "orthographic"
    fig.show()

def plot_trimesh(points, tris, intensity = None, colorscale="Viridis", plot_edges = True, padding = 0.1, data = None):
    if data is None:
        data = []
    if intensity is None:
        intensity = points[:,2]
        intensity = np.maximum(np.minimum(intensity, 1), 0)
    mesh = go.Mesh3d(
        x = points[:,0], 
        y = points[:,1], 
        z = points[:,2],
        colorscale=colorscale,
        intensity=intensity,
        i=tris[:,0],
        j=tris[:,1],
        k=tris[:,2],
        cmax=0, cmin=1
    )
    data.append(mesh)
    if plot_edges:
        tri_edges = points[tris]
        tri_edges = np.concatenate((tri_edges, tri_edges[:, None, 0,:], np.full((tri_edges.shape[0],1,3), None)), axis=1).reshape(-1,3) 
        lines = go.Scatter3d(
                   x=tri_edges[:, 0],
                   y=tri_edges[:, 1],
                   z=tri_edges[:, 2],
                   mode='lines',
                   name='',
                   line=dict(color= 'rgb(50,50,50)', width=1.5))
        data.append(lines)
    
    fig1 = go.Figure(data=data)
    fig1.update_layout(scene = {
        "xaxis": {"range": [bounding_box[0] - padding, bounding_box[1] + padding]},
        "yaxis": {"range": [bounding_box[2] - padding, bounding_box[3] + padding]},
        "zaxis": {"range": [bounding_box[4] - padding, bounding_box[5] + padding]},
        "aspectmode": 'cube'
    })
    fig1.layout.scene.camera.projection.type = "orthographic"
    fig1.show()