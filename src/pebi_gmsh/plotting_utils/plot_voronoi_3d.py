import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.spatial import Voronoi
from pebi_gmsh.utils_3D.tri_mesh import inside_mesh


def get_sphere_points(center, radius, theta_res = 100, tau_res = 50):
    theta = np.linspace(0,2*np.pi, theta_res, endpoint=True)
    tau = np.linspace(0, np.pi, tau_res, endpoint=True)

    theta, tau = np.meshgrid(theta, tau)
    z = radius*np.cos(tau) + center[2]
    r = radius*np.sin(tau)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return (x,y,z)

def plot_spheres(centers, radii, theta_res = 50, tau_res = 25):
    data = []
    centers = centers.reshape(-1,3)
    radii = radii.flatten()
    max_coord = -np.inf
    min_coord = np.inf
    for center, radius in zip(centers, radii):
        x, y, z = get_sphere_points(center, radius, theta_res, tau_res)
        max_coord = np.max((max_coord, np.max(x),np.max(y),np.max(z)))
        min_coord = np.min((min_coord, np.min(x),np.min(y),np.min(z)))
        data.append(go.Surface(
            x = x,
            y = y,
            z = z,

            colorscale=[[0, "cyan"], [1, "black"]],
            surfacecolor = np.zeros(x.shape[0]),
            # color = "cyan",
            opacity = 0.2,
            showscale=False,
        ))
    fig = go.Figure(
        data=data,
        layout=dict(
            scene=dict(
                aspectmode = "cube",
                aspectratio=dict(x=1, y=1, z=1),
                xaxis=dict(range=[min_coord, max_coord]),
                yaxis=dict(range=[min_coord, max_coord]),
                zaxis=dict(range=[min_coord, max_coord]),
            )
        )
    ).show()
    # scene = {
    #     "xaxis": {"range": [bounding_box[0] - padding, bounding_box[1] + padding]},
    #     "yaxis": {"range": [bounding_box[2] - padding, bounding_box[3] + padding]},
    #     "zaxis": {"range": [bounding_box[4] - padding, bounding_box[5] + padding]},
    #     "aspectmode": 'cube'

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




def plot_edges(vertices, edges, show_plot = False):
    x = np.array([])
    y = np.array([])
    z = np.array([])
    
    for edge in edges:
        x = np.r_[x, vertices[edge, 0], None]
        y = np.r_[y, vertices[edge, 1], None]
        z = np.r_[z, vertices[edge, 2], None]

    data = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                name='',
                line=dict(color= 'rgb(50,50,50)', width=3))
    
    if show_plot:
        go.Figure(data=[data]).show()
    else:
        return data

def plot_voronoi_3d(voronoi: Voronoi, mesh_verts, mesh_faces, padding = .2, data = None, cut_plane = None, background_start = None):# b_plane_normals, b_plane_d, padding = .2, sites = None):
    
    if data is None:
        data = []
    # edges = get_voronoi_edges(voronoi, points)
    vertices = voronoi.vertices
    # inside = inside_mesh(vertices, mesh_verts, mesh_faces)
    inside = inside_mesh(voronoi.points, mesh_verts, mesh_faces)#np.ones(voronoi.points.shape[0], dtype=bool)

    if background_start is None:
        background_start = voronoi.points.shape[0]

    if cut_plane is not None:
        plane_normal, plane_d = cut_plane
        inside = np.logical_and(np.sum(voronoi.points*plane_normal, axis=1) > plane_d, inside)
    
    # for i in range(b_plane_normals.shape[0]):
    #     inside = np.logical_and(inside, np.sum(voronoi.points * b_plane_normals[i], axis=1) < -b_plane_d[i])        
    # accepted_sites = (voronoi.points[:,1] < 0.5) & inside_box(voronoi.points, bounding_box)

    # border_ridges = np.where(np.logical_and(inside[voronoi.ridge_points[:,0]], inside[voronoi.ridge_points[:,1]]))[0]
    tris = []
    colors = []
    ridges = []
    for ridge_id in range(voronoi.ridge_points.shape[0]):
        points = voronoi.ridge_points[ridge_id]
        ridge = voronoi.ridge_vertices[ridge_id]
        if (inside[points[0]] != inside[points[1]]):
            if not -1 in ridge:
                inside_point = points[0] if inside[points[0]] else points[1]

                constrained_ridge = inside_point < background_start
                for i, j in zip(ridge[1:-1], ridge[2:]):
                    tris.append([ridge[0], i, j])
                    colors.append("yellow" if constrained_ridge else "white")
                ridges.append(ridge)
        
    tris = np.array(tris)
    
    # tris = tris[np.any(tris == -1, axis = 1) == False]

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
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    for ridge in ridges:
        x = np.r_[x, vertices[ridge, 0], vertices[ridge[0], 0], None]
        y = np.r_[y, vertices[ridge, 1], vertices[ridge[0], 1], None]
        z = np.r_[z, vertices[ridge, 2], vertices[ridge[0], 2], None]

    lines = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                name='',
                line=dict(color= 'rgb(50,50,50)', width=3))
    data.append(lines)

    

    return plot_trimesh(vertices, tris, plot_edges=False, face_color=colors, data = data)
    # ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
    # plt.show()
    # for edge in edges:
    #     edge_coords = vertices[edge]
    #     ax.plot(edge_coords[:,0], edge_coords[:,1], edge_coords[:,2], color="C0")


def plot_3d_points(points, radii = None, color = None, data = None, return_data = False):
    if data is None:
        data = []
    if radii is None:
        radii = 10
    if color is None:
        color = points[:,2]
    
    
    trace = go.Scatter3d(
            x = points[:,0], y = points[:,1], z = points[:,2], mode = 'markers', marker = dict(
            size = radii,
            color = color, # set color to an array/list of desired values
            colorscale = 'Viridis',
        ),
        showlegend=False
    )

    data.append(trace)
    if return_data:
        return data
    
    layout = go.Layout(title = '3D Scatter plot')
    fig = go.Figure(data = data, layout = layout)
    # fig.layout.scene.camera.projection.type = "orthographic"
    fig.show()

def plot_trimesh(points, tris, intensity = None, colorscale=None, plot_edges = True, padding = 0.1, data = None, color = "white", face_color = None):
    if data is None:
        data = []

    mesh = go.Mesh3d(
        x = points[:,0], 
        y = points[:,1], 
        z = points[:,2],
        i=tris[:,0],
        j=tris[:,1],
        k=tris[:,2],
        colorscale = colorscale,
        color=color,
        facecolor=face_color,
        intensity=intensity,
        flatshading=True,
        lighting=dict(
            ambient = 1,
            diffuse = 0
        )
    )
   
    data.insert(0,mesh)
    if plot_edges:
        tri_edges = points[tris]
        tri_edges = np.concatenate((tri_edges, tri_edges[:, None, 0,:], np.full((tri_edges.shape[0],1,3), None)), axis=1).reshape(-1,3) 
        lines = go.Scatter3d(
                   x=tri_edges[:, 0],
                   y=tri_edges[:, 1],
                   z=tri_edges[:, 2],
                   mode='lines',
                   name='',
                   line=dict(color= 'rgb(50,50,50)', width=3))
        data.append(lines)
    

    return data
    # fig1 = go.Figure(data=data)
    # fig1.update_layout(scene = {
    #     "xaxis": {"range": [bounding_box[0] - padding, bounding_box[1] + padding]},
    #     "yaxis": {"range": [bounding_box[2] - padding, bounding_box[3] + padding]},
    #     "zaxis": {"range": [bounding_box[4] - padding, bounding_box[5] + padding]},
    #     "aspectmode": 'cube'
    # })
    # fig1.layout.scene.camera.projection.type = "orthographic"
    # fig1.show()
