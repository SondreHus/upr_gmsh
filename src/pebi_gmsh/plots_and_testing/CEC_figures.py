from plotly import figure_factory as ff
from plotly import graph_objects as go
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import numpy as np

from pebi_gmsh.constraints_3D.constrained_edges import ConstrainedEdgeCollection
from pebi_gmsh.constraints_3D.triangulated_surface import TriangulatedSurface
from pebi_gmsh.plotting_utils.plot_voronoi_3d import plot_voronoi_3d, inside_mesh, plot_trimesh, plot_3d_points
from pebi_gmsh.constraints_3D.fill_voronoi_mesh import add_background_sites

CEC = ConstrainedEdgeCollection()

# Defining corner vertices coordinates 
# Standard unit cube

cube_points = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,0,1],
    [1,0,1],
    [1,1,1],
    [0,1,1], 
])

cube_idx = CEC.add_vertices(cube_points)

data = []

# data.append(go.Scatter3d(
#     x = CEC.vertex_coords[:,0], y = CEC.vertex_coords[:,1], z = CEC.vertex_coords[:,2], mode = 'markers', marker = dict(
#         size = 5,
#         color = np.ones(CEC.vertex_coords.shape[0]), # set color to an array/list of desired values
#         colorscale = 'Viridis',
#     ),
#     showlegend=False
# ))

# camera = dict(
#     up=dict(x=0, y=0, z=1),
#     center=dict(x=0, y=0, z=0),
#     eye=dict(x=1.6, y=1.1, z=1.1),
#     # projection = dict(type="orthographic")
# )

# fig = go.Figure(data=data,
#     layout = dict(
#         margin=dict(l=0,r=0,b=0,t=0),
#         width=1000,
#         height = 850,
#         scene_camera = camera,
#         scene = dict(
#             # xaxis = dict(visible=False, range=[center_coords[0]-0.1, center_coords[0]+0.1]),
#             # yaxis = dict(visible=False, range=[0, 0.2]),
#             # zaxis = dict(visible=False, range=[-.1, 0.1]),
#             # aspectratio=dict(x=2, y=2, z=2)
#             # aspectmode = "cube"
#         )
#     )                
# )
# fig.show()
# fig.write_image("CEC_cube_points.svg")


face_vertices = [
    [3,2,1,0], #bottom
    [4,5,6,7], #top
    [0,4,5,1], #Sides
    [1,5,6,2],
    [2,6,7,3],
    [3,7,4,0],
]
for face in face_vertices:
    CEC.add_face(face)

CEC.set_max_size(0.1)
CEC.calculate_edge_vertex_radii()
CEC.populate_edge_vertices()

# fan_start = CEC.vertex_coords.shape[0]

CEC.construct_face_padding()
CEC.fill_inner_loops()



# color = np.ones(CEC.vertex_coords.shape[0])
# color[fan_start:] += 1


# data.append(go.Scatter3d(
#     x = CEC.vertex_coords[:,0], y = CEC.vertex_coords[:,1], z = CEC.vertex_coords[:,2], mode = 'markers', marker = dict(
#         size = 5,
#         color = np.ones(CEC.vertex_coords.shape[0]),
#         # color = color, # set color to an array/list of desired values
#         colorscale = 'Viridis',
#     ),
#     showlegend=False
# ))

data = plot_trimesh(CEC.vertex_coords, CEC.triangles)

# x = np.array([])
# y = np.array([])
# z = np.array([])

# for tri in CEC.triangles:
#     x = np.r_[x, CEC.vertex_coords[tri, 0],  CEC.vertex_coords[tri[0], 0], None]
#     y = np.r_[y, CEC.vertex_coords[tri, 1],  CEC.vertex_coords[tri[0], 1], None]
#     z = np.r_[z, CEC.vertex_coords[tri, 2],  CEC.vertex_coords[tri[0], 2], None]

# for egde in CEC.edge_corners:
#     x = np.r_[x, CEC.vertex_coords[egde[0], 0],  CEC.vertex_coords[egde[1], 0], None]
#     y = np.r_[y, CEC.vertex_coords[egde[0], 1],  CEC.vertex_coords[egde[1], 1], None]
#     z = np.r_[z, CEC.vertex_coords[egde[0], 2],  CEC.vertex_coords[egde[1], 2], None]

# data.append(go.Scatter3d(
#     x = x, y = y, z = z, mode="lines",
#     showlegend=False,
#     line=dict(
#         color = "black"
#     )
# ))

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.6, y=1.1, z=1.1),
    # projection = dict(type="orthographic")
)
tri_surf = TriangulatedSurface(CEC.vertex_coords, CEC.triangles, CEC.vertex_radii, CEC.radius_constricted)
outer, inner, _ = tri_surf.generate_voronoi_sites()

data = plot_3d_points(np.vstack((inner, outer)), radii = 5, color = "yellow", data = data, return_data=True)
fig = go.Figure(data=data,
    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,b=0,t=0),
        width=1000,
        height = 850,
        scene_camera = camera,
        scene = dict(
            # xaxis = dict(visible=False, range=[center_coords[0]-0.1, center_coords[0]+0.1]),
            # yaxis = dict(visible=False, range=[0, 0.2]),
            # zaxis = dict(visible=False, range=[-.1, 0.1]),
            # aspectratio=dict(x=2, y=2, z=2)
            # aspectmode = "cube"
        )
    )                
)
fig.show()
fig.write_image("CEC_cube_sites.svg")

sites = np.vstack((outer, inner))
background_sites = add_background_sites(
    CEC.vertex_coords, CEC.edge_corners, np.vstack(CEC.constraint_tris), 
    CEC.face_edges, 
    sites, 
    tri_surf.vertex_radii
)
voronoi = Voronoi(np.vstack((sites, background_sites)))


data = plot_voronoi_3d(voronoi, cube_points, CEC.face_corners, cut_plane=(np.array([-.6,-.6,-.6]), -1.33), background_start = sites.shape[0])#, data=data)
fig = go.Figure(data=data,
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,b=0,t=0),
        width=1000,
        height = 850,
        scene_camera = camera,
        scene = dict( 
            xaxis=dict(range=[-.1,1.1]),#[-., 1.5]),
            yaxis=dict(range=[-.1,1.1]),#[-.5, 1.5]),
            zaxis=dict(range=[-.1,1.1]),#[-.05, 2.05]),
            aspectmode="cube",
            # aspectratio=dict(z=3)
        ),
        # scene_camera = camera
    )
)
fig.show()
fig.write_image("CEC_cube_voronoi.svg")
# plot_3d_points(CEC.vertex_coords, color= np.ones(CEC.vertex_coords.shape[0]))


