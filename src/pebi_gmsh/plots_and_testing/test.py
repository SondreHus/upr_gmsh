import gmsh
import numpy as np
from pebi_gmsh.utils_3D.plane_densityfield import triangle_inscribed_circle_field
from pebi_gmsh.constraints_3D.triangulated_surface import TriangulatedSurface
from pebi_gmsh.constraints_3D.constrained_edges import ConstrainedEdgeCollection
import os
from plotly import graph_objects as go
from pebi_gmsh.plotting_utils.plot_voronoi_3d import plot_trimesh, plot_3d_points, plot_voronoi_3d, inside_mesh
from scipy.spatial import Voronoi
base_square = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
])


skew_plane = np.array([
    [0,0,.3],
    [1,0,.3],
    [1,1,.05],
    [0,1,.05],
])

triangle_ids = np.array([
    [0,1,2],
    [2,3,0]
])

skew_edges = np.array([
    [0,1],
    [1,2],
    [2,3],
    [3,0]
])


data = []

data.append(go.Scatter3d(
    x=np.r_[base_square[:,0], base_square[0,0], None, skew_plane[:,0], skew_plane[0,0]],
    y=np.r_[base_square[:,1], base_square[0,1], None, skew_plane[:,1], skew_plane[0,1]], 
    z=np.r_[base_square[:,2], base_square[0,2], None, skew_plane[:,2], skew_plane[0,2]],
    mode="lines",
    line=dict(
        color='black',
        width=2
    ),
    showlegend=False
))

data.append(go.Mesh3d(
    x = base_square[:,0],
    y = base_square[:,1],
    z = base_square[:,2],
    i = triangle_ids[:,0],
    j = triangle_ids[:,1],
    k = triangle_ids[:,2],
    color='white'
))
data.append(go.Mesh3d(
    x = skew_plane[:,0],
    y = skew_plane[:,1],
    z = skew_plane[:,2],
    i = triangle_ids[:,0],
    j = triangle_ids[:,1],
    k = triangle_ids[:,2],
    color='white'
))

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.3, y=1.3, z=0.7)
)

layout = dict(
        scene_camera = camera,
        margin=dict(l=0,r=0,b=0,t=0),
        width=1000,
        height = 625,
        scene = dict(
            xaxis = dict(visible=False, range = [0,1]),
            yaxis = dict(visible=False, range = [0,1]),
            zaxis = dict(visible=False, range = [-.5,.5]),
            aspectmode = "cube"
        ),
        showlegend=False
)    

fig = go.Figure(layout=layout, data=data)
fig.show()
fig.write_image("skew_test_case.svg")

gmsh.initialize()
gmsh.model.add("plane_1")

base_points = []
for point in base_square:
    base_points.append(gmsh.model.geo.add_point(point[0], point[1], point[2], 0.35))

base_edges = []
for i in range(len(base_points)):
    base_edges.append(gmsh.model.geo.add_line(base_points[i], base_points[(i+1)%len(base_points)]))

base_loop = gmsh.model.geo.add_curve_loop(base_edges)
base_plane = gmsh.model.geo.add_plane_surface([base_loop])


gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


skew_triangle_coords = skew_plane[triangle_ids]

# Sets up the inscribed circle size field
current = os.getcwd()
density_field = triangle_inscribed_circle_field(skew_triangle_coords, np.array([0,0,1]), 1, data_path=os.path.join(current, "skew_planes.npy"))

gmsh.model.mesh.field.setAsBackgroundMesh(density_field)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.create_faces()
# gmsh.fltk.run()

_, node_coords, *_ = gmsh.model.mesh.get_nodes()
_, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)
node_coords = node_coords.reshape(-1,3)
tri_nodes = tri_nodes.reshape(-1,3)-1

all_coords = node_coords
all_tris = tri_nodes
gmsh.model.add("plane_2")



base_points = []
for point in skew_plane:
    base_points.append(gmsh.model.geo.add_point(point[0], point[1], point[2], 0.35))

base_edges = []
for i in range(len(base_points)):
    base_edges.append(gmsh.model.geo.add_line(base_points[i], base_points[(i+1)%len(base_points)]))

base_loop = gmsh.model.geo.add_curve_loop(base_edges)
base_plane = gmsh.model.geo.add_plane_surface([base_loop])


gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)



normal = np.cross(skew_plane[1]-skew_plane[0], skew_plane[-1]-skew_plane[0])
normal = normal/np.linalg.norm(normal)
skew_triangle_coords = skew_plane[triangle_ids]

# Sets up the inscribed circle size field
current = os.getcwd()
density_field = triangle_inscribed_circle_field(base_square[triangle_ids], normal, 1.4, data_path=os.path.join(current, "skew_planes.npy"))

gmsh.model.mesh.field.setAsBackgroundMesh(density_field)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.create_faces()
# gmsh.fltk.run()

_, node_coords, *_ = gmsh.model.mesh.get_nodes()
_, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)
node_coords = node_coords.reshape(-1,3)
tri_nodes = tri_nodes.reshape(-1,3)-1

all_tris = np.vstack((all_tris, tri_nodes + all_coords.shape[0]))
all_coords = np.vstack((all_coords, node_coords))




trimesh = TriangulatedSurface(all_coords, all_tris)
inner, outer, _ = trimesh.generate_voronoi_sites()
all_sites = np.vstack((inner, outer))

mesh_faces = np.array([
    [3,2,1,0],
    [4,5,6,7],
    [0,1,5,4],
    [1,2,6,5],
    [2,3,7,6],
    [3,0,4,7],
])


inside_sites = inside_mesh(all_sites, np.vstack((base_square, skew_plane)), mesh_faces)

# Plotting triangulation with sites

data = []
data = plot_trimesh(all_coords, all_tris, data=data)

fig = go.Figure(layout=layout, data=data)
fig.show()
fig.write_image("skew_test_triangulation.svg")

data = plot_3d_points(all_sites, color = np.where(inside_sites, 0, 1), data=data, return_data = True)
fig = go.Figure(layout=layout, data=data)
fig.show()
fig.write_image("skew_test_triangulation_with_points.svg")

# plot_3d_points(all_sites)

voronoi = Voronoi(all_sites)

# mesh_faces = np.array([
#     [0,1,2],
#     [2,3,0],
#     [6,5,4],
#     [0,7,6],
# ])

# side_edges_0 = np.c_[np.arange(4), np.arange(4)+4, (np.arange(4)+1)%4 + 4]
# side_edges_1 = np.c_[np.arange(4), (np.arange(4)+1)%4 + 4, (np.arange(4)+1)%4]

# mesh_faces = np.vstack((mesh_faces, side_edges_0, side_edges_1))




mesh_verts = np.vstack((base_square, skew_plane))
mesh_verts[:,0] = mesh_verts[:,0]*0.8 + 0.1
data = plot_voronoi_3d(voronoi, mesh_verts, mesh_faces)
fig = go.Figure(layout=layout, data=data)
fig.show()
fig.write_image("skew_voronoi.svg")
# skew__tris = np.array([
#     [0,1,2],
#     [2,3,0]
# ])


# CEC = ConstrainedEdgeCollection()

# CEC.add_vertices(np.vetack(base_square, skew_plane))
# CEC.add_edges(np.vstack(base_edges, skew_edges + 4))
# CEC.add_face(np.array([0,1,2,3]))
# CEC.add_face(np.array([4,5,6,7]))
# CEC.




