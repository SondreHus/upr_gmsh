from plotly import graph_objects as go
from scipy.spatial import Voronoi
import numpy as np

from pebi_gmsh.constraints_3D.constrained_edges import ConstrainedEdgeCollection
from pebi_gmsh.constraints_3D.triangulated_surface import TriangulatedSurface
from pebi_gmsh.plotting_utils.plot_voronoi_3d import plot_voronoi_3d, plot_trimesh, plot_3d_points
from pebi_gmsh.constraints_3D.fill_voronoi_mesh import add_background_sites


# Set up CEC
CEC = ConstrainedEdgeCollection()
CEC.set_max_size(0.1)


# Vertices of one of the sides
front_box_points = np.array([
    [0.66, 0.2, 0],# 0
    [0.409, 0.2, 0.33],# 1
    [0.88, 0.2, 0.66],# 2
    [0.33, 0.2, 1],# 2
    [0.59, 0.2, 0.66],# 2
    [0.11, 0.2, 0.33],# 2
])


loop_size = front_box_points.shape[0]

# Add thickness to the mesh

back_box_points = front_box_points + np.array([0,.4,0])

# Add all the vertices
front_point_ids = CEC.add_vertices(front_box_points)
back_point_ids = CEC.add_vertices(back_box_points)

loop_id = np.arange(loop_size)
inner_plane_points = CEC.add_vertices


# Add the faces of the constraint
front_face = CEC.add_face(front_point_ids[::-1])
back_face = CEC.add_face(back_point_ids)
cross_faces = []
for n in range(loop_size):

    cross_faces.append(CEC.add_face([front_point_ids[n], front_point_ids[(n+1)%loop_size], back_point_ids[(n+1)%loop_size], back_point_ids[n]]))


CEC.calculate_edge_vertex_radii()
CEC.populate_edge_vertices()
plot_3d_points(CEC.vertex_coords, color = np.arange(CEC.vertex_coords.shape[0]))
for id in cross_faces:
    CEC.construct_face_padding(id)
CEC.construct_face_padding(front_face)
CEC.construct_face_padding(back_face)


plot_3d_points(CEC.vertex_coords, color = np.arange(CEC.vertex_coords.shape[0]))
CEC.fill_inner_loops()


# Show the target ridges
data = plot_trimesh(CEC.vertex_coords, CEC.triangles, intensity = np.ones(CEC.vertex_coords.shape[0]))#CEC.vertex_radii)
fig = go.Figure(data=data)
fig.show()

# Get a radius solution
tri_surf = TriangulatedSurface(CEC.vertex_coords, CEC.triangles, CEC.vertex_radii, CEC.radius_constricted)

outer, inner, _ = tri_surf.generate_voronoi_sites()

sites = np.vstack((outer, inner))

# If by chance any nan site shows up, this prevents Scipy from throwing a fit
sites = sites[~np.any(np.isnan(sites), axis=1),:]

background_sites = add_background_sites(CEC.vertex_coords, CEC.edge_corners, np.vstack(CEC.constraint_tris), CEC.face_edges, sites, tri_surf.vertex_radii)

voronoi = Voronoi(np.vstack((sites, background_sites)))

mesh_points = np.vstack((front_box_points, back_box_points))

# Plotting

data = plot_voronoi_3d(voronoi, mesh_points, np.vstack(CEC.constraint_tris), cut_plane=(np.array([0,1,0]), .45), background_start = sites.shape[0])#, data=data)

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.5, y=-1.5, z=.3),
)

fig = go.Figure(data=data,
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,b=0,t=0),
        width=1000,
        height = 850,
        scene_camera = camera,
        scene = dict( 
            xaxis=dict(range=[0,1]),
            yaxis=dict(range=[0,1]),
            zaxis=dict(range=[-0,1]),
            aspectmode="cube",

        ),

    )
)
fig.show()
