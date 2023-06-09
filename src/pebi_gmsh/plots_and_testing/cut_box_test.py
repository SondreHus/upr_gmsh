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
CEC.set_max_size(0.1)

front_box_points = np.array([
    [0, 0, 0],# 0
    [0, 0, 1],# 1
    [.25, 0, .25],# 2
    [1, 0, 0],# 3
])

loop_size = front_box_points.shape[0]

front_point_ids = CEC.add_vertices(front_box_points)

back_box_points = front_box_points + np.array([0,1,0])
back_point_ids = CEC.add_vertices(back_box_points)

loop_id = np.arange(loop_size)



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
# CEC.construct_face_padding(1)


plot_3d_points(CEC.vertex_coords, color = np.arange(CEC.vertex_coords.shape[0]))
CEC.fill_inner_loops()
# plot_trimesh(CEC.vertex_coords, CEC.triangles, intensity = CEC.vertex_radii)

# fig = ff.create_trisurf(x=CEC.vertex_coords[:,0], y=CEC.vertex_coords[:,1], z=CEC.vertex_coords[:,2],
#                         simplices=CEC.triangles,y
#                         title="Target tri-mesh", aspectratio=dict(x=1, y=1, z=1))
# fig.show()


data = plot_trimesh(CEC.vertex_coords, CEC.triangles, intensity = np.ones(CEC.vertex_coords.shape[0]))#CEC.vertex_radii)
fig = go.Figure(data=data)
fig.show()
tri_surf = TriangulatedSurface(CEC.vertex_coords, CEC.triangles, CEC.vertex_radii, CEC.radius_constricted)



outer, inner, _ = tri_surf.generate_voronoi_sites()

sites = np.vstack((outer, inner))
sites = sites[~np.any(np.isnan(sites), axis=1),:]

background_sites = add_background_sites(CEC.vertex_coords, CEC.edge_corners, np.vstack(CEC.constraint_tris), CEC.face_edges, sites, tri_surf.vertex_radii)


voronoi = Voronoi(np.vstack((sites, background_sites)))

mesh_points = np.vstack((front_box_points, back_box_points))

vertices = voronoi.vertices
inside = inside_mesh(vertices, mesh_points, CEC.face_corners)
vertices = vertices[inside]

# plot_3d_points(sites)
# plot_3d_points(vertices)


data = plot_3d_points(sites, return_data = True)
data = plot_voronoi_3d(voronoi, mesh_points, CEC.face_corners, cut_plane=(np.array([0,1,0]), 0.3), background_start = sites.shape[0])#, data=data)
fig = go.Figure(data=data,
    layout=dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width=1000,
        height = 625,
        scene = dict( 
            xaxis=dict(range=[-.1,1.1]),#[-., 1.5]),
            yaxis=dict(range=[-.1,1.1]),#[-.5, 1.5]),
            zaxis=dict(range=[-.1,1.1]),#[-.05, 2.05]),
            # aspectratio=dict(z=3)
        ),
        # scene_camera = camera
    )
)
fig.show()

