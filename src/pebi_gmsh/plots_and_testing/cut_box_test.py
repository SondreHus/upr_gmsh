from plotly import figure_factory as ff
from plotly import graph_objects as go
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import numpy as np

from pebi_gmsh.constraints_3D.constrained_edges import ConstrainedEdgeCollection
from pebi_gmsh.constraints_3D.triangulated_surface import TriangulatedSurface
from pebi_gmsh.plotting_utils.plot_voronoi_3d import plot_voronoi_3d, inside_mesh, plot_trimesh, plot_3d_points


# # Triangle over plane test

# base = np.array([
#     [0,0,0],
#     [0,1,0],
#     [1,1,0],
#     [1,0,0],
#     [0.5, 0.7, 0.2],
#     [0.5, 0.4, 0.2],
#     [0.5, 0.7, 0.75]
# ])

# plot_trimesh(base, np.array([[0,1,2], [0,2,3], [4,5,6]]))

CEC = ConstrainedEdgeCollection()
CEC.set_max_size(0.1)
# CEC.set_max_size(0.02)
# CEC.set_max_size(0.025)
front_box_points = np.array([
    [0, 0, 0],# 0
    [0, 0, 1],# 1
    [.3, 0, .3],# 2
    [1, 0, 0],# 3
])

loop_size = front_box_points.shape[0]

front_point_ids = CEC.add_vertices(front_box_points)

back_box_points = front_box_points + np.array([0,1,0])
back_point_ids = CEC.add_vertices(back_box_points)

loop_id = np.arange(loop_size)

front_edges = front_point_ids[np.vstack((loop_id, np.roll(loop_id, -1))).T]
front_edge_ids = CEC.add_edges(front_edges)

back_edges = back_point_ids[np.vstack((loop_id, np.roll(loop_id, -1))).T]
back_edge_ids = CEC.add_edges(back_edges)

cross_edges = np.vstack((front_point_ids, back_point_ids)).T
cross_edge_ids = CEC.add_edges(cross_edges)

cross_faces = []
for n in range(loop_size):
    face_edges = np.array([front_edge_ids[n], cross_edge_ids[(n+1)%loop_size], back_edge_ids[n], cross_edge_ids[n]])
    cross_faces.append(CEC.add_face(face_edges))

front_face = CEC.add_face(front_edge_ids[::-1])
back_face = CEC.add_face(back_edge_ids)
# faces = np.array([
#     [0,1,2,3],
#     [0,5,6,4],
#     [6,7,8,9],
# ])

# CEC.add_vertices(box_points)
# CEC.add_edges(l_edges)
# CEC.add_face(faces[0])
# CEC.add_face(faces[1])
# CEC.add_face(faces[2])

CEC.populate_edge_vertices()
CEC.calculate_edge_vertex_radii()

for id in cross_faces:
    CEC.construct_face_padding(id)
# CEC.construct_face_padding(2)
# CEC.construct_face_padding(0)
# CEC.construct_face_padding(1)


plot_3d_points(CEC.vertex_coords, color = np.arange(CEC.vertex_coords.shape[0]))
CEC.fill_inner_loops()
# plot_trimesh(CEC.vertex_coords, CEC.triangles, intensity = CEC.vertex_radii)

# fig = ff.create_trisurf(x=CEC.vertex_coords[:,0], y=CEC.vertex_coords[:,1], z=CEC.vertex_coords[:,2],
#                         simplices=CEC.triangles,
#                         title="Target tri-mesh", aspectratio=dict(x=1, y=1, z=1))
# fig.show()


plot_trimesh(CEC.vertex_coords, CEC.triangles, intensity = np.ones(CEC.vertex_coords.shape[0]))#CEC.vertex_radii)
tri_surf = TriangulatedSurface(CEC.vertex_coords, CEC.triangles, CEC.vertex_radii, CEC.radius_constricted)



outer, inner, _ = tri_surf.generate_voronoi_sites()
print("GOT THROUGH IT")
sites = np.vstack((outer, inner))
sites = sites[~np.any(np.isnan(sites), axis=1),:]
voronoi = Voronoi(sites)

mesh_points = np.vstack((front_box_points, back_box_points))

vertices = voronoi.vertices
inside = inside_mesh(vertices, mesh_points, CEC.face_corners)
vertices = vertices[inside]

# plot_3d_points(sites)
# plot_3d_points(vertices)


data = plot_3d_points(sites, return_data = True)
plot_voronoi_3d(voronoi, mesh_points, CEC.face_corners, data=data)


