import gmsh
import numpy as np
# Initialing gmsh
gmsh.initialize()
gmsh.model.add("Demo")




# Point coordinates
# Must be in 3D, even if all z values are 0
boundary_points = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
])
# Ids/tags of the points added
boundary_point_tags = []
#Adding the points
for point in boundary_points:
    tag = gmsh.model.geo.add_point( x = point[0], y = point[1], z = point[2])
    boundary_point_tags.append(tag)
# mesh_size = [0.2, 0.01, 0.2, 0.01]
# for point, size in zip(boundary_points, mesh_size):
#     tag = gmsh.model.geo.add_point(point[0], point[1], point[2], meshSize = size)
#     boundary_point_tags.append(tag)

# End points of the line segments
edge_point_tags = np.array([
    [1,2],
    [2,3],
    [3,4],
    [4,1],
])

# Tags of the border edges
border_edge_tags = []
for edge in edge_point_tags:
    tag = gmsh.model.geo.add_line(edge[0], edge[1])
    border_edge_tags.append(tag)

# Adding the surface plane
border_loop_tag = gmsh.model.geo.add_curve_loop(border_edge_tags)
surface_tag = gmsh.model.geo.add_plane_surface([border_loop_tag])

# border_edge_tags = []
# for edge in edge_point_tags:
# tag = gmsh.model.geo.add_line(egde[0], edge[1])
# border_edge_tags.append(tag)
# gmsh.model.geo.mesh.set_transfinite_curve(border_edge_tags[0], 2)



# line_pos = np.linspace(0.3,0.7, 10, endpoint=True)\
#     .reshape(-1,1) * np.array([1,1,0])
# line_point_tags = []
# for point in line_pos:
#     tag = gmsh.model.geo.add_point(point[0], point[1], point[2])
#     line_point_tags.append(tag)
# line_tags = []
# for i in range(len(line_point_tags)-1):
#     tag = gmsh.model.geo.add_line(line_point_tags[i], line_point_tags[i+1])
#     line_tags.append(tag)

gmsh.model.geo.synchronize()

# gmsh.model.mesh.embed(1, line_tags, 2, 1)

# Updated the added geometry
# Generates the 2D mesh
gmsh.model.mesh.generate(2)
# Visualizes the mesh in the GMSH GUI
gmsh.fltk.run()

import matplotlib.pyplot as plt

gmsh.model.mesh.create_faces()
node_tags, node_coords, node_params = gmsh.model.mesh.get_nodes()
tri_tags, tri_node_tags = gmsh.model.mesh.get_all_faces(3)
# The tags and coordinates are returned as a flattened array
node_coords = node_coords.reshape(-1,3)


fig, ax = plt.subplots()
ax.triplot(node_coords[:,0], node_coords[:,1], triangles = tri_node_tags)
ax.set_aspect('equal', 'box')
plt.axis('off')
plt.savefig("gmsh_to_python.png", bbox_inches='tight', dpi=1000, pad_inches = 0)
plt.show()