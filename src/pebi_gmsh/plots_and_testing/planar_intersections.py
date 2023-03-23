import gmsh
from math import pi
from pebi_gmsh.constraints_3D.triangulated_surface import TriangulatedSurface
from pebi_gmsh.utils_3D.sphere_intersection import (sphere_intersections, flatten_sphere_centers)
from pebi_gmsh.utils_2D.circumcircle import circumcircle
from pebi_gmsh.plotting_utils.plot_voronoi_3d import plot_voronoi_3d
from scipy.spatial import Voronoi, Delaunay
import matplotlib.pyplot as plt
import numpy as np
gmsh.initialize()
model_name = "planar_intersections"
gmsh.model.add(model_name)
factory = gmsh.model.occ

# A box
box = factory.addBox(0.01, 0.01, 0.01, 0.99, 0.99, 0.99)

# A plane
xyz = [[0.5, 0.2, 0.2],
       [0.5, 0.6, 0.2],
       [0.5, 0.6, 0.8],
       [0.5, 0.2, 0.8]]
pts = [factory.addPoint(x[0], x[1], x[2]) for x in xyz]
lines = [factory.addLine(pts[k - 1], pts[k]) for k in range(len(pts))]
cloop = factory.addCurveLoop(lines)
surf = factory.addPlaneSurface([cloop])
surf_dt = [(2, surf)]

# Another plane
surfcopy_dt = factory.copy(surf_dt)
factory.translate(surfcopy_dt, 0, 0.2, 0)
factory.rotate(surfcopy_dt, 0.5, 0.5, 0.5, 0, 1, 0, pi / 4)

# Put the planes in the box
factory.fragment([(3, box)], surf_dt + surfcopy_dt)
# factory.fragment(surf_dt, surfcopy_dt)

# Add automatic labels for all objects
factory.synchronize()
for dim in range(0, 4):
    for dt in gmsh.model.getEntities(dim):
        gmsh.model.addPhysicalGroup(dim, [dt[1]])

# Add a special label for the (single) intersecting line between the
# two surfaces
intersection = gmsh.model.mesh.getEmbedded(2, surf_dt[0][1])
# assert len(intersection) == 1
# assert intersection == gmsh.model.mesh.getEmbedded(2, surfcopy_dt[0][1])
# special_label = 999
# gmsh.model.addPhysicalGroup(1, [intersection[0][1]], special_label)

# Mesh
gmsh.model.mesh.generate(2)

# Dump
# gmsh.write(model_name + ".msh")
# gmsh.write(model_name + ".mesh")
# gmsh.write(model_name + ".m")


# Get the node coordinates of the model surfaces
gmsh.model.mesh.create_faces()
node_ids, node_coords, *_ = gmsh.model.mesh.get_nodes()
tri_ids, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)

node_coords = node_coords.reshape(-1,3)
tri_nodes = tri_nodes.reshape(-1,3) - 1
# Get a simple example of 
radii_aspect = 1.03

node_radii = np.zeros(node_coords.shape[0])# - np.inf
flattened_coords, *_ = flatten_sphere_centers(node_coords[tri_nodes])
_, distances = circumcircle(flattened_coords[:,0,:2], flattened_coords[:,1,:2], flattened_coords[:,2,:2])
# offsets = flattened_coords[:,:,:2] - np.sum(flattened_coords[:, :, :2], axis=1).reshape(-1,1,2) / 3
# distances = np.sqrt(np.sum(offsets * offsets, axis=2))
for i, distance in enumerate(distances):
    node_radii[tri_nodes[i]] = np.where(node_radii[tri_nodes[i]] > distance , node_radii[tri_nodes[i]], distance)

node_radii = node_radii * radii_aspect

# gmsh.fltk.run()
gmsh.finalize()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

tri_surface = TriangulatedSurface(node_coords, tri_nodes)
a, b, vertex_neighbours = tri_surface.generate_voronoi_sites()
points = np.vstack((a,b))
# ax.scatter(points[:,0], points[:,1], points[:,2])
points = points[~np.isnan(points[:,0])]
voronoi = Voronoi(points)

plot_voronoi_3d(voronoi)


print("oy")
gmsh.initialize()
gmsh.model.add("voronoi_generation")
factory = gmsh.model.occ
bounds_min = np.min(points, axis=0)
bounds_max = np.max(points, axis=0)
generation_box = factory.addBox(*(bounds_min.tolist() + bounds_max.tolist()))

embedded_points = [factory.add_point(p[0], p[1], p[2]) for p in points]

factory.synchronize()

gmsh.model.mesh.embed(0, embedded_points, 3, generation_box)

gmsh.model.mesh.generate(3)

# Get the node coordinates of the model surfaces
gmsh.model.mesh.create_faces()
node_ids, node_coords, *_ = gmsh.model.mesh.get_nodes()
node_coords = node_coords.reshape(-1,3)
keep_node = np.ones(node_coords.shape[0], dtype=bool)


delaunay = Delaunay(node_coords)
delaunay.neighbors
neighbouring_nodes = [[]]*node_coords.shape[0]

voronoi = Voronoi(node_coords)
for a, b in voronoi.ridge_points:
    neighbouring_nodes[a].append(b)
    neighbouring_nodes[b].append(a)


print("Yo")
