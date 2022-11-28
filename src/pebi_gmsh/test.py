
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from pebi_gmsh.convert_GMSH import convert_GMSH
from pebi_gmsh.generate_constrained_mesh import generate_constrained_mesh_2d
import gmsh
from pebi_gmsh.site_locations import create_site_locations
from pebi_gmsh.site_data import (FConstraint, CConstraint)



# Face constraint test


# x = np.linspace(0.05,0.95,30)
# y = 0.3*np.sin(4*x) + 0.35
# x = np.array([0.1,0.5, 0.9])
# y = np.array([0.1, 0.3, 0.1])
# c_constraint = CConstraint(np.c_[x,y], 0.05, protection_sites=1)



# site_data = create_site_locations(c_constraints=[c_constraint])
# sites = site_data.sites



# mesh = generate_constrained_mesh_2d(site_data, h0=0.3, popup=True)

# fig, ax = plt.subplots()


# ax.plot(x,y, "r--", zorder=-10, linewidth=3)
# ax.set_ylim((0,1))
# ax.set_xlim((0,1))

# ax.scatter(sites[:,0], sites[:,1], marker=".")
# ax.axis('off')
# voronoi = scipy.spatial.Voronoi(mesh["node_coords"].reshape((-1,3))[:,:2])
# scipy.spatial.voronoi_plot_2d(voronoi, ax=ax, show_points=False, show_vertices=False)

# plt.show()




gmsh.initialize()
gmsh.model.add("MRST")
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.model.geo.addPoint(-2, -2, 0, 0.1, 0)
gmsh.model.geo.addPoint(-2, 2, 0, 0.1, 1)
gmsh.model.geo.addPoint(3, 2, 0, 0.1, 2)
gmsh.model.geo.addPoint(3, -2, 0, 0.1, 3)

gmsh.model.geo.addLine(0,1,0)
gmsh.model.geo.addLine(1,2,1)
gmsh.model.geo.addLine(2,3,2)
gmsh.model.geo.addLine(3,0,3)

gmsh.model.geo.addCurveLoop([0,1,2,3], 1)


f_line_0 = np.array([[-0.4,1],[2,-0.5]])


x = np.linspace(-1,1.5,1000)
y = np.sin(3*x)*0.48

f_line_1 = np.c_[x,y]

f_constraint_0 = FConstraint(f_line_0, 0.01)
f_constraint_1 = FConstraint(f_line_1, 0.01)

ax = plt.axes()

data = create_site_locations([f_constraint_0, f_constraint_1], )
sites = data.sites
edges = data.edges 
edge_loops = data.f_edge_loops
# plt.scatter(sites[:,0], sites[:,1])
# plt.show()

constraint_site_idx = []
for site in sites:
    site_point = gmsh.model.geo.add_point(site[0],site[1],0, 0.5)
    constraint_site_idx.append(site_point)
constraint_edge_idx = []

for edge in edges:
    edge_id = gmsh.model.geo.addLine(constraint_site_idx[edge[0]], constraint_site_idx[edge[1]])
    constraint_edge_idx.append(edge_id)



gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()
gmsh.model.mesh.embed(1,constraint_edge_idx, 2, 1)

extend = gmsh.model.mesh.field.add("Extend")
gmsh.model.mesh.field.set_numbers(extend, "CurvesList", constraint_edge_idx + [0,1,2,3])
gmsh.model.mesh.field.set_number(extend, "DistMax", 2)
gmsh.model.mesh.field.set_number(extend, "SizeMax", 0.5)
constant = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.set_string(constant, "F", str(0.5))
min = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.set_numbers(min, "FieldsList", [constant, extend])

gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

# gmsh.model.mesh.field.set_number(extend, "CurvesList", [edge_idx])
gmsh.model.mesh.field.setAsBackgroundMesh(min)

gmsh.model.mesh.generate(2)
gmsh.model.mesh.create_faces()
node_ids, node_coords, *_ = gmsh.model.mesh.get_nodes()
voronoi = scipy.spatial.Voronoi(node_coords.reshape((-1,3))[:,:2])
ax.scatter(sites[:,0],sites[:,1])
ax.axis("equal")
scipy.spatial.voronoi_plot_2d(voronoi,show_vertices=False,show_points=False, ax=ax)
plt.show()

gmsh.fltk.run()