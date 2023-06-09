import gmsh
import numpy as np
from plotly import graph_objects as go
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
gmsh.initialize()
gmsh.model.add("nortched_rectangle")
gmsh.option.setNumber("Mesh.Algorithm", 6)
# gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
# gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
# gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

density = 0.03
radius = 0.5
gmsh.model.geo.add_point(0,0,0, meshSize=density, tag=1)
gmsh.model.geo.add_point(.75,0,0, meshSize=density, tag=2)
gmsh.model.geo.add_point(.75,1,0, meshSize=density, tag=3)

gmsh.model.geo.add_point(0 + radius,1,0, meshSize=density, tag=4)
gmsh.model.geo.add_point(0,1,0, meshSize=density, tag=5)
gmsh.model.geo.add_point(0,1-radius,0, meshSize=density, tag=6)


gmsh.model.geo.add_line(1,2, tag=1)
gmsh.model.geo.add_line(2,3, tag=2)
gmsh.model.geo.add_line(3,4, tag=3)

gmsh.model.geo.add_circle_arc(4,5,6, tag=4)
gmsh.model.geo.add_line(6,1, tag=5)

gmsh.model.geo.add_curve_loop([1,2,3,4,5], tag=1)
gmsh.model.geo.add_plane_surface([1], 1)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
# gmsh.fltk.run()

gmsh.model.mesh.create_faces()

_, node_coords, *_ = gmsh.model.mesh.get_nodes()
_, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)
node_coords = node_coords.reshape(-1,3)
tri_nodes = tri_nodes.reshape(-1,3) - 1
x = np.zeros(0)
y = np.zeros(0)
for tri in tri_nodes:
    x = np.r_[x, node_coords[tri, 0], node_coords[tri[0], 0], None]
    y = np.r_[y, node_coords[tri, 1], node_coords[tri[0], 1], None]

data = []
data.append(go.Scatter(
    x = x,
    y = y,
    mode="lines",
    line=dict(
        color='black',
        width=1.5,
        # size=10,
        # dash="dash"
    ),
    fill="toself",
    fillcolor="cyan"
))
fig = go.Figure(data=data,
    layout=dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width=750*1.5,
        height = 1000*1.5,
        xaxis=dict(showgrid=False, visible=False),# range=[x_start, x_stop]),
        yaxis=dict(showgrid=False, visible=False),# range=[0, frame_size]),
        plot_bgcolor='white',
    ))
# fig.show()
# fig.write_image("fem_artifacts_{}.svg".format(density))
density = 0.1

gmsh.finalize()
gmsh.initialize()

gmsh.model.add("rectangle")
gmsh.model.geo.add_point(0,0,0, meshSize=density, tag=1)
gmsh.model.geo.add_point(1,0,0, meshSize=density, tag=2)
gmsh.model.geo.add_point(1, 1,0, meshSize=density, tag=3)
gmsh.model.geo.add_point(0, 1,0, meshSize=density, tag=4)

gmsh.model.geo.add_line(1,2, tag=1)
gmsh.model.geo.add_line(2,3, tag=2)
gmsh.model.geo.add_line(3,4, tag=3)
gmsh.model.geo.add_line(4,1, tag=4)


gmsh.model.geo.add_curve_loop([1,2,3,4], tag=1)
gmsh.model.geo.add_plane_surface([1], 1)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
# gmsh.fltk.run()

gmsh.model.mesh.create_faces()

_, node_coords, *_ = gmsh.model.mesh.get_nodes()
_, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)
node_coords = node_coords.reshape(-1,3)
tri_nodes = tri_nodes.reshape(-1,3) - 1

x = np.zeros(0)
y = np.zeros(0)
for tri in tri_nodes:
    x = np.r_[x, node_coords[tri, 0], node_coords[tri[0], 0], None]
    y = np.r_[y, node_coords[tri, 1], node_coords[tri[0], 1], None]


fig, ax = plt.subplots()
plt.axis('off')
ax.plot(x,y, "r", zorder=-1, linewidth=0.8)
ax.scatter(node_coords[:,0], node_coords[:,1])
plt.ylim((0.1,0.9))
plt.xlim((0.1,0.9))
fig.tight_layout()
plt.savefig('triangles_to_voronoi_1.png', dpi=800)
# plt.show()


fig, ax = plt.subplots()
plt.axis('off')
voronoi = Voronoi(node_coords[:,:2])
voronoi_plot_2d(voronoi, ax = ax)
plt.ylim((0.1,0.9))
plt.xlim((0.1,0.9))
fig.tight_layout()
plt.savefig('triangles_to_voronoi_2.png', dpi=800)
# plt.show()