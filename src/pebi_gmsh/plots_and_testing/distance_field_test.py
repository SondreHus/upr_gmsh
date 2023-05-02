import gmsh
from pebi_gmsh.constraints_3D.triangulated_surface import TriangulatedSurface
from pebi_gmsh.utils_3D.sphere_intersection import (sphere_intersections, flatten_sphere_centers)
from pebi_gmsh.utils_2D.circumcircle import circumcircle
from pebi_gmsh.plotting_utils.plot_voronoi_3d import plot_voronoi_3d
from pebi_gmsh.utils_3D.densityfield import InscribedCircleField
from scipy.spatial import Voronoi, Delaunay
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import os
from time import time
current = os.getcwd()
# Test of planar distance

factory = gmsh.model.occ

points_plane_0 = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
])

# points_plane_1 = np.array([
#     [0, 1, 0.001],
#     [0, 0, 0.001],
#     [1, 0, 0.2],
#     [1, 1, 0.2],
# ])

# normal_0 = np.cross(points_plane_0[1]-points_plane_0[0], points_plane_0[3]-points_plane_0[0])
# normal_0 = normal_0/np.sqrt(np.sum(normal_0**2, axis=-1))

# normal_1 = np.cross(points_plane_1[1]-points_plane_1[0], points_plane_1[3]-points_plane_1[0])
# normal_1 = normal_1/np.sqrt(np.sum(normal_1**2, axis=-1))

# D_0, n_0 = planar_distance(normal_0, points_plane_1[0], -normal_1)
# D_1, n_1 = planar_distance(normal_1, points_plane_0[0], -normal_0)


# def generate_constrained_plane(points, normals, Ds):
#     gmsh.initialize()
#     gmsh.model.add("distance_field_test")

#     point_idx = []
#     for row in points:
#         point_idx.append(gmsh.model.geo.add_point(row[0], row[1], row[2]))

#     line_idx = []
#     for i in range(4):
#         line_idx.append(gmsh.model.geo.add_line(point_idx[i], point_idx[(i+1) % 4]))

#     loop = gmsh.model.geo.add_curve_loop(line_idx)
#     plane = gmsh.model.geo.add_plane_surface([loop])

#     gmsh.model.geo.synchronize()

#     gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
#     gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
#     gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
#     fields = []
#     for n, d in zip(normals, Ds):    
#         inscribed_circle = gmsh.model.mesh.field.add("MathEval")
#         gmsh.model.mesh.field.set_string(inscribed_circle, "F", "abs(0.66*(x*({}) + y*({}) + z*({}) + ({})))".format(n[0], n[1], n[2], d))
#         fields.append(inscribed_circle)

#     min = gmsh.model.mesh.field.add("Min")
#     gmsh.model.mesh.field.set_numbers(min, "FieldsList", fields)
#     gmsh.model.mesh.field.setAsBackgroundMesh(min)

#     gmsh.model.mesh.generate(2)
#     gmsh.model.mesh.create_faces()
#     node_ids, node_coords, *_ = gmsh.model.mesh.get_nodes()
#     tri_ids, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)

#     node_coords = node_coords.reshape(-1,3)
#     tri_nodes = tri_nodes.reshape(-1,3) - 1
#     # gmsh.fltk.run()
#     return node_coords, tri_nodes

# node_coords_0, tri_nodes_0 = generate_constrained_plane(points_plane_0, [n_0], [D_0])
# node_coords_1, tri_nodes_1 = generate_constrained_plane(points_plane_1, [n_1], [D_1])

# tri_nodes_1 = tri_nodes_1 + node_coords_0.shape[0]

# surface = TriangulatedSurface(np.vstack((node_coords_0, node_coords_1)), np.vstack((tri_nodes_0, tri_nodes_1)))
# o, i, _ = surface.generate_voronoi_sites()

# voronoi = Voronoi(np.vstack((o,i)))
# normals = np.array([
#     [-1,0,0],
#     [1,0,0],
#     [0,-1,0],
#     [0,1,0],
#     [0, 0, -1], 
#     normal_1
# ])

# ds = np.array([
#     0.01,
#     -0.9,
#     0.2,
#     -0.8,
#     -np.dot(normal_0, points_plane_0[0]), 
#     -np.dot(normal_1, points_plane_1[0])
# ])
# plot_voronoi_3d(voronoi, normals, ds)
# min = gmsh.model.mesh.field.add("Min")
# gmsh.model.mesh.field.set_numbers(min, "FieldsList", [constant, extend])


# Test of point distance




def create_test_surface(m, n, w=1, h=1, f = lambda x, y: x*0.1 + 0.1):

    surface = []
    # tris = [[0,1,10]]
    tris = []
    for x in range(m):
        for y in range(n):
            surface.append([x*w/(m-1),y*h/(n-1), f(x*w/(m-1),y*h/(n-1))])
    
    for x in range(m-1):
        for y in range(n-1):
            if x%2 == 0:
                tris.append([n*x + y, n*x + y + 1, n*x + y + n + 1])
                tris.append([n*x + y, n*x + y + n + 1, n*x + y + n])
            else:
                tris.append([n*x + y, n*x + y + n + 1, n*x + y + n])
                tris.append([n*x + y, n*x + y + 1, n*x + y + n + 1])

    return np.array(surface), np.array(tris)

# verts, tris = create_test_surface(10,5)
# fig = ff.create_trisurf(verts[:,0], verts[:,1], verts[:,2], tris,"Portland")
# fig.update_layout(scene = {
#         "xaxis": {"range": [0,1]},
#         "yaxis": {"range": [0,1]},
#         "zaxis": {"range": [0,1]},
# })
# fig.show()

#field_0 = InscribedCircleField(normal, triangle)
#field_1 = InscribedCircleField(normal, triangle[::-1,:])
#field_2 = InscribedCircleField(normal, triangle @ np.array([[1,0,0],[0,1,0],[0,0,-1]]))
normal = np.array([0,0,1])

start = time()
verts, tris = create_test_surface(20,20,1,1, lambda x, y: 0.1 * np.random.rand() - 0.1*x) 
surface = np.zeros((200,200))
field = InscribedCircleField(normal, verts[tris])
for i in range(200):
    print(i)
    for j in range(200):
        # assert np.isclose(a,b) and np.isclose(a,c) and np.isclose(b,c)
        surface[j,i] = field.distance(np.array([i/200, j/200, 0]))# field_1.distance(np.array([i/200, j/200, 0])))

print(time()-start)
plt.imshow(surface)
plt.show()

test_dims = [3, 10, 100, 1000, 10000]


testing_tri_num = []
vertex_num = []
field_setup_time = []
generation_time = []
total_time = []
for dim in test_dims:

    gmsh.initialize()
    gmsh.model.add("distance_field_test")

    point_idx = []
    for row in points_plane_0:
        point_idx.append(gmsh.model.geo.add_point(row[0], row[1], row[2]))

    line_idx = []
    for i in range(4):
        line_idx.append(gmsh.model.geo.add_line(point_idx[i], point_idx[(i+1) % 4]))

    loop = gmsh.model.geo.add_curve_loop(line_idx)
    plane = gmsh.model.geo.add_plane_surface([loop])

    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


    triangle = np.roll(np.array([[0.35, 0.5, 0.1],[0.5, 0.5, 0.1],[0.35, 0.7, -0.4]]), 0, axis=0)
    verts, tris = create_test_surface(dim, dim)

    
    print("Contstraint triangle number: {}".format(tris.shape[0]))
    testing_tri_num.append(tris.shape[0])

    start_time = time()
    
    command_test_field = triangle_inscribed_circle_field(verts[tris], normal)


    # min_field = gmsh.model.mesh.field.add("Min")
    # gmsh.model.mesh.field.set_numbers(min_field, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(command_test_field)
    setup_time = time()

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.create_faces()
    end_time = time()

    print("Setup: {}, Generation: {}, total: {}".format(setup_time-start_time, end_time-setup_time, end_time-start_time))

    field_setup_time.append(setup_time-start_time)
    generation_time.append(end_time-setup_time)
    total_time.append(end_time-start_time)
    vertex_num
    node_ids, node_coords, *_ = gmsh.model.mesh.get_nodes()
    tri_ids, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)
    node_coords = node_coords.reshape(-1,3)
    print("vertex number: {}".format(node_coords.shape[0]))
    #tri_nodes = tri_nodes.reshape(-1,3) - 1
    gmsh.fltk.run()


# gmsh.model.mesh.field.add("ExternalProcess")
# gmsh.model.mesh.field.set_string(command_test_field, "CommandLine", "python " + "src/pebi_gmsh/utils_3D/inscribed_circle_field.py " + "0.1 0.001")#os.path.join(current, "src", "pebi_gmsh", "utils_3D", "inscribed_circle_field.py"))

# inscribed_circle = gmsh.model.mesh.field.add("MathEval")
# gmsh.model.mesh.field.set_string(inscribed_circle, "F",  "If(x > 0.5) 0.1 Else 0.01 EndIf")# point_distance([0.5,0.5,0.001], [0,0,1])) #
# fields.append(inscribed_circle)

# constant_max = gmsh.model.mesh.field.add("MathEval")
# gmsh.model.mesh.field.set_string(constant_max, "F", "0.01")
# fields.append(constant_max)