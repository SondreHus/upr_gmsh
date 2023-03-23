from stl import mesh
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywavefront
from pebi_gmsh.triangulated_surface import TriangulatedSurface
from pebi_gmsh.sphere_intersection import sphere_intersections
from scipy.spatial import Voronoi
import gmsh
from pebi_gmsh.convert_GMSH import convert_GMSH





ico_sphere = mesh.Mesh.from_file("./ico.stl")

icosphere_obj = pywavefront.Wavefront("ico.obj", collect_faces=True)#pd.read_csv("ico.obj", header=None, delimiter=' ')
verts = np.array(icosphere_obj.vertices)
tris = np.array(icosphere_obj.meshes["Icosphere"].faces)

surface = TriangulatedSurface(verts, tris)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xs = surface.vertices[surface.edges[:,:],0].reshape((-1,2))
ys = surface.vertices[surface.edges[:,:],1].reshape((-1,2))
zs = surface.vertices[surface.edges[:,:],2].reshape((-1,2))
for i in range(120):
    ax.plot(xs[i], ys[i], zs[i], color="C0")

outer, inner = sphere_intersections(surface.vertices[surface.triangle_vertices], 1/5*np.ones((surface.triangle_vertices.shape[0], 3)))

ax.scatter(outer[:,0], outer[:,1], outer[:,2], color="C3")
#ax.scatter(inner[:,0], inner[:,1], inner[:,2])

voronoi = Voronoi(np.vstack((inner,outer)))

# ax.scatter(voronoi.vertices[:,0], voronoi.vertices[:,1], voronoi.vertices[:,2], color="C2")
plt.show()



gmsh.initialize()
gmsh.initialize()
gmsh.model.add("MRST")
# gmsh.option.setNumber("Mesh.Algorithm", algorithm)
# gmsh.option.setNumber('General.Terminal', 0)
# gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
# gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
# gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Voronoi", 1)


gmsh.model.geo.addPoint(-1.5,-1.5,-1.5, 0.8, 1)
gmsh.model.geo.addPoint(-1.5,-1.5, 1.5, 0.8, 2)
gmsh.model.geo.addPoint(-1.5, 1.5, 1.5, 0.8, 3)
gmsh.model.geo.addPoint(-1.5, 1.5,-1.5, 0.8, 4)

gmsh.model.geo.addPoint(1.5,-1.5,-1.5, 0.8, 5)
gmsh.model.geo.addPoint(1.5, 1.5,-1.5, 0.8, 6)
gmsh.model.geo.addPoint(1.5, 1.5, 1.5, 0.8, 7)
gmsh.model.geo.addPoint(1.5, -1.5, 1.5, 0.8, 8)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 5, 8)

gmsh.model.geo.addLine(1, 5, 9)
gmsh.model.geo.addLine(2, 8, 10)
gmsh.model.geo.addLine(3, 7, 11)
gmsh.model.geo.addLine(4, 6, 12)

gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
gmsh.model.geo.addCurveLoop([5,6,7,8], 2)
gmsh.model.geo.addCurveLoop([1,10,8,-9], 3)
gmsh.model.geo.addCurveLoop([-3,11,-6,-12], 4)
gmsh.model.geo.addCurveLoop([10,-7,-11,-2], 5)
gmsh.model.geo.addCurveLoop([-4,12,-5,-9], 6)

gmsh.model.geo.addPlaneSurface([1],1)
gmsh.model.geo.addPlaneSurface([2],2)
gmsh.model.geo.addPlaneSurface([3],3)
gmsh.model.geo.addPlaneSurface([4],4)
gmsh.model.geo.addPlaneSurface([5],5)
gmsh.model.geo.addPlaneSurface([6],6)

gmsh.model.geo.addSurfaceLoop([1,2,3,4,5,6],1)

gmsh.model.geo.add_volume([1], 1)


points = []
for point in np.vstack((inner, outer)):
    points.append(gmsh.model.geo.addPoint(point[0], point[1], point[2], 0.5))

gmsh.model.geo.synchronize()
gmsh.model.mesh.embed(0, points, 3, 1)

gmsh.model.mesh.generate(3)


# gmsh.model.mesh.create_faces()
#     
gmsh.fltk.run()

mesh_dict = convert_GMSH()
vertices = mesh_dict["node_coords"].reshape(-1,3)
vertex_list = []
for vertex in vertices:
    outside = True
    for v in verts:
        if np.sum((vertex-v)**2) < 1/26:
            outside = False
    if outside:
        vertex_list.append(vertex)
vertices = np.array(vertex_list)
voronoi = Voronoi(vertices)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plot_voronoi_3d(voronoi, ax, points)
plt.show()
print("Hi")
