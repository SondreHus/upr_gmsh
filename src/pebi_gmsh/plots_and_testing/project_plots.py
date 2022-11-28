import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import (Voronoi, voronoi_plot_2d, Delaunay)
from pebi_gmsh.site_locations import create_site_locations
from pebi_gmsh.generate_constrained_mesh import generate_constrained_mesh_2d
from pebi_gmsh.site_data import (FConstraint, CConstraint)

np.random.seed(1948)

def get_circumcenters(points, tris):
    for tri in tris:
        pass    

    return
# Voronoi demonstration
fig, (ax_0, ax_1) = plt.subplots(1,2)
points = np.array([
    [0.05, 0.8],
    [0.4, 0.6],
    [0.7, 0.2],
    [0.1, 0.4],
    [0.6, 0.8]])
tris = Delaunay(points)
voronoi = Voronoi(points)
centroids = voronoi.vertices
radii = np.linalg.norm(centroids -  points[tris.simplices[:,0]], axis=1)
ax_0.triplot(points[:,0], points[:,1], tris.simplices, c="r")
for centroid, radius in zip(centroids, radii):
    circle = plt.Circle(centroid,radius, fill=False, linestyle="--", color="C0")
    ax_0.add_patch(circle)
ax_0.scatter(centroids[:,0], centroids[:,1], marker="x", color="C0")
ax_0.scatter(points[:,0], points[:,1],c="C7", zorder=10)
ax_0.axis("Equal")
ax_0.set_ylim(0,1)
ax_0.set_xlim(0,1)


voronoi_plot_2d(voronoi,ax_1,show_points = False, show_vertices = False, line_colors="C0")
ax_1.scatter(centroids[:,0], centroids[:,1], marker="x", color="C0")
ax_1.scatter(points[:,0], points[:,1],c="C7", zorder=10)
ax_1.triplot(points[:,0], points[:,1], tris.simplices, linestyle=":", c="r")
ax_1.axis("Equal")
ax_1.set_ylim(0,1)
ax_1.set_xlim(0,1)
plt.show()


# Face constraint test


x = np.linspace(0.1,0.9,30)
y = np.sqrt(np.abs(np.sin(x*2)))

plt.plot