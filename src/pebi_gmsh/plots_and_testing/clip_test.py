import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d




points = np.zeros((10*9,2))

for x in range(10):
    for y in range(9):
        points[x*9 + y] = np.array([
            x*0.1 + 0.05*y%2,
            y*0.3
        ])


# sites = np.vstack((np.meshgrid(x,y)[0].flatten(), np.meshgrid(x,y)[1].flatten())).T

voronoi = Voronoi(points)
fig, ax = plt.subplots()
voronoi_plot_2d(voronoi,ax = ax, show_vertices=False)
ax.set_xlim(0.15,0.85)
ax.set_ylim(.2,1)
ax.axis('off')
plt.savefig("kinda_rectangular.png", bbox_inches='tight', pad_inches=0, dpi=1000)
# plt.show()
# sites = np.random.rand(6,2)* np.array([1.5,0.3]) + np.array([0,0.5])

# line = np.array([[0, 0.5], [1.5, 0.5]])

# voronoi = Voronoi(sites)

# fig, ax = plt.subplots()
# voronoi_plot_2d(voronoi, ax=ax, show_points=False, show_vertices=False)
# ax.scatter(sites[:,0], sites[:,1])
# ax.set_xlim(0,1.5)
# ax.set_ylim(0,1)

# ax.plot(line[:,0], line[:,1], "r--")
# ax.axis('off')
# # plt.show()
# plt.savefig("mirror_border.png", bbox_inches='tight', pad_inches=0, dpi=1000)
# mirror = []
# for i in range(sites.shape[0]):
#     mir = False

#         if vert == -1:
#             mir = True
#         else:
#             if voronoi.vertices[vert, 1] < 0.5:
#                 mir = True 
#     if mir:
#         mirror.append(i)

# mirror = np.array(mirror).astype(int)       
# new_sites = sites[mirror]*np.array([1,-1]) + np.array([0, 1])

# voronoi = Voronoi(np.vstack((sites,new_sites)))

# fig, ax = plt.subplots()
# voronoi_plot_2d(voronoi, ax=ax, show_points=False, show_vertices=False)
# ax.scatter(sites[:,0], sites[:,1])
# ax.scatter(new_sites[:,0], new_sites[:,1])
# ax.set_xlim(0,1.5)
# ax.set_ylim(0,1)
# ax.axis('off')
# # ax.plot(line[:,0], line[:,1], "r--")
# # plt.show()
# plt.savefig("mirrored.png", bbox_inches='tight', pad_inches=0, dpi=1000)


# import gmsh
# from pebi_gmsh.generate_constrained_mesh import generate_constrained_mesh_2d
# import matplotlib.pyplot as plt
# from pebi_gmsh.convert_GMSH import convert_GMSH
# import numpy as np
# from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d
# from pebi_gmsh.site_locations import create_site_locations
# from pebi_gmsh.site_data import (FConstraint, CConstraint)
# from matplotlib.collections import LineCollection
# from pebi_gmsh.clip_pebi import clip_pebi

# # Background min resolution
# h = 0.2

# x = np.linspace(0.1, 0.8, 100)
# y = 0.4 + 0.4*x**2

# # Constraint curves
# line_1 = np.c_[x,y]
# line_2 = np.array([[0.2, 0.2],[0.3, 0.8]])
# line_3 = np.array([[0.6, 0.9], [0.7, 0.2]])

# # Constraints
# f_1 = FConstraint(line_1, 0.08)
# f_2 = FConstraint(line_2, 0.08)
# c_1 = CConstraint(line_3, 0.04)

# # Generating constrained sites
# site_data = create_site_locations([f_1, f_2], [c_1])

# mesh = generate_constrained_mesh_2d(site_data, h, np.array([[-.1,0], [1.1,1]]))

# fig, ax = plt.subplots()
# ax.plot(line_1[:,0], line_1[:,1], "r--")
# ax.plot(line_2[:,0], line_2[:,1], "r--")
# ax.plot(line_3[:,0], line_3[:,1], "b--")
# ax.axis('off')
# ax.set_ylim((0.05,0.95))
# ax.set_xlim((0,1))

# plt.savefig("demo_grid_0", bbox_inches='tight', pad_inches=0, dpi=600)

# fig, ax = plt.subplots()
# ax.plot(line_1[:,0], line_1[:,1], "r--")
# ax.plot(line_2[:,0], line_2[:,1], "r--")
# ax.plot(line_3[:,0], line_3[:,1], "b--")
# ax.axis('off')
# ax.set_ylim((0.05,0.95))
# ax.set_xlim((0,1))

# ax.scatter(site_data.sites[:,0], site_data.sites[:,1], marker=".",c="C7")
# xx = np.vstack((site_data.sites[site_data.edges[:,0]][:,0], site_data.sites[site_data.edges[:,1]][:,0]))
# yy = np.vstack((site_data.sites[site_data.edges[:,0]][:,1], site_data.sites[site_data.edges[:,1]][:,1]))
# plt.plot(xx,yy, c="C0")

# # plt.grid()
# plt.savefig("demo_grid_1", bbox_inches='tight', pad_inches=0, dpi=600)

# fig, ax = plt.subplots()
# ax.plot(line_1[:,0], line_1[:,1], "r--")
# ax.plot(line_2[:,0], line_2[:,1], "r--")
# ax.plot(line_3[:,0], line_3[:,1], "b--")
# ax.axis('off')
# ax.set_ylim((0,1))
# ax.set_xlim((-.1,1.1))

# tris = Delaunay(mesh["node_coords"].reshape((-1,3))[:,:2])
# sites = mesh["node_coords"].reshape((-1,3))[:,:2]
# ax.triplot(sites[:,0], sites[:,1], tris.simplices)

# plt.savefig("demo_grid_2", bbox_inches='tight', pad_inches=0, dpi=600)
# fig, ax = plt.subplots()
# ax.plot(line_1[:,0], line_1[:,1], "r--")
# ax.plot(line_2[:,0], line_2[:,1], "r--")
# ax.plot(line_3[:,0], line_3[:,1], "b--")
# ax.axis('off')
# ax.set_ylim((0,1))
# ax.set_xlim((-.1,1.1))

# voronoi = Voronoi(sites)
# voronoi_plot_2d(voronoi, ax=ax, show_vertices=False)
# plt.savefig("demo_grid_3", bbox_inches='tight', pad_inches=0, dpi=600)

# fig, ax = plt.subplots()
# ax.plot(line_1[:,0], line_1[:,1], "r--")
# ax.plot(line_2[:,0], line_2[:,1], "r--")
# ax.plot(line_3[:,0], line_3[:,1], "b--")
# ax.axis('off')
# ax.set_ylim((0,1))
# ax.set_xlim((-.1,1.1))

# clip_pebi(voronoi, boundary = np.array([[-.1,0],[-.1,1],[1.1,1],[1.1,0]])[::-1])
# voronoi_plot_2d(voronoi, ax=ax, show_vertices=False)
# plt.savefig("demo_grid_4", bbox_inches='tight', pad_inches=0, dpi=600)