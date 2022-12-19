import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import (Voronoi, voronoi_plot_2d, Delaunay)
from pebi_gmsh.site_locations import create_site_locations
from pebi_gmsh.generate_constrained_mesh import generate_constrained_mesh_2d
from pebi_gmsh.site_data import (FConstraint, CConstraint)
from pebi_gmsh.clip_pebi import clip_voronoi
from time import time
from sympy import Point, Polygon, S
from numpy import arange
from numba import njit, prange

def second_moments(polygon: np.ndarray):
    """The second moments of area of the polygon.

    Returns
    =======

    Ix, Iy, Ixy : Second moments of area

    Examples
    ========

    >>> from sympy import Point, Polygon, S
    >>> from numpy import arange
    >>> p=[(cos(i),sin(i)) for i in arange(6)/S.One/3*pi]
    >>> poly = Polygon(*p)
    >>> second_moments(poly)
        (5*sqrt(3)/16, 5*sqrt(3)/16, 0)

    """
    c = polygon_centroid(polygon[:,0], polygon[:,1])
    if np.isnan(c).any():
        return 0 
    xc, yc = c[0], c[1]
    args = polygon

    args[:,0] -= xc
    args[:,1] -= yc
    x1, y1 = np.split(np.roll(args,-1,0),2, 1)
    x2, y2 = np.split(args, 2, 1)
    
    v = x1*y2 - x2*y1
    I = np.abs(np.sum(v*(y1*y1 + y1*y2 +y2*y2 + x1*x1 + x1*x2 + x2*x2), axis=0))/12

    # Ixy = np.sum(v*(x1*y2 + 2*x1*y1 + 2*x2*y2 + x2*y1),axis=0)
    # Ix /= 12
    # Iy /= 12
    # Ixy /= 24
    return I


def calculate_inertia(points: np.ndarray):
    if points.shape[0] < 3:
        return 0
    
    return second_moments(points)

def calculate_distance(points: np.ndarray, site):
    if points.shape[0] < 3:
        return 0
    c = polygon_centroid(points[:,0], points[:,1])
    return np.linalg.norm(c-site)

def polygon_area(xs, ys):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    # https://stackoverflow.com/a/30408825/7128154
    return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

def polygon_centroid(xs, ys):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    xy = np.array([xs, ys])
    area = polygon_area(xs, ys)
    if area == 0:
        return np.array([np.nan, np.nan])
    c = np.dot(xy + np.roll(xy, 1, axis=1),
               xs * np.roll(ys, 1) - np.roll(xs, 1) * ys
               ) / (6 * polygon_area(xs, ys))
    return c

def calculate_voronoi_energy(voronoi):

    sum = 0.0
    for i in range(len(voronoi.points)):
        region = voronoi.regions[voronoi.point_region[i]]
        if -1 in region:
            continue
        sum += calculate_distance(voronoi.vertices[region], voronoi.points[i])
        
        
    return sum
    
test = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=np.float64)
print(calculate_inertia(test))
print(calculate_inertia(test[::-1,:]))


x = np.linspace(0.1, 0.8, 100)
y = 0.4 + 0.2*x**2


line_1 = np.c_[x,y]
line_2 = np.array([
    [0.2, 0.2],
    [0.3, 0.8]
])

line_3 = np.array([
    [0.6, 0.9],
    [0.7, 0.2]
])




h_values = 10**-np.linspace(1,2.5, 15)

recalculate = True

#0.0023592161196014485

if recalculate:
    runtimes = np.zeros(15)
    vertex_num = np.zeros(15)
    energy = np.zeros(15)
    for i, h in enumerate(h_values):
        print("step : {}".format(i))
        print("resolution : {}".format(h))
        start_time = time()

        f_constraint_0 = FConstraint(line_1, h*0.2)
        f_constraint_1 = FConstraint(line_2, h*0.2)
        c_constraint = CConstraint(line_3, h*0.1, 1)

        site_data = create_site_locations(f_constraints=[f_constraint_0, f_constraint_1], c_constraints=[c_constraint])

        mesh = generate_constrained_mesh_2d(site_data=site_data, h0 = h)

        sites = mesh["node_coords"].reshape((-1,3))[:,:2]
        voronoi = Voronoi(sites)
        end_time = time()


        nodeSqrt = int(mesh["node_coords"].reshape((-1,3))[:,:2].shape[0]**0.5)
        steps = np.linspace(1/(2*nodeSqrt),1-1/(2*nodeSqrt),nodeSqrt)
        gridx, gridy = np.meshgrid(steps, steps)
        best_points = np.vstack((gridx.ravel(), gridy.ravel())).T
        worst_points = np.random.rand(mesh["node_coords"].reshape((-1,3))[:,:2].shape[0], 2)
        # comparrison_points = np.random.random((mesh["node_coords"].reshape((-1,3))[:,:2].shape[0], 2))
        best_voronoi = Voronoi(best_points)
        worst_voronoi = Voronoi(worst_points)
        clip_voronoi(best_voronoi)
        clip_voronoi(worst_voronoi)
        clip_voronoi(voronoi)
        
        vertices = np.array(voronoi.vertices)
        fig, ax = plt.subplots()
        voronoi_plot_2d(voronoi, ax = ax, show_vertices = False)
        centers = np.array([polygon_centroid(vertices[r][:,0], vertices[r][:,1]) for r in voronoi.regions])
        ax.scatter(centers[:,0], centers[:,1], marker="x", c="r")
        plt.show()
      

        best_vertices = np.array(best_voronoi.vertices)
        fig, ax = plt.subplots()
        voronoi_plot_2d(best_voronoi,ax = ax, show_vertices=False)
        centers = np.array([polygon_centroid(best_vertices[r][:,0], best_vertices[r][:,1]) for r in best_voronoi.regions])
        ax.scatter(centers[:,0], centers[:,1], marker="x", c="r")
        plt.show()

        worst_vertices = np.array(worst_voronoi.vertices)
        fig, ax = plt.subplots()
        voronoi_plot_2d(worst_voronoi,ax = ax, show_vertices=False)
        centers = np.array([polygon_centroid(worst_vertices[r][:,0], worst_vertices[r][:,1]) for r in worst_voronoi.regions])
        ax.scatter(centers[:,0], centers[:,1], marker="x", c="r")
        plt.show()

        runtimes[i] = end_time-start_time
        vertex_num[i] = mesh["node_coords"].reshape((-1,3))[:,:2].shape[0]

        energy[i] = calculate_voronoi_energy(voronoi)
        best_energy = calculate_voronoi_energy(best_voronoi)
        worst_energy = calculate_voronoi_energy(worst_voronoi)
        print("vertex number : {}".format(vertex_num[i]))
        print("Runtime : {}".format(runtimes[i]))
        print("Energy : {}".format(energy[i]))
        print("Best_Energy : {}".format(best_energy))
        print("Worst_Energy : {}".format(worst_energy))

    with open('runtime.npy', 'wb') as f:
        np.save(f, runtimes)
        np.save(f, vertices)
else:
    with open('runtime.npy', 'rb') as f:
        runtimes = np.load(f)
        vertices = np.load(f)
m, b = np.polyfit(np.log10(vertices), np.log10(runtimes), 1)

# matlab_runtimes = [
#     0.28128490000000,
#     0.37122640000000,
#     0.61624160000000,
#     0.86354780000000,
#     1.32977780000000,
#     2.13119020000000,
#     2.93442460000000,
#     4.70623400000000,
#     7.40815390000000,
#     11.8783662000000,
#     20.8619519000000,
#     36.3788368000000,
#     64.3708215000000,
#     115.839409700000,
#     209.413769600000,
# ]
matlab_runtimes= [
    2.24451260000000,
    2.83412380000000,
    3.78932130000000,
    4.93274780000000,
    6.89819930000000,
    9.62709970000000,
    13.8727035000000,
    16.9925385000000,
    25.7313073000000,
    35.5613537000000,
    49.1548024000000,
    69.8683425000000,
    101.617590000000,
    143.244652000000,
    206.789312900000,
]
matlab_nodes = [
    1860,
    2326,
    3038,
    3782,
    4789,
    6262,
    8008,
    10297,
    13353,
    17101,
    22196,
    28877,
    37068,
    47889,
    61958,
]
# matlab_nodes = [
#     273,
#     360,
#     499,
#     746,
#     1086,
#     1653,
#     2303,
#     3549,
#     5072,
#     7638,
#     11679,
#     17710,
#     26788,
#     40653,
#     62249,
# ]



print(m)

m_2, b_2 = np.polyfit(np.log10(matlab_nodes), np.log10(matlab_runtimes), 1)
plt.loglog(vertices, 10**b*vertices**m, c="C0",label= 'GMSH: O(n$^{' + '{:.2}'.format(m) + '}$)')
plt.loglog(matlab_nodes, 10**b_2*matlab_nodes**m_2, c="C1", label='UPR: O(n$^{' + '{:.2}'.format(m_2) + '}$)')

plt.scatter(vertices, runtimes, c="C0", marker="x")
plt.scatter(matlab_nodes, matlab_runtimes, c="C1", marker="x")
plt.grid("k--")
plt.xlabel("Number of vertices")
plt.ylabel("Runtime (s)")
plt.legend()
plt.savefig("Performance.png")
