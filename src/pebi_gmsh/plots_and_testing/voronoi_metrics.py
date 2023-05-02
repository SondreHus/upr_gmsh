from plotly import figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt

points = np.array([
    [0.90383448, 0.0841829 ],
    [0.42717591, 0.43612741],
    [0.16015486, 0.05666762],
    [0.62951053, 0.76198861],
    [0.84007348, 0.94489851],
    [0.13082893, 0.7370706 ],
    [0.48813197, 0.74398238],
    [0.78092733, 0.55482526],
    [0.49512409, 0.02098421],
    [0.55420788, 0.11353278]
])

def plot_metric(m,n, metric = lambda x,y, xt, yt: np.sqrt(np.square(x-xt) + np.square(y-yt)), tol = 1e-2, name = "Voronoi"):
    grid = np.zeros((points.shape[0],m,n))
    grid_2 = np.zeros((m,n))
    grid_3 = np.zeros((m,n))
    for x in range(m):
        for y in range(n):
            dist = np.inf
            border = False
            for i, point in enumerate(points):
                val = metric((x+0.5)/m, (y+0.5)/n, point[0], point[1])
                if val < dist:
                    grid_3[x,y] = i
                    if dist - val < tol:
                        border = True
                    else:
                        border = False
                    dist = val
                grid[i,x,y] = val
            if border:
                grid_2[x,y] = 1
    
    plt.imshow(grid_3)
    plt.show()
    plt.imshow(np.min(grid, axis=0))
    plt.savefig(name + "dist.png")
    plt.imshow(grid_2)
    plt.savefig(name + "borders.png")
      
plot_metric(300,300, name="euclidean")
plot_metric(300,300, lambda x,y, xt, yt: (np.square(x-xt) + np.square(y-yt)), name="euclidean")
plot_metric(300,300, lambda x,y, xt, yt: np.abs(x-xt) + np.abs(y-yt), name="taxicab")
plot_metric(300,300, lambda x,y, xt, yt: max(np.abs(x-xt),np.abs(y-yt)), name="max")


fig, ax = plt.subplots()
ax.scatter(points[:,0], points[:,1])
ax.set_xlim((0,1))
ax.set_ylim((0,1))
ax.set_aspect("equal", "box")
fig.tight_layout()
plt.savefig("points.png")
print("hi")