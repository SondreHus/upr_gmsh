import numpy as np

def circumcircle(a, b, c):
    a = a.reshape((-1,2))
    b = b.reshape((-1,2))
    c = c.reshape((-1,2))

    B = b-a
    C = c-a
    D = 2*np.cross(B,C)
    
    U = np.c_[
        (C[:,1]*np.linalg.norm(B, axis=1)**2 -B[:,1]*np.linalg.norm(C, axis=1)**2)/D,
        (B[:,0]*np.linalg.norm(C, axis=1)**2 -C[:,0]*np.linalg.norm(B, axis=1)**2)/D
    ]
    R = np.linalg.norm(U, axis=1)

    return U + a, R

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    points = np.random.rand(3*2,2)
    

    fig, ax_0 = plt.subplots()
    
    centroid, radius = circumcircle(points[0::3], points[1::3], points[2::3])
    ax_0.scatter(points[:, 0], points[:, 1])
    ax_0.scatter(centroid[:, 0], centroid[:, 1])

    for i, (c, r) in enumerate(zip(centroid, radius)):
        circle = plt.Circle(c,r, fill=False, linestyle="--") # type: ignore
        ax_0.add_patch(circle)
    plt.show()
