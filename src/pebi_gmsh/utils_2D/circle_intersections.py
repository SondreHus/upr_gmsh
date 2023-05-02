from textwrap import fill
import numpy as np
import matplotlib.pyplot as plt
normal_matrix = np.array([[0,-1],[1,0]])


def circle_intersections(p_0, p_1, r_0, r_1, accept_nan = False):
    """Returns the intersection points of two sets of circles.

    Args:
        p_0 (_type_): First circle set centroids
        p_1 (_type_): Second circle set centroids
        r_0 (_type_): First circle set radii
        r_1 (_type_): Second circle set radii
        accept_nan (bool, optional): If False, raises error if any intersection is impossible. Defaults to False.

    Raises:
        Exception: _description_

    Returns:
        Two sets of points, the first on the right-hand side of the vector p_1-p_0, the other set is on the opposite side.
    """
    p_0 = p_0.reshape((-1,2))
    p_1 = p_1.reshape((-1,2))

    if isinstance(r_0, float):
        r_0 = np.ones(p_0.shape[0]) * r_0
    if isinstance(r_1, float):
        r_1 = np.ones(p_1.shape[0]) * r_0

    r_0 = r_0.reshape((-1,1))
    r_1 = r_1.reshape((-1,1))
    deltas = p_1-p_0
    dists = np.linalg.norm(deltas, axis=1).reshape((-1,1))

    x = ((dists**2 + r_0**2 - r_1**2))/(2*dists)
    y = np.sqrt(r_0**2 - x**2).reshape((-1,1))
    x_0 = deltas*x/dists
    normals = (deltas/dists)@normal_matrix
    

    if (not accept_nan) and any(np.isnan(y)):
        raise Exception("No intersection found")

    return p_0 + x_0 + normals*y , p_0 + x_0 - normals*y

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0,5,10)
    y = (0.5*x)**2
    p = np.c_[x,y]
    r = 0.66*np.linalg.norm(p[1::]-p[:-1:], axis=1)

    lh, rh = circle_intersections(p[:-1:], p[1::], r, r)
    ax = plt.gca()
    ax.plot(
        p[:,0], p[:, 1],
        lh[:,0], lh[:, 1],
        rh[:,0], rh[:,1],
    )
    for i, r_i in enumerate(r):
        circle_1 = plt.Circle((p[i,0], p[i,1]), r_i, color="b", fill=False) # type: ignore
        circle_2 = plt.Circle((p[i+1,0], p[i+1,1]), r_i, color="r", fill=False)  # type: ignore

        ax.add_patch(circle_1)
        ax.add_patch(circle_2)

    # plt.ylim((0,5))
    ax.scatter(
        lh[:,0], lh[:, 1]
    )
    ax.scatter(
        rh[:,0], rh[:,1]
    )
    plt.show()