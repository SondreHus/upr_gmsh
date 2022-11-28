from pickletools import uint8
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def polyline_intersections(A, B = None, accept_perfect_match = False):
    """ 
        Finds intersections of 2 or one polyline

    Args:
        A (np.ndarray): array of the points in the polyline 
        B (np.ndarray): polyline to compare to

    Returns:
        np.ndarray: intersection points
        np.ndarray: index of the point on A the points should be inserted after
        np.ndarray: index of the point on B the points should be inserted after
    """

    if B is None:
        B = A
        self_intersect = True
    else:
        self_intersect = False
    # Line segment indexing
    i = np.arange(A.shape[0]-1, dtype=np.uintc)
    j = np.arange(B.shape[0]-1, dtype=np.uintc)
    
    ii, jj = np.meshgrid(i, j)

    ii = ii.flatten()
    jj = jj.flatten()

    if self_intersect:
        removed_rows = np.where((ii == jj) | (ii + 1 == jj ) | (ii - 1 == jj))
        ii = np.delete(ii.flatten(), removed_rows)
        jj = np.delete(jj.flatten(), removed_rows)
    

    A_x = np.sort(np.c_[A[i,0], A[i+1,0]], axis=1)
    A_y = np.sort(np.c_[A[i,1], A[i+1,1]], axis=1)
    B_x = np.sort(np.c_[B[j,0], B[j+1,0]], axis=1)
    B_y = np.sort(np.c_[B[j,1], B[j+1,1]], axis=1)
    
    rect = \
        np.less_equal(A_x[ii,0], B_x[jj,1]) & \
        np.less_equal(A_y[ii,0], B_y[jj,1]) & \
        np.less_equal(B_x[jj,0], A_x[ii,1]) & \
        np.less_equal(B_y[jj,0], A_y[ii,1])
    
    # Check for rectangle intersections

    ii = ii[np.where(rect)]
    jj = jj[np.where(rect)]

    A_d = A[ii+1]-A[ii]
    B_d = B[jj+1]-B[jj]
    a = scipy.sparse.csc_matrix(scipy.sparse.block_diag([*np.c_[A_d[:,0], -B_d[:,0], A_d[:,1], -B_d[:,1]].reshape((-1,2,2))]))
    b = np.c_[B[jj,0] - A[ii,0], B[jj,1] - A[ii,1]].reshape((-1,1))

    test = scipy.sparse.linalg.spsolve(a, b).reshape((-1,2))

    idx = np.where(np.all(np.greater(test, 0) & np.less(test, 1), axis = 1))

    points = A[ii[idx]] + A_d[idx]*test[idx,0].T
    return points, ii[idx], jj[idx]


if __name__ == "__main__":

    x = np.linspace(0,1,2)
    B = np.array([[0,0],[2,1]])#np.c_[0.5 + 0.5*x*np.cos(15*x),np.sin(50*x)]#np.c_[x,x]
    A = np.array([[1,-1],[1,0.51]])#np.c_[0.5 + 0.5*x*np.cos(15*x),np.sin(50*x)+0.1]
    a, i, j = polyline_intersections(A,B)
    plt.plot(A[:,0], A[:,1])
    plt.plot(B[:,0], B[:,1])
    plt.scatter(a[:,0], a[:,1])

    plt.show()
    print(a)