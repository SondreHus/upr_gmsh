import numpy as np
def mean_unit_vector(p0, p1, p2):
    mean = (p1 - p0)/np.linalg.norm(p1-p0, axis=1) + (p1 - p0)/np.linalg.norm(p1-p0, axis=1)
    mean /= np.linalg.norm(mean, axis=1)
    return mean