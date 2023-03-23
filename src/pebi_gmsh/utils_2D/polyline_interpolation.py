import numpy as np
from scipy.interpolate import interp1d

def polyline_interpolation(line: np.ndarray):
    
    dist = np.zeros(line.shape[0])
    dist[1:] = np.linalg.norm(line[:-1:,:] - line[1::,:], axis=1)
    dist = np.array([np.sum(dist[:i+1]) for i in range(line.shape[0])])
    interp = interp1d(dist, line.T, kind="linear", fill_value="extrapolate")
    
    return interp, dist