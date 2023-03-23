import numpy as np

def get_site_lenghts(len_1, len_2, res, startpoint = False, endpoint= False):
    
    num_sites = int((len_2-len_1)//res)

    if num_sites < 0:
        return np.zeros(0)
    
    site_distances = np.linspace(len_1, len_2, num_sites+2)

    if not endpoint:
        site_distances = site_distances[:-1]
    if not startpoint:
        site_distances = site_distances[1:]

    return site_distances