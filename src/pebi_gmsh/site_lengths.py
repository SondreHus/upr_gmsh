import numpy as np

def get_site_lenghts(len_1, len_2, res, startpoint = False, endpoint= False):
    
    num_sites = round((len_2-len_1)/res) 
    if startpoint:
        len_start = len_1 
    else:
        len_start = (len_2-len_1)/(num_sites+1) + len_1

    if num_sites < 0:
        return np.zeros(0)
    
    site_distances = np.linspace(len_start, len_2, num_sites, endpoint=endpoint, )
    return site_distances