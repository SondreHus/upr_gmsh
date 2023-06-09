import numpy as np
from typing import List
from pebi_gmsh.utils_2D.polyline_interpolation import polyline_interpolation
from dataclasses import dataclass
from typing import (List, Any)

@dataclass
class FConstraint:
    points: Any = None   
    
    resolution: float = 0.1

    width_ratio: float = 1

@dataclass
class CConstraint:
    points: Any = None   
    
    resolution: float = 0.1

    protection_sites: int = 0
    


class SiteData:
    site_num = 0
    sites = np.zeros((0,2))
    
    def add_sites(self, new_sites: np.ndarray):
        idx = list(range(self.site_num, self.site_num+new_sites.shape[0]))
        self.sites = np.vstack((self.sites, new_sites))
        self.site_num = self.sites.shape[0]
        return idx

    edge_num = 0
    edges = np.zeros((0,2), dtype=int)

    def add_edges(self, new_edges: np.ndarray):
        if new_edges.dtype == np.float64:
            raise Exception("FAAAK")
        self.edges = np.vstack((self.edges,np.array(new_edges)))
        idx = list(range(self.edge_num, self.edges.shape[0]))
        self.edge_num = self.edges.shape[0]
        return idx
    
    def __init__(self, f_constraints: List[FConstraint], c_constraints: List[CConstraint]) -> None:
        

        self.c_constraints = c_constraints
        self.f_constraints = f_constraints

        # Distance along path of constraint vertices
        self.c_dist = []
        self.f_dist = []

        # Interpolation functions mapping length (along path) to coordinates
        self.c_interps = []
        self.f_interps = []
        for c_constraint in self.c_constraints:
            interp, dist = polyline_interpolation(c_constraint.points)
            self.c_dist.append(dist)
            self.c_interps.append(interp)
        
        for f_constraint in self.f_constraints:
            interp, dist = polyline_interpolation(f_constraint.points)
            self.f_dist.append(dist)
            self.f_interps.append(interp)

        self.c_intersections: List[List[Intersection]] = [[] for i in range(len(c_constraints))]
        self.f_intersections: List[List[Intersection]] = [[] for i in range(len(f_constraints))]
    
        self.f_edge_loops = []

@dataclass
class Intersection:
    """Per-constraint intersection dataclass
    """
    # Distance along the line the intersection occurs
    distance: float = 0

    # Nodes created at the intersection for connecting line segment nodes to the intersection nodes
    end_sites: Any = None

    # Whether to fill the following segment with nodes
    split: bool = False

    # End vertex used for calculating the end_nodes placement
    end_vertex: Any = None

    # End vertex radius used for the intersection
    end_radius: float = 0

    # Edge at the end of the node for creating loops
    end_edge: Any = None