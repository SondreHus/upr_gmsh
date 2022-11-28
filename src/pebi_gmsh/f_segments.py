from dataclasses import dataclass
from pebi_gmsh.site_data import (
    Intersection,
)
from pebi_gmsh.site_lengths import get_site_lenghts
from typing import List
import numpy as np

@dataclass
class FSegment:
    start_site_idx: List[int] = None
    end_site_idx: List[int] = None
    
    start_edge_id: int = None
    end_edge_id: int = None
    
    vertices: any = None
    radiuses: any = None


def create_f_segments(intersections: List[Intersection], resolution, radius, interp, stop_dist) -> List[FSegment]:
    
    split = False
    current_dist = 0
    start_vertex = interp(0)
    start_radius = radius
    segments: List[FSegment] = []
    start_idx = None
    start_edge = None

    for intersection in intersections:
        
        if not split:
            vertices = start_vertex
            radiuses = start_radius
            vertex_distances = get_site_lenghts(current_dist, intersection.distance, resolution)
            vertices = np.vstack((start_vertex, interp(vertex_distances).T, intersection.end_vertex))
            radiuses = np.vstack((start_radius, np.ones((vertex_distances.shape[0],1))*radius, intersection.end_radius))
            segments.append(FSegment(
                start_site_idx = start_idx,
                end_site_idx = intersection.end_sites,
                start_edge_id = start_edge,
                end_edge_id = intersection.end_edge,
                vertices = vertices,
                radiuses = radiuses
            ))
            

        split = intersection.split
        current_dist = intersection.distance
        # Setting up the following start values
        start_radius = intersection.end_radius
        start_vertex = intersection.end_vertex
        start_idx = intersection.end_sites
        start_edge = intersection.end_edge

    
    vertex_distances = get_site_lenghts(current_dist, stop_dist, resolution, endpoint=True)
    vertices = np.vstack((start_vertex, interp(vertex_distances).T))
    radiuses = np.vstack((start_radius, np.ones((vertex_distances.shape[0],1))*radius))

    segments.append(FSegment(
                start_site_idx = start_idx,
                end_site_idx = None,
                start_edge_id = start_edge,
                end_edge_id = None,
                vertices = vertices,
                radiuses = radiuses
            ))

    return segments
