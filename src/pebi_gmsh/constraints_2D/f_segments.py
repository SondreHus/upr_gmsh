from dataclasses import dataclass
from pebi_gmsh.constraints_2D.site_data import (
    Intersection,
)
from pebi_gmsh.utils_2D.site_lengths import get_site_lenghts
from typing import List, Optional
import numpy as np
@dataclass
class FSegment:
    start_site_idx: Optional[List[int]] = None
    end_site_idx: Optional[List[int]] = None
    
    start_edge_id: Optional[int] = None
    end_edge_id: Optional[int] = None
    
    vertices: np.ndarray = np.zeros((0,2))
    radiuses: np.ndarray = np.zeros((0,1))

def get_vertex_radius(width, resolution):
    return np.sqrt((width/2)**2 + (resolution/2)**2)

def create_f_segments(intersections: List[Intersection], resolution, relative_width, interp, stop_dist) -> List[FSegment]:
    
    split = False
    current_dist = 0
    start_vertex = interp(0)
    start_radius = get_vertex_radius(relative_width*resolution, resolution)
    segments: List[FSegment] = []
    start_idx = None
    start_edge = None

    for intersection in intersections:
        
        if not split:
            vertices = start_vertex
            
            radiuses = start_radius
            vertex_distances = get_site_lenghts(current_dist, intersection.distance, resolution)
            
            if(vertex_distances.shape[0] > 1):
                vertex_spacing = vertex_distances[1]-vertex_distances[0]
                vertex_radius = get_vertex_radius(relative_width*resolution, vertex_spacing)
            else:
                vertex_radius = get_vertex_radius(relative_width*resolution, resolution)
            vertices = np.vstack((start_vertex, interp(vertex_distances).T, intersection.end_vertex))
            radiuses = np.vstack((start_radius, np.ones((vertex_distances.shape[0],1))*vertex_radius, intersection.end_radius))
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
    if(vertex_distances.shape[0] > 1):
        vertex_spacing = vertex_distances[1]-vertex_distances[0]
        vertex_radius = get_vertex_radius(relative_width*resolution, vertex_spacing)
    else:
        vertex_radius = get_vertex_radius(relative_width*resolution, resolution)
    vertices = np.vstack((start_vertex, interp(vertex_distances).T))
    radiuses = np.vstack((start_radius, np.ones((vertex_distances.shape[0],1))*vertex_radius))

    segments.append(FSegment(
                start_site_idx = start_idx,
                end_site_idx = None,
                start_edge_id = start_edge,
                end_edge_id = None,
                vertices = vertices,
                radiuses = radiuses
            ))

    return segments
