from dataclasses import dataclass
from pebi_gmsh.site_data import (
    Intersection,
)
from pebi_gmsh.site_lengths import get_site_lenghts
from pebi_gmsh.circle_intersections import circle_intersections
from typing import List
import numpy as np


@dataclass
class CSegment:
    start_site_idx: int = None
    end_site_idx: int = None
    sites: any = None



def create_c_segments(intersections: List[Intersection], resolution, interp, stop_dist) -> List[CSegment]:
    
    split = False
    current_dist = 0
    segments: List[CSegment] = []
    start_idx = None
    start_node = interp

    for intersection in intersections:
        
        if not split:
            site_distances = get_site_lenghts(current_dist, intersection.distance, resolution, startpoint = (start_idx is None))
            sites = interp(site_distances).T
            segments.append(CSegment(
                start_site_idx = start_idx,
                end_site_idx = intersection.end_sites,
                sites = sites
            ))
            

        split = intersection.split
        current_dist = intersection.distance
        # Setting up the following start values         
        start_idx = intersection.end_sites

    
    site_distances = get_site_lenghts(current_dist, stop_dist, resolution, endpoint = True)
    sites = interp(site_distances).T

    segments.append(CSegment(
                start_site_idx = start_idx,
                end_site_idx = None,
                sites = sites
    ))

    return segments

def pad_polyline(points):
    point_before = 2*points[0]-points[1]
    point_after = 2*points[-1]-points[-2]
    return np.vstack((point_before, points, point_after))

def generate_protection_sites(points: np.ndarray, res, width, number):
    
    radius = np.sqrt(res**2 + width**2)/2

   
    points = pad_polyline(points)
    vertices_l, vertices_r = circle_intersections(points[:-1], points[1:], radius, radius)
    
    nodelayers_l = []
    nodelayers_r = []
    
    for i in range(number):
        
        
        nodes_l = circle_intersections(vertices_l[:-1], vertices_l[1:], radius, radius)[0]
        nodes_r = circle_intersections(vertices_r[:-1], vertices_r[1:], radius, radius)[1]
        
        nodelayers_l.append(nodes_l)
        nodelayers_r.append(nodes_r)

        if i < number - 1:
            vertices_l = pad_polyline(circle_intersections(nodes_l[:-1], nodes_l[1:], radius, radius)[0])
            vertices_l = pad_polyline(circle_intersections(nodes_l[:-1], nodes_l[1:], radius, radius)[1])

    return nodelayers_l, nodelayers_r

