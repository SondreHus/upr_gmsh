

import numpy as np
from pebi_gmsh.circle_intersections import circle_intersections
from pebi_gmsh.intersections_2d import polyline_intersections
from pebi_gmsh.f_segments import (FSegment, create_f_segments)
from pebi_gmsh.c_segments import (CSegment, create_c_segments, generate_protection_sites)
from pebi_gmsh.site_data import (SiteData, Intersection, FConstraint, CConstraint)
from dataclasses import dataclass
from typing import List
from pebi_gmsh.circumcircle import circumcircle

import matplotlib.pyplot as plt


def create_site_locations(f_constraints: List[FConstraint] = [], c_constraints: List[CConstraint] = []):
    

    data = SiteData(f_constraints, c_constraints)

    for i, c_constraint_0 in enumerate(c_constraints):
        for j, c_constraint_1 in enumerate(c_constraints[i:], i):
            points, ii, jj = polyline_intersections(c_constraint_0.points, c_constraint_1.points, self_intersect = (i==j))
            if points.shape[0] == 0:
                continue
            new_site_idx = data.add_sites(points)

            # Distances between previous point in path and the new intersection 
            dist_a = np.linalg.norm(points-c_constraint_0.points[ii], axis=1) + data.c_dist[i][ii]
            dist_b = np.linalg.norm(points-c_constraint_1.points[jj], axis=1) + data.c_dist[j][jj]
            for k in range(points.shape[0]):
                data.c_intersections[j].append(Intersection(dist_b[k], new_site_idx[k], False))
                data.c_intersections[i].append(Intersection(dist_a[k], new_site_idx[k], False))
    
    
    for i, f_constraint_0 in enumerate(f_constraints):
        for j, f_constraint_1 in enumerate(f_constraints[i:], i):
            points, ii, jj = polyline_intersections(f_constraint_0.points, f_constraint_1.points, self_intersect = (i==j))
            if points.shape[0] == 0:
                continue
            

            center_dist_0 = np.linalg.norm(points - f_constraint_0.points[ii], axis=1)
            center_dist_1 = np.linalg.norm(points - f_constraint_1.points[jj], axis=1)
            f_dir_0 = ((points - f_constraint_0.points[ii]).T / center_dist_0).T
            f_dir_1 = ((points - f_constraint_1.points[jj]).T / center_dist_1).T
            
            angles = np.arcsin(np.cross(f_dir_0, f_dir_1))

            aligned = np.where(np.sum(f_dir_0*f_dir_1, axis=1) > 0, 1, -1)

            res = (f_constraint_0.resolution + f_constraint_1.resolution)/2
            r = res / (2 ** 0.5)
            for k, point in enumerate(points):
                

                vector_i = point - data.f_interps[i](center_dist_0[k] + data.f_dist[i][ii[k]] - res)
                vector_i = vector_i*res/np.linalg.norm(vector_i)
                vector_j = point - data.f_interps[j](center_dist_1[k] + data.f_dist[j][jj[k]] - res)
                vector_j = vector_j*res/np.linalg.norm(vector_j)

                orientation = 1 if np.cross((vector_i), (vector_j)) > 0 else -1
                
                if np.abs(angles[k]) > np.pi*0.45:
                    mean =  (vector_i + vector_j)/2
                    mean =  r * mean / (np.linalg.norm(mean))
                    normal = np.array([-mean[1], mean[0]])

                    site_5 = point + normal * orientation
                    site_0 = point + mean
                    site_1 = point - normal * orientation
                    site_2 = point - mean
                    new_sites = (site_5, site_0, site_1, site_2)
                    new_site_idx = data.add_sites(np.vstack(new_sites))
                    new_edges = np.c_[new_site_idx, np.roll(new_site_idx, -1)].tolist() 
                    
                    if orientation == 1:
                        splits = [[0,3],[1,2], [3,2], [0,1]]
                        edge_idx = data.add_edges(np.array(
                            [[new_site_idx[split[0]], new_site_idx[split[1]]] for split in splits] + [[new_site_idx[0], new_site_idx[2]]]
                        ))
                        data.f_edge_loops += [[edge_idx[0], edge_idx[2], -edge_idx[1], -edge_idx[3]]]
                    else:
                        splits = [[3,0],[2,1],[2,3],[1,0]]
                        edge_idx = data.add_edges(np.array(
                            [[new_site_idx[split[0]], new_site_idx[split[1]]] for split in splits] + [[new_site_idx[0], new_site_idx[2]]]
                        ))
                        data.f_edge_loops += [[edge_idx[0], -edge_idx[3], -edge_idx[1], edge_idx[2]]]

                    data.f_intersections[i].append(Intersection(
                        distance    = center_dist_0[k] + data.f_dist[i][ii[k]] - res, 
                        end_sites   = (new_site_idx[splits[0][0]], new_site_idx[splits[0][1]]), 
                        split       = True,
                        end_vertex  = new_sites[splits[0][0]] + new_sites[splits[0][1]] - point,
                        end_radius  = r,
                        end_edge   = edge_idx[0]
                    ))
                    
                    data.f_intersections[i].append(Intersection(
                        distance    = center_dist_0[k] + data.f_dist[i][ii[k]] + res, 
                        end_sites   = (new_site_idx[splits[1][0]], new_site_idx[splits[1][1]]), 
                        split       = False,
                        end_vertex  = new_sites[splits[1][0]] + new_sites[splits[1][1]] - point,
                        end_radius  = r,
                        end_edge   = edge_idx[1]
                    ))
                    
                    data.f_intersections[j].append(Intersection(
                        distance    = center_dist_1[k] + data.f_dist[j][jj[k]] - res, 
                        end_sites   = (new_site_idx[splits[2][0]], new_site_idx[splits[2][1]]), 
                        split       = True,
                        end_vertex  = new_sites[splits[2][0]] + new_sites[splits[2][1]] - point,
                        end_radius  = r,
                        end_edge   = edge_idx[2]
                    ))

                    data.f_intersections[j].append(Intersection(
                        distance    = center_dist_1[k] + data.f_dist[j][jj[k]] + res, 
                        end_sites   = (new_site_idx[splits[3][0]], new_site_idx[splits[3][1]]), 
                        split       = False,
                        end_vertex  = new_sites[splits[3][0]] + new_sites[splits[3][1]] - point,
                        end_radius  = r,
                        end_edge   = edge_idx[3]    
                    ))

                else:
                    median_dir = int(np.sign(angles[k]))
                    vector_j = vector_j*aligned[k]
                    mean = vector_i/2 + vector_j/2
                    mean = mean/np.linalg.norm(mean)

                    res_0 = mean - vector_i*np.dot(vector_i, mean)/np.dot(vector_i, vector_i)
                    res_1 = mean - vector_j*np.dot(vector_j, mean)/np.dot(vector_j, vector_j)
                    
                    steps = int(np.ceil(np.abs(0.5/np.tan(angles[k]/2))))

                    step_radius = res/(2*steps*np.abs(np.tan(angles[k]/2)))

                    old_site_idx = None
                    new_sites = []
                    new_site_idx = []
                    
                    start_radius = res/(2*np.cos(angles[k]/2))
                    step_radii = np.linspace(start_radius, steps*step_radius, steps)
                    for radius in step_radii:
                        
                        site_0 = point - mean * radius + 2 * res_1 * radius
                        site_1 = point - mean * radius
                        site_2 = point - mean * radius + 2 * res_0 * radius
                        
                        site_3 = point + mean * radius - 2 * res_1 * radius
                        site_4 = point + mean * radius 
                        site_5 = point + mean * radius - 2 * res_0 * radius

                        new_sites = [site_0, site_1, site_2, site_3, site_4, site_5]
                        new_site_idx = data.add_sites(np.vstack((site_0, site_1, site_2, site_3, site_4, site_5)))

                        new_edges_f = data.add_edges(np.array([[new_site_idx[0], new_site_idx[1]], [new_site_idx[1], new_site_idx[2]]]))
                        new_edges_b = data.add_edges(np.array([[new_site_idx[3], new_site_idx[4]], [new_site_idx[4], new_site_idx[5]]]))
                        
                        if old_site_idx is not None:
                            data.add_edges(np.array([new_site_idx, old_site_idx]).T)
                            data.add_edges(np.array([[new_site_idx[0], old_site_idx[1]], [old_site_idx[1], new_site_idx[2]]]))
                            data.add_edges(np.array([[new_site_idx[3], old_site_idx[4]], [old_site_idx[4], new_site_idx[5]]]))
                        else:
                            data.add_edges(np.array([
                                [new_site_idx[0],new_site_idx[2]],
                                [new_site_idx[3],new_site_idx[5]], 
                                [new_site_idx[0],new_site_idx[3]],
                                [new_site_idx[2],new_site_idx[3]], 
                                [new_site_idx[5],new_site_idx[0]] 
                            ]))
                        old_site_idx = new_site_idx
                        



                    # new_site_idx = data.add_sites(np.vstack((site_5, site_0, site_1, site_2, site_3, site_4)))
                    # new_edges = np.c_[new_site_idx, np.roll(new_site_idx, -1)].tolist() + \
                    #     [[new_site_idx[0], new_site_idx[3]], [new_site_idx[0], new_site_idx[4]], [new_site_idx[1], new_site_idx[3]]]
                    
                    # edge_idx = data.add_edges(np.array(new_edges))
                    # data.f_edge_loops += [edge_idx[:6]]

                    # r = np.linalg.norm(site_1 + site_2 - steps*mean)
                    # end_r = np.linalg.norm(point - mean * steps +  res_0 * steps - new_sites[2])
                    
                    node_order = int(orientation*aligned[k])
                    intersection_index = 1 if node_order == 1 else 0
                    dist_to_vertex = np.linalg.norm(point - circle_intersections(new_sites[1], new_sites[2], r, r)[intersection_index][0])
                    data.f_intersections[i].append(Intersection(
                        distance    = center_dist_0[k] + data.f_dist[i][ii[k]] - dist_to_vertex, 
                        end_sites   = (new_site_idx[2], new_site_idx[1])[::node_order], 
                        split       = True,
                        end_vertex  = circle_intersections(new_sites[1], new_sites[2], r, r)[intersection_index][0],
                        end_radius  = r,
                        end_edge    = None
                    ))

                    data.f_intersections[i].append(Intersection(
                        distance    = center_dist_0[k] + data.f_dist[i][ii[k]] + dist_to_vertex, 
                        end_sites   = (new_site_idx[4], new_site_idx[5])[::node_order], 
                        split       = False,
                        end_vertex  = circle_intersections(new_sites[4], new_sites[5], r, r)[intersection_index][0],
                        end_radius  = r,
                        end_edge    = None
                    ))

                    data.f_intersections[j].append(Intersection(
                        distance    = center_dist_1[k] + data.f_dist[j][jj[k]] - dist_to_vertex*aligned[k], 
                        end_sites   = (new_site_idx[1], new_site_idx[0])[::orientation], 
                        split       = aligned[k] == 1,
                        end_vertex  = circle_intersections(new_sites[0], new_sites[1], r, r)[intersection_index][0],
                        end_radius  = r,
                        end_edge    = None
                    ))

                    data.f_intersections[j].append(Intersection(
                        distance    = center_dist_1[k] + data.f_dist[j][jj[k]] + dist_to_vertex*aligned[k], 
                        end_sites   = (new_site_idx[3], new_site_idx[4])[::orientation], 
                        split       = aligned[k] == -1,
                        end_vertex  = circle_intersections(new_sites[3], new_sites[4], r, r)[intersection_index][0],#point + mean * steps - res_1 * steps,
                        end_radius  = r,
                        end_edge    = None
                    ))
                    

                    

    for i, c_constraint in enumerate(c_constraints):
        for j, f_constraint in enumerate(f_constraints):
                points, ii, jj = polyline_intersections(c_constraint.points, f_constraint.points)
                if points.shape[0] == 0:
                    continue
                
                n = points.shape[0]
                dist_center = np.linalg.norm(points-c_constraint.points[ii], axis=1) + data.c_dist[i][ii]
                dist_b = np.linalg.norm(points-f_constraint.points[jj], axis=1) + data.f_dist[j][jj]

                f_vertex_pre = data.f_interps[j](dist_b - f_constraint.resolution/2).T
                f_vertex_post = data.f_interps[j](dist_b + f_constraint.resolution/2).T
                f_r = f_constraint.resolution*np.ones(n)

                c_vertex_pre = data.c_interps[i](dist_center-f_constraint.resolution/2).T
                
                angles = np.arcsin(np.cross(c_vertex_pre - points, f_vertex_pre - points) \
                    / (np.linalg.norm(c_vertex_pre - points, axis=1)*np.linalg.norm(f_vertex_pre - points, axis=1))) + np.pi/2
                
                c_dists = f_r*np.sqrt(1 + np.tan(angles)**2)


                c_sites_pre = data.add_sites(data.c_interps[i](dist_center - c_dists).T)
                c_sites_post = data.add_sites(data.c_interps[i](dist_center + c_dists).T)
                
              
                dist_offset = np.r_[dist_center-c_constraint.resolution/2, dist_center+c_constraint.resolution/2]
               
                for k in range(n):
                    # data.f_intersections[j].append(Intersection(
                    #     distance = dist_b[k] - f_resolutions[j]/2,
                    #     end_sites = edges[k][::-1],
                    #     end_edge = edge_idx[k],
                    #     end_vertex = f_vertex_pre[k],
                    #     split = True,
                    #     end_radius= f_r[k]
                    # ))
                    # data.f_intersections[j].append(Intersection(
                    #     distance = dist_b[k] + f_resolutions[j]/2,
                    #     end_sites = edges[k][::-1],
                    #     end_edge = edge_idx[k],
                    #     end_vertex = f_vertex_post[k],
                    #     split = False,
                    #     end_radius= f_r[k]
                    # ))

                    data.c_intersections[i].append(Intersection(
                        distance= dist_center[k] - c_dists[k],
                        end_sites = (c_sites_pre[k]),
                        split = True
                    ))
                    data.c_intersections[i].append(Intersection(
                        distance= dist_center[k] + c_dists[k],
                        end_sites = (c_sites_post[k]),
                        split = False
                    ))
    
                
    for i in range(len(f_constraints)):
        data.f_intersections[i].sort(key = lambda x: x.distance)
        segments = create_f_segments(data.f_intersections[i], f_constraints[i].resolution, 1, data.f_interps[i], data.f_dist[i][-1])
        for segment in segments:
            sites_l, sites_r = circle_intersections(segment.vertices[:-1,:], segment.vertices[1::,:], segment.radiuses[:-1,:], segment.radiuses[1:,:])

            site_idx_l = data.add_sites(sites_l)
            site_idx_r = data.add_sites(sites_r)[::-1]
            
            cross_sites_l = ([] if segment.start_site_idx is None else [segment.start_site_idx[1]]) + \
                site_idx_l + ([] if segment.end_site_idx is None else [segment.end_site_idx[1]])


            cross_sites_r = ([] if segment.start_site_idx is None else [segment.start_site_idx[0]]) + \
                site_idx_r[::-1] + ([] if segment.end_site_idx is None else [segment.end_site_idx[0]])


            data.add_edges(np.c_[site_idx_l, site_idx_r[::-1]])
            data.add_edges(np.c_[cross_sites_l[1::], cross_sites_r[:-1:]])

            data.add_edges(np.array(
                ([[site_idx_r[-1], site_idx_l[0]]] if segment.start_site_idx is None else [[segment.start_site_idx[1], site_idx_l[0]]]) +                 
                np.c_[site_idx_l[:-1:], site_idx_l[1::]].tolist() + 
                ([] if segment.end_site_idx is None else [[site_idx_l[-1], segment.end_site_idx[1]]])
                ))
            data.add_edges(np.array(
                ([[site_idx_l[-1], site_idx_r[0]]] if segment.end_site_idx is None else [[segment.end_site_idx[0], site_idx_r[0]]]) + 
                np.c_[site_idx_r[:-1:], site_idx_r[1::]].tolist()  +
                ([] if segment.start_site_idx is None else [[site_idx_r[-1], segment.start_site_idx[0]]])
                ))

    for i in range(len(c_constraints)):
        data.c_intersections[i].sort(key = lambda x: x.distance)
        segments = create_c_segments(data.c_intersections[i], c_constraints[i].resolution, data.c_interps[i], data.c_dist[i][-1])
        for segment in segments:

            new_sites =  data.add_sites(segment.sites)
                
            
            segment_site_idx = \
                ([] if segment.start_site_idx is None else [segment.start_site_idx]) + \
                new_sites +\
                ([] if segment.end_site_idx is None else [segment.end_site_idx])
            if len(segment_site_idx) > 1:
                data.add_edges(np.array([segment_site_idx[:-1], segment_site_idx[1:]]).T)
        
            protection_nodes_l, protection_nodes_r = generate_protection_sites(segment.sites, c_constraints[i].resolution, c_constraints[i].resolution, 1)
            
            old_sites_l = old_sites_r = new_sites
            for i in range(1):
                new_sites_l = data.add_sites(protection_nodes_l[i])
                data.add_edges(np.array([new_sites_l[:-1], new_sites_l[1:]]).T)
                data.add_edges(np.array([new_sites_l[:-1], old_sites_l[1:]]).T)
                data.add_edges(np.array([new_sites_l, old_sites_l]).T)
                old_sites_l = new_sites_l

                new_sites_r = data.add_sites(protection_nodes_r[i])
                data.add_edges(np.array([new_sites_r[:-1], new_sites_r[1:]]).T)
                data.add_edges(np.array([new_sites_r[:-1], old_sites_r[1:]]).T)
                data.add_edges(np.array([new_sites_r, old_sites_r]).T)
                old_sites_r = new_sites_r



    return data
