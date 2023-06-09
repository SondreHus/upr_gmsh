import numpy as np
from scipy.spatial import Voronoi
from pebi_gmsh.utils_3D.sphere_intersection import (sphere_intersections, flatten_sphere_centers)
from pebi_gmsh.utils_2D.circle_intersections import circle_intersections, circle_intersection_height
from pebi_gmsh.utils_2D.circumcircle import circumcircle
from pebi_gmsh.utils_3D.line_sphere_intersection import line_sphere_intersection, sphere_line_reduction
from functools import reduce


def test_triangle_intersection(triangle_coords, triangle_radii):
 
    inner, _ = sphere_intersections(triangle_coords, triangle_radii)

    return ~np.any(np.isnan(inner), axis=1)

def minimum_intersecting_radii(flattened_triangles, triangle_radii, accept_nan = False):
    comp = np.array([[1,2],[2,0],[0,1]])
    _, inner = circle_intersections(
        flattened_triangles[...,comp[:,0], :-1], 
        flattened_triangles[...,comp[:,1], :-1], 
        triangle_radii[...,comp[:,0]], 
        triangle_radii[...,comp[:,1]],
        accept_nan=accept_nan
    )

    return np.sqrt(np.sum((inner - flattened_triangles[:, :-1].reshape(-1,2))**2, axis=1)).reshape(triangle_radii.shape)





class TriangulatedSurface:

    # Coordinates of vertices
    vert_coords: np.ndarray
    
    # Idx of every triangle a vertex is a part of 
    vert_tris: list

    # Radii of the vertex spheres
    vertex_radii: np.ndarray

    # Idx of each vertex every triangle consists of
    tri_verts: np.ndarray

    # Idx of each edge every triangle consists of
    tri_edges: np.ndarray
    
    # Idx of vertices each edge consists of 
    edges: np.ndarray

    # Dict from vertex id pair -> edge id
    edge_dict: dict
    
    # Triangles adjacent to each edge
    edge_tris: list
    
    def __init__(self, vertices = None, triangles: np.ndarray = None, radii: np.ndarray = None, constricted_radii: np.ndarray = None) -> None:
        self.vert_coords = vertices
        self.tri_verts = triangles
        self.tri_edges = np.zeros(triangles.shape)
        self.edge_dict = {}
        self.vert_tris = [[] for i in range(vertices.shape[0])]
        self.vertex_radii = (np.zeros(vertices.shape[0])) if radii is None else radii
        self.constricted_radii = np.zeros(vertices.shape[0], dtype=bool) if constricted_radii is None else constricted_radii
        self.edge_tris = []
        
        edges = []
        for comp_vertex_id, triangle in enumerate(triangles):
            
            tri_edges = np.zeros(3)
            for j in range(3):
                self.vert_tris[triangle[j]].append(comp_vertex_id)
                if (edge_vertex_idx := tuple(sorted((triangle[j], triangle[(j+1)%3])))) not in self.edge_dict:
                    # Add new edge and assign it to the edge dictionary
                    tri_edges[j] = self.edge_dict[edge_vertex_idx] = len(edges)
                    edges.append(edge_vertex_idx)
                    self.edge_tris.append([comp_vertex_id])
                else:
                    tri_edges[j] = self.edge_dict[edge_vertex_idx]
                    # Update edge_triangles with new triangle
                    self.edge_tris[self.edge_dict[edge_vertex_idx]].append(comp_vertex_id)
            
            self.tri_edges[comp_vertex_id] = tri_edges
           
        self.edges = np.array(edges)
        self.vert_tris = [np.array(v) for v in self.vert_tris]
        
        # Naive vertex radii
        # current_radius = np.zeros(vertices.shape[0])
        current_radius = self.vertex_radii #np.zeros(self.vertex_radii.shape)

        flattened_triangles, *_ = flatten_sphere_centers(vertices[triangles])

        vert_dists = np.zeros(vertices.shape[0])
        vert_num = np.zeros(vertices.shape[0])
        for tri in triangles:
            dists = np.sqrt(np.sum((self.vert_coords[tri] - self.vert_coords[np.roll(tri,-1)])**2, axis=1))
            dists = (dists + np.roll(dists,1))/2
            vert_dists[tri] += dists
            vert_num[tri] += 1

        mean_vert_dist = vert_dists/vert_num
        current_radius = np.where(self.constricted_radii, self.vertex_radii, mean_vert_dist*5/6)
        # Maximum approach

        # edge_dists = np.sqrt(np.sum((self.vert_coords[self.edges[:,0]] - self.vert_coords[self.edges[:,1]])**2, axis=-1))
        # for non_padding_tri in triangles[~self.padding_tri]:
            
        #     # if not test_triangle_intersection(self.vert_coords[non_padding_tri], current_radius[non_padding_tri]):
            
        #     dists = np.sqrt(np.sum((self.vert_coords[non_padding_tri] - self.vert_coords[np.roll(non_padding_tri,-1)])**2, axis=1))
        #     # Vertex set to max of edges of connected triangles
        #     # max_dist = np.max(dists)
        #         # vertex radius set to max of connected edges
        #     max_dist = np.maximum(dists, np.roll(dists,1))
                
        #     current_radius[non_padding_tri] = np.maximum(current_radius[non_padding_tri], max_dist * np.sqrt(5/3)/2)
        #     if test_triangle_intersection(self.vert_coords[non_padding_tri], current_radius[non_padding_tri]):
        #         pass

        # assert test_triangle_intersection(self.vert_coords[non_padding_tri], current_radius[non_padding_tri])
        # for i, edge in enumerate(self.edges):
        #     current_radius[edge[0]] = max(current_radius[edge[0]], edge_dists[i] * np.sqrt(5/3)/2)
        #     current_radius[edge[1]] = max(current_radius[edge[1]], edge_dists[i] * np.sqrt(5/3)/2)
        #np.where(constricted_radii, self.vertex_radii, current_radius)
        
        # current_radius = np.where(self.constricted_radii, self.vertex_radii, np.minimum(current_radius, self.vertex_radii))
        
        # # Calculate minimum radius for each vertex

        min_radius = np.zeros(vertices.shape[0])

        outer, _ = sphere_intersections(vertices[triangles], current_radius[triangles])

        intersection_nan = np.any(np.isnan(outer), axis=1)

        for tri_id, tri in enumerate(triangles):

            if not intersection_nan[tri_id]:
                
                dists = minimum_intersecting_radii(flattened_triangles[tri_id], current_radius[tri], accept_nan=True)
            
            else:
         
                center = np.mean(flattened_triangles[tri_id], axis=0)

                dists = np.linalg.norm(flattened_triangles[tri_id] - center, axis=1)

                max_height = np.nanmax(np.sqrt(current_radius[tri]**2-dists**2))

                mean_height = np.mean(dists)/3

                height = np.nanmax((max_height, mean_height))

                current_radius[tri] = np.sqrt(dists**2 + height**2)


                dists = minimum_intersecting_radii(flattened_triangles[tri_id], current_radius[tri])

                assert not np.any(np.isnan(dists)), "Minimum distance not calculated"

            # match num_nan:
            #     case 1:
            #         non_nan = tri[~isnan]
            #         outer, inner = circle_intersections(
            #             flattened_triangles[tri_id, ~isnan][0,:2], 
            #             flattened_triangles[tri_id, ~isnan][1,:2], 
            #             current_radius[tri[~isnan][0]],
            #             current_radius[tri[~isnan][1]],
            #         )
            #         min_dist = np.linalg.norm(flattened_triangles[tri_id, isnan][0,:2] - inner)
            #         max_dist = np.linalg.norm(flattened_triangles[tri_id, isnan][0,:2] - outer)
            #         current_radius[tri[isnan][0]] = min_dist + (max_dist-min_dist)/3
            #         try:
            #             dists = minimum_intersecting_radii(flattened_triangles[tri_id], current_radius[tri])
            #         except:
            #             print("Radii correction did not work")
            #     case 2:
            #         raise Exception("NEED TO IMPLEMENT FOR {}".format(num_nan))
            #     case 3:
            #         raise Exception("NEED TO IMPLEMENT FOR {}".format(num_nan))
        

            min_radius[tri] = np.where(min_radius[tri] > dists , min_radius[tri], dists)
        
        # min_radius = np.where(constricted_radii, current_radius,  min_radius)
        
        # assert not np.any(current_radius < min_radius)

        # Find overlapping sites
        outer, inner = sphere_intersections(self.vert_coords[self.tri_verts], current_radius[self.tri_verts])
        # TODO: Limit the comparrison to close sites 

        diff = (current_radius - min_radius)
        comparrison_order = np.argsort(diff)
        for i in range(5):
            for comp_vertex_id in comparrison_order[::-1]:
                site_dists = np.sqrt(np.sum((np.vstack((outer, inner)) - vertices[comp_vertex_id])**2, axis=1))
                
                # id of site
                closest_site_id = np.nanargmin(site_dists)
                closest_site_coords = np.vstack((outer, inner))[closest_site_id]
                closest_site_triangle = closest_site_id % self.tri_verts.shape[0]
                closest_site_dist = site_dists[closest_site_id]
                
                affected_tris = self.vert_tris[comp_vertex_id]

                tri_vert_coords = self.vert_coords[self.tri_verts[closest_site_triangle]]
                if closest_site_dist < self.vertex_radii[comp_vertex_id] and comp_vertex_id not in self.tri_verts[closest_site_triangle]:
                    if closest_site_dist < min_radius[comp_vertex_id]:
                        if i == 4:
                            print("Problem with vertex id {} and site {}".format(comp_vertex_id, closest_site_id))
                            print("constricted: {} - {}".format(comp_vertex_id, self.constricted_radii[comp_vertex_id]))
                            print("site tri vert ids: {}".format(self.tri_verts[closest_site_triangle]))
                            print("site tri vert constricted: {} \n".format(self.constricted_radii[self.tri_verts[closest_site_triangle]]))
                    else:
                        current_radius[comp_vertex_id] = (closest_site_dist + min_radius[comp_vertex_id])/2
                        for tri_id in self.vert_tris[comp_vertex_id]:
                            tri = self.tri_verts[tri_id]
                            dists = minimum_intersecting_radii(flattened_triangles[tri_id], current_radius[tri])
                            min_radius[tri] = np.where(min_radius[tri] > dists , min_radius[tri], dists)

                new_outer, new_inner = sphere_intersections(self.vert_coords[self.tri_verts[affected_tris]], current_radius[self.tri_verts[affected_tris]])
                outer[affected_tris] = new_outer
                inner[affected_tris] = new_inner

        self.vertex_radii = current_radius
    

    def generate_voronoi_sites(self):
        
        outer, inner = sphere_intersections(self.vert_coords[self.tri_verts], self.vertex_radii[self.tri_verts])
        n = outer.shape[0]
        vertex_neighbouring_sites = [[] for n in range(self.vert_coords.shape[0])]
        for i in range(self.tri_verts.shape[0]):
            for j in range(3):
                vertex_neighbouring_sites[self.tri_verts[i,j]].append(i)
                vertex_neighbouring_sites[self.tri_verts[i,j]].append(i+n)

        return outer, inner, vertex_neighbouring_sites 


