import numpy as np
from scipy.spatial import Voronoi
from pebi_gmsh.utils_3D.sphere_intersection import (sphere_intersections, flatten_sphere_centers)
from pebi_gmsh.utils_2D.circle_intersections import circle_intersections
from pebi_gmsh.utils_2D.circumcircle import circumcircle
from pebi_gmsh.utils_3D.line_sphere_intersection import line_sphere_intersection, sphere_line_reduction
from functools import reduce

def minimum_intersecting_radii(flattened_triangles, triangle_radii):
    comp = np.array([[1,2],[2,0],[0,1]])
    _, inner = circle_intersections(
        flattened_triangles[...,comp[:,0], :-1], 
        flattened_triangles[...,comp[:,1], :-1], 
        triangle_radii[...,comp[:,0]], 
        triangle_radii[...,comp[:,1]]
    )

    return np.sqrt(np.sum((inner - flattened_triangles[:, :-1].reshape(-1,2))**2, axis=1)).reshape(triangle_radii.shape)


class TriangulatedSurface:

    # Coordinates of vertices
    vert_coords: np.ndarray
    
    # Idx of every triangle a vertex is a part of 
    vert_tris: list

    # Idx of each vertex every triangle consists of
    tri_verts: np.ndarray

    # Idx of each edge every triangle consists of
    tri_edges: np.ndarray

    # triangle_neighbours: np.ndarray
    
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
        self.vertex_radii = np.zeros(vertices.shape[0]) if radii is None else radii
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
        current_radius = np.zeros(vertices.shape[0])
        # edge_num = np.zeros(vertices.shape[0])

        # flattened_triangles, *_ = flatten_sphere_centers(vertices[triangles])

        # _, tri_distances = circumcircle(flattened_triangles[:,0, :-1], flattened_triangles[:,1, :-1], flattened_triangles[:,2, :-1])

        # for comp_vertex_id, distance in enumerate(tri_distances):
        #     current_radius[triangles[comp_vertex_id]] = np.where(current_radius[triangles[comp_vertex_id]] > distance , current_radius[triangles[comp_vertex_id]], distance)
        
        edge_dists = np.sqrt(np.sum((self.vert_coords[self.edges[:,0]] - self.vert_coords[self.edges[:,1]])**2, axis=-1))
        for i, edge in enumerate(self.edges):
            current_radius[edge[0]] = max(current_radius[edge[0]], edge_dists[i])
            current_radius[edge[1]] = max(current_radius[edge[1]], edge_dists[i])


        current_radius = current_radius * 0.61

        # # Calculate minimum radius for each vertex

        min_radius = np.zeros(vertices.shape[0])

        # for comp_vertex_id, tri in enumerate(triangles):
        #     # comp = np.array([[1,2],[2,0],[0,1]])
            
        #     # _, inner = circle_intersections(
        #     #     flattened_triangles[comp_vertex_id, comp[:,0], :-1], 
        #     #     flattened_triangles[comp_vertex_id, comp[:,1], :-1], 
        #     #     current_radius[tri[comp][:,0]], 
        #     #     current_radius[tri[comp][:,1]]
        #     # )

        #     # dists = np.sqrt(np.sum((inner - flattened_triangles[comp_vertex_id, :, :-1])**2, axis=1))
            

        #     dists = minimum_intersecting_radii(flattened_triangles[comp_vertex_id], current_radius[tri])
        #     min_radius[tri] = np.where(min_radius[tri] > dists , min_radius[tri], dists)
        
        
        
        # assert not np.any(current_radius < min_radius)

        # # Find overlapping sites
        # outer, inner = sphere_intersections(self.vert_coords[self.tri_verts], current_radius[self.tri_verts])
        # # TODO: Limit the comparrison to close sites 

        # diff = (current_radius - min_radius)
        # comparrison_order = np.argsort(diff)
        

        # Problemet, reduksjon av en enkelt vertex sin radius er ikke tilstrekkelig
        # videre reduksjon krever at den "andre siten" blir flyttet vekk
        # Det er 3 spherer som er ansvarlige for denne siten
        # Kompromisset?
        # Prøv å holde alle spherer så store (Taxicab) som mulig
        # Steg 1: reduser alle radiusene til en faktor av deres (orginale?) diff
        # Steg 2: repeter fram til de ikke intersecter
        # Steg 3: Oppdater min dist og diff slik at meshet fortsatt er sammenkoblet

        # assert np.all(current_radius > min_radius)
        # # Må gjøre dette for alle overlappende sites
        # for comp_vertex_id in comparrison_order:
        #     site_dists = np.sqrt(np.sum((np.vstack((outer, inner)) - vertices[comp_vertex_id])**2, axis=1))

        #     closest_site_id = np.nanargmin(site_dists)
        #     closest_site_coords = np.vstack((outer, inner))[closest_site_id]
        #     closest_site_triangle = closest_site_id % self.tri_verts.shape[0]
        #     closest_site_dist = site_dists[closest_site_id]

        #     tri_vert_coords = self.vert_coords[self.tri_verts[closest_site_triangle]]
        #     if closest_site_dist < min_radius[comp_vertex_id]:


        #         tri_flat, c0, inverse = flatten_sphere_centers(tri_vert_coords) 
                
        #         center, _ = circumcircle(tri_flat[:,0,:2], tri_flat[:,1,:2], tri_flat[:,2,:2])
        #         center = np.pad(center, ((0,0),(0,1))) @ inverse + c0
                
        #         tri_dir = center - closest_site_coords
                
        #         dists = line_sphere_intersection(tri_vert_coords, min_radius[self.tri_verts[closest_site_triangle]], np.tile(closest_site_coords, (3,1)), np.tile(tri_dir, (3,1)))[0]
                
        #         max_tri_dist = np.nanmin(dists)
        #         if not np.isnan(max_tri_dist):
        #             tri_dir = tri_dir * max_tri_dist
                
        #         a, _ = sphere_line_reduction(self.vert_coords[comp_vertex_id], current_radius[comp_vertex_id], min_radius[comp_vertex_id], closest_site_coords, tri_dir)
        #         assert a > 0 and a <= 1 
        #         new_site = tri_dir * a + closest_site_coords

        #         new_tri_radii = np.sqrt(np.sum((tri_vert_coords - new_site)**2, axis=-1))

        #         current_radius[self.tri_verts[closest_site_triangle]] = new_tri_radii 
        #         assert np.all(current_radius > min_radius)
             
        #         current_radius[comp_vertex_id] = current_radius[comp_vertex_id] - a * (current_radius[comp_vertex_id] - min_radius[comp_vertex_id])        
        #         assert np.all(current_radius > min_radius)
        #         # factors = 0.8 ** (np.arange(10) + 1)

        #         # tri_radii = min_radius[closest_site_vertices] + factors.reshape(-1,1)*diff[closest_site_vertices]
                
        #         # outer, inner = sphere_intersections(np.tile(self.vertices[closest_site_vertices], (10,1,1)), tri_radii)
        #         # dists = np.min((
        #         #     np.sqrt(np.sum((outer - vertices[comp_vertex_id])**2, axis=1)),
        #         #     np.sqrt(np.sum((inner - vertices[comp_vertex_id])**2, axis=1))
        #         # ), axis=0)

        #         # vertex_radii = min_radius[comp_vertex_id] + factors.reshape(-1,1) * diff[comp_vertex_id]

        #         # factor_index = np.where(vertex_radii < dists)[0][0]

        #         # factor = factors[factor_index]

        #         # current_radius[closest_site_vertices] = min_radius[closest_site_vertices] + factors * diff[closest_site_vertices]
        #         # current_radius[comp_vertex_id] = min_radius[comp_vertex_id] + factor*diff[comp_vertex_id]



        #         # Update the minimum distance of all affected vertices
                
        #         affected_tris = reduce(
        #             np.union1d, 
        #             [self.vert_tris[site_vertex] for site_vertex in self.tri_verts[closest_site_triangle]] +\
        #             [self.vert_tris[comp_vertex_id]]
        #         )
        #         assert np.all(current_radius > min_radius)
        #         for tri in affected_tris:
        #             print(tri)
        #             dists = minimum_intersecting_radii(flattened_triangles[tri], current_radius[self.tri_verts[tri]])
        #             min_radius[self.tri_verts[tri]] = np.where(min_radius[self.tri_verts[tri]] > dists , min_radius[self.tri_verts[tri]], dists)
        #         assert np.all(current_radius > min_radius)
        #         # Limit this to only the affected vertices

        #         diff = current_radius - min_radius
                
        #         #closest_site_dist = min_radius[i]
        #         #print("FUCK")
        #     else:
        #         current_radius[comp_vertex_id] = closest_site_dist
        #         diff[comp_vertex_id] = current_radius[comp_vertex_id] - min_radius[comp_vertex_id]

        #     tri_ids = self.vert_tris[comp_vertex_id]
        #     tri_verts = self.tri_verts[tri_ids]
        #     new_outer, new_inner = sphere_intersections(self.vert_coords[tri_verts], current_radius[tri_verts])

        #     outer[tri_ids] = new_outer
        #     inner[tri_ids] = new_inner
        
        self.vertex_radii = np.where(constricted_radii, self.vertex_radii, current_radius)
    
    

    def generate_voronoi_sites(self):
        
        outer, inner = sphere_intersections(self.vert_coords[self.tri_verts], self.vertex_radii[self.tri_verts])
        n = outer.shape[0]
        vertex_neighbouring_sites = [[] for n in range(self.vert_coords.shape[0])]
        for i in range(self.tri_verts.shape[0]):
            for j in range(3):
                vertex_neighbouring_sites[self.tri_verts[i,j]].append(i)
                vertex_neighbouring_sites[self.tri_verts[i,j]].append(i+n)

        return outer, inner, vertex_neighbouring_sites 



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from pebi_gmsh.sphere_intersection import sphere_intersections
    
    surface, tris = create_test_surface()
    radii = np.ones(tris.shape)
    test = TriangulatedSurface(surface, tris, radii)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.set_aspect("equal")
    edge_coords = surface[test.edges]
    for i in range(edge_coords.shape[0]):
        ax.plot(edge_coords[i,:,0], edge_coords[i,:,1], edge_coords[i,:,2])
    # ax.scatter(surface[:,0], surface[:,1], surface[:,2])
    a, b = sphere_intersections(surface[tris], radii)
    ax.scatter(b[:,0], b[:,1], b[:,2])
    voronoi = Voronoi(np.vstack((a,b)))
    verts = voronoi.vertices
    ax.scatter(verts[:,0], verts[:,1], verts[:,2])
    # ax.triplot(a)
    plt.show()

                    
