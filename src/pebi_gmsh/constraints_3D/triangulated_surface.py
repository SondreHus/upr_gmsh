import numpy as np
from scipy.spatial import Voronoi
from pebi_gmsh.utils_3D.sphere_intersection import (sphere_intersections, flatten_sphere_centers)
from pebi_gmsh.utils_2D.circle_intersections import circle_intersections
from pebi_gmsh.utils_2D.circumcircle import circumcircle

from functools import reduce

def minimum_intersecting_radii(flattened_triangles, triangle_radii):
    comp = np.array([[1,2],[2,0],[0,1]])
    _, inner = circle_intersections(
        flattened_triangles[comp[:,0], :-1], 
        flattened_triangles[comp[:,1], :-1], 
        triangle_radii[comp][:,0], 
        triangle_radii[comp][:,1]
    )

    return np.sqrt(np.sum((inner - flattened_triangles[:, :-1])**2, axis=1))


class TriangulatedSurface:

    # Coordinates of vertices
    vertices: np.ndarray
    
    # Idx of every triangle a vertex is a part of 
    vertex_triangles: list

    # Idx of each vertex every triangle consists of
    triangle_vertices: np.ndarray

    # Idx of each edge every triangle consists of
    triangle_edges: np.ndarray

    # triangle_neighbours: np.ndarray
    
    # Idx of vertices each edge consists of 
    edges: np.ndarray

    # Dict from vertex id pair -> edge id
    edge_dict: dict
    
    # Triangles adjacent to each edge
    edge_triangles: list
    def __init__(self, vertices = None, triangles: np.ndarray = None) -> None:
        self.vertices = vertices
        self.triangle_vertices = triangles
        self.triangle_edges = np.zeros(triangles.shape)
        self.edge_dict = {}
        self.vertex_triangles = [[] for i in range(vertices.shape[0])]
        self.vertex_radii = np.zeros(vertices.shape[0])
        self.edge_triangles = []
        
        
        edges = []
        for comp_vertex_id, triangle in enumerate(triangles):
            
            tri_edges = np.zeros(3)
            for j in range(3):
                self.vertex_triangles[triangle[j]].append(comp_vertex_id)
                if (edge_vertex_idx := tuple(sorted((triangle[j], triangle[(j+1)%3])))) not in self.edge_dict:
                    # Add new edge and assign it to the edge dictionary
                    tri_edges[j] = self.edge_dict[edge_vertex_idx] = len(edges)
                    edges.append(edge_vertex_idx)
                    self.edge_triangles.append([comp_vertex_id])
                else:
                    tri_edges[j] = self.edge_dict[edge_vertex_idx]
                    # Update edge_triangles with new triangle
                    self.edge_triangles[self.edge_dict[edge_vertex_idx]].append(comp_vertex_id)
            
            self.triangle_edges[comp_vertex_id] = tri_edges
           
        self.edges = np.array(edges)
        self.vertex_triangles = [np.array(v) for v in self.vertex_triangles]
        
        # Naive vertex radii
        current_radius = np.zeros(vertices.shape[0])

        flattened_triangles, *_ = flatten_sphere_centers(vertices[triangles])

        _, tri_distances = circumcircle(flattened_triangles[:,0, :-1], flattened_triangles[:,1, :-1], flattened_triangles[:,2, :-1])

        for comp_vertex_id, distance in enumerate(tri_distances):
            current_radius[triangles[comp_vertex_id]] = np.where(current_radius[triangles[comp_vertex_id]] > distance , current_radius[triangles[comp_vertex_id]], distance)
        
        current_radius = current_radius * 1.01

        # Calculate minimum radius for each vertex

        min_radius = np.zeros(vertices.shape[0])

        for comp_vertex_id, tri in enumerate(triangles):
            # comp = np.array([[1,2],[2,0],[0,1]])
            
            # _, inner = circle_intersections(
            #     flattened_triangles[comp_vertex_id, comp[:,0], :-1], 
            #     flattened_triangles[comp_vertex_id, comp[:,1], :-1], 
            #     current_radius[tri[comp][:,0]], 
            #     current_radius[tri[comp][:,1]]
            # )

            # dists = np.sqrt(np.sum((inner - flattened_triangles[comp_vertex_id, :, :-1])**2, axis=1))
            

            dists = minimum_intersecting_radii(flattened_triangles[comp_vertex_id], current_radius[tri])
            min_radius[tri] = np.where(min_radius[tri] > dists , min_radius[tri], dists)
        
        
        
        assert not np.any(current_radius < min_radius)

        # Find overlapping sites
        outer, inner = sphere_intersections(self.vertices[self.triangle_vertices], current_radius[self.triangle_vertices])
        # TODO: Limit the comparrison to close sites 

        diff = (current_radius - min_radius)
        comparrison_order = np.argsort(diff)
        

        # Problemet, reduksjon av en enkelt vertex sin radius er ikke tilstrekkelig
        # videre reduksjon krever at den "andre siten" blir flyttet vekk
        # Det er 3 spherer som er ansvarlige for denne siten
        # Kompromisset?
        # Prøv å holde alle spherer så store (Taxicab) som mulig
        # Steg 1: reduser alle radiusene til en faktor av deres (orginale?) diff
        # Steg 2: repeter fram til de ikke intersecter
        # Steg 3: Oppdater min dist og diff slik at meshet fortsatt er sammenkoblet


        # Må gjøre dette for alle overlappende sites
        for comp_vertex_id in comparrison_order:
            site_dists = np.sqrt(np.sum((np.vstack((outer, inner)) - vertices[comp_vertex_id])**2, axis=1))

            closest_site_id = np.nanargmin(site_dists)
            closest_site_dist = site_dists[closest_site_id]
            if closest_site_dist < min_radius[comp_vertex_id]:
                closest_site_vertices = self.triangle_vertices[closest_site_id]
                factors = 0.8 ** (np.arange(10) + 1)

                tri_radii = min_radius[closest_site_vertices] + factors.reshape(-1,1)*diff[closest_site_vertices]
                
                outer, inner = sphere_intersections(np.tile(self.vertices[closest_site_vertices], (10,1,1)), tri_radii)
                dists = np.min((
                    np.sqrt(np.sum((outer - vertices[comp_vertex_id])**2, axis=1)),
                    np.sqrt(np.sum((inner - vertices[comp_vertex_id])**2, axis=1))
                ), axis=0)

                vertex_radii = min_radius[comp_vertex_id] + factors.reshape(-1,1)*diff[comp_vertex_id]

                factor_index = np.where(vertex_radii < dists)[0][0]

                factor = factors[factor_index]

                current_radius[closest_site_vertices] = min_radius[closest_site_vertices] + factors*diff[closest_site_vertices]
                current_radius[comp_vertex_id] = min_radius[comp_vertex_id] + factor*diff[comp_vertex_id]

                # Update the minimum distance of all affected vertices

                affected_tris = reduce(
                    np.union1d, 
                    [self.vertex_triangles[site_vertex] for site_vertex in self.triangle_vertices[closest_site_id]] +\
                    [self.vertex_triangles[comp_vertex_id]]
                )

                for comp_vertex_id, tri in enumerate(triangles[affected_tris]):

                    dists = minimum_intersecting_radii(flattened_triangles[affected_tris[comp_vertex_id]], current_radius[tri])
                    min_radius[tri] = np.where(min_radius[tri] > dists , min_radius[tri], dists)

                # Limit this to only the affected vertices

                diff = current_radius - min_radius
                
                #closest_site_dist = min_radius[i]
                #print("FUCK")
            else:
                current_radius[comp_vertex_id] = closest_site_dist
                diff[comp_vertex_id] = current_radius[comp_vertex_id] - min_radius[comp_vertex_id]

            tri_ids = self.vertex_triangles[comp_vertex_id]
            tri_verts = self.triangle_vertices[tri_ids]
            new_outer, new_inner = sphere_intersections(self.vertices[tri_verts], current_radius[tri_verts])

            outer[tri_ids] = new_outer
            inner[tri_ids] = new_inner
        
        self.vertex_radii = current_radius
    
    

    def generate_voronoi_sites(self):
        
        outer, inner = sphere_intersections(self.vertices[self.triangle_vertices], self.vertex_radii[self.triangle_vertices])
        n = outer.shape[0]
        vertex_neighbouring_sites = [[] for n in range(self.vertices.shape[0])]
        for i in range(self.triangle_vertices.shape[0]):
            for j in range(3):
                vertex_neighbouring_sites[self.triangle_vertices[i,j]].append(i)
                vertex_neighbouring_sites[self.triangle_vertices[i,j]].append(i+n)

        return outer, inner, vertex_neighbouring_sites 
# def create_test_surface():

#     surface = []
#     # tris = [[0,1,10]]
#     tris = []
#     for x in range(10):
#         for y in range(10):
#             surface.append([x + (y%2)/2,y,x])
    
#     for x in range(9):
#         for y in range(9):
#             if x%2 == 0:
#                 tris.append([x + 10*y, x + 10*y + 1, x + 10*y + 10])
#                 tris.append([x + 10*y + 1, x + 10*y + 11, x + 10*y + 10])
#             else:

#                 tris.append([x + 10*y, x + 10*y + 11, x + 10*y + 10])
#                 tris.append([x + 10*y, x + 10*y + 1, x + 10*y + 11])

#     return np.array(surface), np.array(tris)


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

                    
