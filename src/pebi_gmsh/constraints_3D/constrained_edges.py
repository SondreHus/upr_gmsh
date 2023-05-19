import numpy as np
from pebi_gmsh.utils_3D.densityfield import InscribedCircleField, triangle_inscribed_circle_field
from pebi_gmsh.utils_3D.sphere_intersection import flatten_planar_centers, sphere_intersections
from pebi_gmsh.utils_2D.circle_intersections import circle_intersections, circle_intersection_height
import gmsh

import plotly.graph_objects as go
    
# Coefficient from x -> y
# I: Inscribed circle radius
# R: Ideal vertex radius
# D: Edge distance of cell triangle
# F: Density field cell size
I_R_COEFF = np.sqrt(5)/3
I_D_COEFF = 2/np.sqrt(3)
I_H_COEFF = 1/3
I_F_COEFF = 1.1*I_D_COEFF

GMSH_COEFF = 0.9

R_D_COEFF = I_D_COEFF/I_R_COEFF
H_D_COEFF = I_D_COEFF/I_H_COEFF

# C: Radius of the intersection between the two spheres on opposing side of an edge
I_C_COEFF = np.sqrt(2)/3
# R_C_COEFF = I_C_COEFF/I_R_COEFF


def get_filled_points(border_coords, normal, constraint_triangles, max_size = 0.1):
    
    # Setup model
    gmsh.initialize()
    gmsh.model.add("face_filler")
    
    # Set parameters which removes default mesh size
    # gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Sets up the inscribed circle size field
    density_field = triangle_inscribed_circle_field(constraint_triangles, normal, I_F_COEFF)
    clamp_field = gmsh.model.mesh.field.add("MathEval")
    
    # Sets up a constant size minimum
    gmsh.model.mesh.field.set_string(clamp_field, "F", str(max_size * I_F_COEFF))
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.set_numbers(min_field, "FieldsList", [density_field, clamp_field])
    
    # The minimum of the two fields is used as the size field
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    points = []
    for coord in border_coords:
        points.append(gmsh.model.geo.add_point(coord[0], coord[1], coord[2]))

    edges = []
    for i in range(len(points)):
        edges.append(gmsh.model.geo.add_line(points[i], points[(i+1)%len(points)]))
    
    loop = gmsh.model.geo.add_curve_loop(edges)

    gmsh.model.geo.add_plane_surface([loop])

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)

    # gmsh.fltk.run()

    gmsh.model.mesh.create_faces()

    _, node_coords, *_ = gmsh.model.mesh.get_nodes()
    _, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)
    node_coords = node_coords.reshape(-1,3)

    return node_coords, tri_nodes


class ConstrainedEdgeCollection:

    def __init__(self) -> None:

        # Coordinates of the vertices
        self.vertex_coords = np.zeros((0,3))

        # Notably, when refering to the edges in this class, we're not talking about the edges between the final triangles,
        # but rather the skeletal edges of the larger mesh constraints, the final conversion to triangular edges will be managed by
        # TriangulatedSurface

        # End vertices of the edges
        self.edge_corners = np.zeros((0,2), dtype=int)

        # Target vertex density of the edges
        self.edge_segment_length = []

        # All vertices of the edges, filled after the edges are subdivided
        self.edge_verts = np.zeros((0,2))

        # Edges making face loop, negative notation means the loop traces the edge backwards
        self.face_edges = []

        self.face_corners = []

        # Orientation of the face edges
        self.face_edge_dirs = []
    
        # Smallest angle(radians) of any other plane intersecting the edge, used for deciding vertex radii 
        self.face_edge_angles = []

        self.face_normals = np.zeros((0,3))

        self.vertex_radii = np.zeros(0, dtype=np.float32)

        self.triangles = np.zeros((0,3), dtype=int)

        self.max_size = 0.05

        self.radius_constricted = np.zeros(0, dtype=bool)

        self.inner_loops = []

    def set_max_size(self, max_size):
        self.max_size = max_size


    def add_vertices(self, coords, constrained_radius = True):
        idx = list(range(self.vertex_coords.shape[0], self.vertex_coords.shape[0] + coords.shape[0]))
        self.vertex_coords = np.vstack((self.vertex_coords, coords))
        self.vertex_radii = np.r_[self.vertex_radii, np.zeros(len(idx)) + np.inf]
        if constrained_radius:
            self.radius_constricted = np.r_[self.radius_constricted, np.ones(len(idx), dtype=bool)]
        else:
            self.radius_constricted = np.r_[self.radius_constricted, np.zeros(len(idx), dtype=bool)]
        return np.array(idx) 
    

    
    def add_edges(self, vertex_idx):
        idx = list(range(self.edge_corners.shape[0], self.edge_corners.shape[0] + vertex_idx.shape[0]))
        self.edge_corners = np.vstack((self.edge_corners, vertex_idx))
        
        # TODO: Implement proper vertex density method
        self.edge_segment_length += [0.2 for n in range(len(idx))]

        return np.array(idx)

    def add_face(self, edge_idx):
        forward_oriented = []
        start_edge = self.edge_corners[edge_idx[0]]
        face_verts = []
        if start_edge[0] in self.edge_corners[edge_idx[1]]:
            last_vert = start_edge[0]
            forward_oriented.append(False)
        elif start_edge[1] in self.edge_corners[edge_idx[1]]:
            last_vert = start_edge[1]
            forward_oriented.append(True)
        else:
            raise Exception("Face edge index {} is not connected to edge index {}".format(edge_idx[0], edge_idx[1]))
        face_verts.append(last_vert)
        for edge_id in edge_idx[1:]:
            if last_vert not in self.edge_corners[edge_id]:
                raise Exception("Face edge index {} is not connected to previous vertex".format(edge_id))
            elif last_vert == self.edge_corners[edge_id, 0]:
                forward_oriented.append(True)
                last_vert = self.edge_corners[edge_id, 1]
            else:
                forward_oriented.append(False)
                last_vert = self.edge_corners[edge_id, 0]
            face_verts.append(last_vert)

        face_normal = np.cross(
            (self.vertex_coords[self.edge_corners[edge_idx[0] , 1]] - self.vertex_coords[self.edge_corners[edge_idx[0], 0]]) * (1 if forward_oriented[0] else -1),
            (self.vertex_coords[self.edge_corners[edge_idx[-1], 0]] - self.vertex_coords[self.edge_corners[edge_idx[-1],1]]) * (1 if forward_oriented[-1] else -1),
        )
        face_normal = face_normal/np.sqrt(np.sum(face_normal**2))
        self.face_normals = np.vstack((self.face_normals, face_normal))
        self.face_edges.append(edge_idx)
        self.face_edge_dirs.append(forward_oriented)
        self.face_corners.append(face_verts)
        self.inner_loops.append(None)
        return len(self.face_edges)-1

    def populate_edge_vertices(self):
        
        self.edge_verts = []

        for length, vertex_idx in zip(self.edge_segment_length, self.edge_corners):
            dist = np.sum((self.vertex_coords[vertex_idx[0]] - self.vertex_coords[vertex_idx[1]])**2)
            vertex_num = int(np.floor(dist/length + 0.5))

            interp_step_size = 1/(vertex_num+1)
            padding = 0.05
            interp = np.linspace(0 + padding, 1 - padding, vertex_num, endpoint=False)[1:]

            point_coords = self.vertex_coords[vertex_idx[0]] + (self.vertex_coords[vertex_idx[1]] - self.vertex_coords[vertex_idx[0]])*interp[:,None]

            new_vert_idx = self.add_vertices(point_coords)

            self.edge_verts.append(np.r_[[vertex_idx[0]], new_vert_idx, [vertex_idx[1]]])
        print("Constricted edge end: {}".format(self.vertex_coords.shape[0]))

    # TODO: this needs to handle the corner cases of a face 
    def calculate_edge_vertex_radii(self):
        vert_edge_count = np.zeros(self.vertex_coords.shape[0])
        vert_total_distance = np.zeros(self.vertex_coords.shape[0])
        for edge in self.edge_verts:
            for i in range(len(edge)-1):
                dist = np.sqrt(np.sum((self.vertex_coords[edge[i]] - self.vertex_coords[edge[i+1]])**2))
                vert_total_distance[edge[i]] += dist
                vert_total_distance[edge[i+1]] += dist

                vert_edge_count[edge[i]] += 1
                vert_edge_count[edge[i+1]] += 1
        # sqrt(3/8) * side length of tetrahedron
        
        vert_avg_dist = vert_total_distance/vert_edge_count

        # TODO: This currently only sort of handles 90 degree angles, this needs to ge generalized for 
        # all angles as well as incorporate a better system for edge vertex density 

        # Ugh, bad coder, bad!
        vert_scalar = np.ones(vert_avg_dist.shape)
        for edge in self.edge_verts:
            vert_scalar[edge[1]] = 0.82
            vert_scalar[edge[-2]] = 0.82


        self.vertex_radii = np.sqrt(3/8) * vert_avg_dist * vert_scalar

    def get_constraint_trignales(self, face_id):
        
        triangles = []
        for i in range(len(self.face_edges)):
            if i == face_id:
                continue
            edges = self.face_edges[i]
            orientation = self.face_edge_dirs[i]
            pivot_vertex = self.vertex_coords[self.edge_corners[edges[0], (0 if orientation[0] else 1)]]
            for edge, dir in zip(edges[1:-1], orientation[1:-1]):
                edge_verts = self.vertex_coords[self.edge_corners[edge, ::(1 if dir else -1)]]
                triangles.append(np.vstack((pivot_vertex, edge_verts)))
            
        return np.array(triangles)
    


    def construct_face_padding(self, face_id):
        
        face_normal = self.face_normals[face_id]
        
        constraint_triangles = self.get_constraint_trignales(face_id)
        density_field = InscribedCircleField(self.face_normals[face_id], constraint_triangles)

        # Constructs main large triangles
        loop_vert_ids = []
        for edge_dir, edge in zip(self.face_edge_dirs[face_id], self.face_edges[face_id]):
            loop_vert_ids += [self.edge_verts[edge][::(1 if edge_dir else -1)][:-1]]
            #edge_vert_coords = self.vertex_coords[self.edge_verts[edge]]
        loop_vert_ids = np.hstack(loop_vert_ids)
        edge_vert_coords = self.vertex_coords[loop_vert_ids]
        flattened_verts, invertion, origo = flatten_planar_centers(edge_vert_coords, face_normal)
        
        flattened_dirs = np.roll(flattened_verts, -1, axis=0) - flattened_verts
        flattened_dirs = flattened_dirs/np.sqrt(np.sum(flattened_dirs**2, axis=1))[:, None]
        flattened_dirs = flattened_dirs@np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        # The true main triangle vertices can be arbitrarily close to these vertices in accordance with their target radii
        main_triangle_inscribed = np.zeros(loop_vert_ids.shape[0])
        for i in range(5):
            _ , main_triangle_verts = circle_intersections(
                flattened_verts[:,:2], 
                np.roll(flattened_verts[:,:2], -1, axis=0), 
                # np.sqrt(self.vertex_radii[loop_vert_ids]**2 - (main_triangle_inscribed*I_H_COEFF)**2) + main_triangle_inscribed * I_H_COEFF,
                # np.sqrt(self.vertex_radii[np.roll(loop_vert_ids, -1)]**2 - (main_triangle_inscribed * I_H_COEFF)**2) + main_triangle_inscribed * I_H_COEFF,

                self.vertex_radii[loop_vert_ids] + main_triangle_inscribed * I_C_COEFF,
                self.vertex_radii[np.roll(loop_vert_ids, -1)] + main_triangle_inscribed * I_C_COEFF
            )
            main_triangle_verts = np.pad(main_triangle_verts, ((0,0),(0,1)))
            for i, t_m in enumerate(main_triangle_verts): 
                main_triangle_inscribed[i] = min(density_field.distance(t_m @ invertion.T + origo), self.max_size)
    
        main_triangle_radii = main_triangle_inscribed * I_R_COEFF
        
        main_triangle_vert_ids = self.add_vertices(main_triangle_verts@invertion.T + origo)

        for i, id in enumerate(main_triangle_vert_ids):
            self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], loop_vert_ids[(i+1)%len(loop_vert_ids)], id]))
        self.vertex_radii[main_triangle_vert_ids] = main_triangle_radii

        inner_loop_vert_ids = []

        for i in range(main_triangle_verts.shape[0]):
            inner_loop_vert_ids.append(main_triangle_vert_ids[i-1])

            r = self.vertex_radii[loop_vert_ids[i]]
            corner = flattened_verts[i]
            dir_0 = main_triangle_verts[i-1] - corner
            dir_1 = main_triangle_verts[i] - corner
            angle_from = (np.arctan2(dir_0[1], dir_0[0]) + 2*np.pi)%(2*np.pi)
            angle_to = (np.arctan2(dir_1[1], dir_1[0]) + 2*np.pi)%(2*np.pi)
            angle_diff = ((angle_to - angle_from - 2*np.pi)%(-2*np.pi))
            
            
            avg_main_radii = (main_triangle_radii[i-1] + main_triangle_radii[i])/2

            sample_angles = np.linspace(angle_from, angle_from + angle_diff, 4,  endpoint=False)[1:]

            sample_verts = corner + np.array([np.cos(sample_angles), np.sin(sample_angles), np.zeros(sample_angles.shape)]).T * (r + avg_main_radii)

            sample_radii = np.minimum(np.array([density_field.distance(sample_vert) for sample_vert in (sample_verts@invertion.T + origo)]), self.max_size) * I_R_COEFF 

            # Sample-wise 1/d
            inverts = np.r_[0.5*main_triangle_radii[i-1]**-1, sample_radii**-1, 0.5*main_triangle_radii[i]**-1] / R_D_COEFF 

            steps = int(np.ceil(r*abs(angle_diff) * np.sum(inverts)/(inverts.shape[0]-1)))+1

            # steps = int(np.ceil(r*abs(angle_diff)/(np.sqrt(8/3)*avg_main_radii)))

            angles = np.linspace(angle_from, angle_from + angle_diff, steps,  endpoint=False)[1:]

            # fan_center_dist = np.sqrt(r**2 - (inscribed_radii * I_H_COEFF)**2) + (inscribed_radii * I_H_COEFF)
            fan_verts = corner + np.array([np.cos(angles), np.sin(angles), np.zeros(angles.shape)]).T * (r + avg_main_radii)#* R_C_COEFF) 
            
            # last_radius = main_triangle_radii[i-1]
            # current_angle_diff = 0
            # potential_angle = current_angle_diff + last_radius*np.sqrt(8/3)/r
            
            
            if len(angles) > 0:
                # pressure = True
                # while pressure:
                for _ in range(20):
                    
                    inscribed_radii = np.minimum(np.array([density_field.distance(fan_vert) for fan_vert in (fan_verts@invertion.T + origo)]), self.max_size)
                    radii = inscribed_radii * I_R_COEFF
                    dists = np.sqrt(np.sum((np.vstack((fan_verts, main_triangle_verts[i])) - np.vstack((main_triangle_verts[i-1], fan_verts)))**2, axis=1))
                    mean_radii = (np.r_[main_triangle_radii[i-1], radii] + np.r_[radii, main_triangle_radii[i]])
                    forces = 1 - (dists/(mean_radii * R_D_COEFF))
                    force_diffs = forces[1:] - forces[:-1]
                    angles += force_diffs*0.1

                    fan_center_dist = np.sqrt(r**2 - (inscribed_radii * I_H_COEFF)**2) + (inscribed_radii * I_H_COEFF)
                    #(r + radii * R_C_COEFF)
                    fan_verts = corner + np.array([np.cos(angles), np.sin(angles), np.zeros(angles.shape)]).T * fan_center_dist[:,None]
                    # if sum(forces) > 0:
                    #     steps -= 1
                    #     angles = np.linspace(angle_from, angle_from + angle_diff, steps,  endpoint=False)[1:]
                    #     fan_verts = corner + np.array([np.cos(angles), np.sin(angles), np.zeros(angles.shape)]).T * (r + avg_main_radii)
                    # else:
                    #     pressure = False
            
                fan_vert_ids = self.add_vertices(fan_verts@invertion.T + origo, constrained_radius=False)
                
                inner_loop_vert_ids = inner_loop_vert_ids + fan_vert_ids.tolist()

                self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], fan_vert_ids[0], main_triangle_vert_ids[i-1]]))

                for j in range(1, fan_vert_ids.shape[0]):
                    self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], fan_vert_ids[j], fan_vert_ids[j-1]]))

                self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], main_triangle_vert_ids[i], fan_vert_ids[-1]]))
                
                self.vertex_radii[fan_vert_ids] = radii 
                
               
            else:
                self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], main_triangle_vert_ids[i], main_triangle_vert_ids[i-1]]))
            # steps = (np.sqrt(np.sum(dir_0**2)) + np.sqrt(np.sum(dir_1**2))/2) 
                
        inner_loop_vert_ids = np.array(inner_loop_vert_ids)


        # return inner_loop_vert_ids
        self.inner_loops[face_id] = inner_loop_vert_ids

        print("Fan edge end: {}".format(self.vertex_coords.shape[0]))
        return
        
    def fill_inner_loops(self):
        for i, loop in enumerate(self.inner_loops):
            if loop is None:
                continue

            constraint_triangles = self.get_constraint_trignales(i)
            density_field = InscribedCircleField(self.face_normals[i], constraint_triangles)

            constraint_triangles = self.get_constraint_trignales(i)
            all_inner_vertices, all_inner_triangles = get_filled_points(self.vertex_coords[loop], self.face_normals[i], constraint_triangles, self.max_size)
            new_vertices = all_inner_vertices[loop.shape[0]:]
            
            new_vertex_ids = self.add_vertices(new_vertices, constrained_radius=False)
            # These coefficients should be wrong
            new_vertex_radii = np.minimum(np.array([density_field.distance(vert) for vert in new_vertices]), self.max_size) * I_R_COEFF
    
            self.vertex_radii[new_vertex_ids] = new_vertex_radii
            
            all_inner_triangles = all_inner_triangles.reshape(-1,3) - 1 # Gmsh starts indexing at 1
            inner_vertex_ids = np.r_[loop, new_vertex_ids]

            new_triangles = inner_vertex_ids[all_inner_triangles]

            self.triangles = np.vstack((self.triangles, new_triangles))



