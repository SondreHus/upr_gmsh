import numpy as np
from pebi_gmsh.utils_3D.plane_densityfield import InscribedSphereField, triangle_inscribed_circle_field, LineInscribedField
from pebi_gmsh.utils_3D.sphere_intersection import flatten_planar_centers, sphere_intersections
from pebi_gmsh.utils_3D.tri_mesh import triangulate_polygon
from pebi_gmsh.utils_2D.circle_intersections import circle_intersections, circle_intersection_height
from pebi_gmsh.constraints_3D.triangulated_surface import test_triangle_intersection
from pebi_gmsh.utils_3D.point_densityfield import point_inscribed_distance
import gmsh
import plotly.graph_objects as go
from scipy.interpolate import interp1d
# Coefficient from x -> y

# I: Inscribed circle distance
# R: Ideal vertex radius
# D: Edge distance of cell triangle
# F: Density field cell size
# B: Edge density coefficient
Primary_scale_factor = 1/2

I_R_COEFF = Primary_scale_factor * np.sqrt(5)/2
I_D_COEFF = Primary_scale_factor * np.sqrt(3)
I_H_COEFF = Primary_scale_factor * 1/2
I_F_COEFF = I_D_COEFF 

#sqrt(3*5)/6

I_B_COEFF = 1/2

D_R_COEFF = I_R_COEFF/I_D_COEFF
R_D_COEFF = I_D_COEFF/I_R_COEFF
H_D_COEFF = I_D_COEFF/I_H_COEFF
R_H_COEFF = I_R_COEFF/I_H_COEFF

EDGE_LIPSCHITZ_FACTOR = 1/2

def get_filled_points(border_coords, normal, constraint_triangles, max_size = 0.1):
    
    # Setup model
    gmsh.initialize()
    gmsh.model.add("face_filler")
    
    # Set parameters which removes default mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Sets up the inscribed circle size field
    density_field = triangle_inscribed_circle_field(constraint_triangles, normal, I_F_COEFF)
    clamp_field = gmsh.model.mesh.field.add("MathEval")
    
    # Sets up a constant size minimum
    gmsh.model.mesh.field.set_string(clamp_field, "F", str(max_size))
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.set_numbers(min_field, "FieldsList", [density_field, clamp_field])
    
    # The minimum of the two fields is used as the size field
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    gmsh.model.mesh.field.set_as_background_mesh
    points = []
    for coord in border_coords:
        points.append(gmsh.model.geo.add_point(coord[0], coord[1], coord[2]))

    edges = []
    for i in range(len(points)):
        line_id = gmsh.model.geo.add_line(points[i], points[(i+1)%len(points)])
        edges.append(line_id)
        gmsh.model.geo.mesh.set_transfinite_curve(line_id, 2)
    
    loop = gmsh.model.geo.add_curve_loop(edges)

    gmsh.model.geo.add_plane_surface([loop])

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)

    # gmsh.fltk.run()

    gmsh.model.mesh.create_faces()

    _, node_coords, *_ = gmsh.model.mesh.get_nodes()
    _, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)
    node_coords = node_coords.reshape(-1,3)

    gmsh.finalize()

    return node_coords, tri_nodes



def get_edge_vertices(p_start, r_start, p_end, r_end, triangle_coords, edge_coords, point_coords, max_inscribed):
    line_dir = p_end - p_start
    line_len = np.linalg.norm(line_dir)
    line_dir = line_dir/line_len
    density_field = LineInscribedField(line_dir, triangle_coords, edge_coords, point_coords)

    lipschitz = lambda x: min((x)*EDGE_LIPSCHITZ_FACTOR, (line_len - x)*EDGE_LIPSCHITZ_FACTOR)

    dist_start = r_start
    dist_end = r_end

    

    v_start = p_start + line_dir * dist_start
    v_end = p_end - line_dir * dist_end


    r_v_start = min(density_field.distance(v_start) * I_B_COEFF, max_inscribed, lipschitz(dist_start))
    r_v_end = min(density_field.distance(v_end) * I_B_COEFF, max_inscribed, lipschitz(line_len-dist_end))

    
    dist_start = r_start + r_v_start/2
    dist_end = r_end + r_v_end/2

    v_start = p_start + line_dir * dist_start
    v_end = p_end - line_dir * dist_end


    current_dist = 0
    current_radius = r_v_start
    max_dist = np.linalg.norm(v_end-v_start)
    dists = []
    while current_dist + current_radius < max_dist:
        current_dist += current_radius
        dists.append(current_dist)
        current_radius = min(density_field.distance(v_start + current_dist*line_dir) * I_B_COEFF, max_inscribed, lipschitz(current_dist + dist_start))



    # ideal_dist = max_dist - current_radius
    dists = np.array(dists)*(max_dist/(current_dist + current_radius))
    dists = np.r_[0, dists, max_dist]
    radii = np.zeros(dists.shape[0])
    radii[0] = r_v_start
    radii[-1] = r_v_end
    for i in range(5):
        radii[1:-1] = np.array([min(density_field.distance(v_start + dist * line_dir) * I_B_COEFF, max_inscribed, lipschitz(dist)) for dist in dists])[1:-1]
        mean_radii = (radii[1:] + radii[:-1])/2
        forces = mean_radii - (dists[1:] - dists[:-1])

        dists[1:-1] += (forces[:-1] - forces[1:]) * 0.2
    mean_dist = (dists[1:] - dists[:-1])
    if mean_dist.shape[0] > 1:
        mean_dist = (mean_dist[1:] + mean_dist[:-1])/2
        mean_dist = np.r_[mean_dist[0], mean_dist, mean_dist[-1]]
    else:
        mean_dist = np.r_[mean_dist[0], mean_dist[0]]

    return v_start.reshape(1,3) + dists.reshape(-1,1)*line_dir, mean_dist * 1.2#radii*D_R_COEFF * 1.8

class ConstrainedEdgeCollection:

    def __init__(self) -> None:

        # Coordinates of the vertices
        self.vertex_coords = np.zeros((0,3))

        # Notably, when refering to the edges in this class, we're not talking about the edges between the final triangles,
        # but rather the skeletal edges of the larger mesh constraints, the final conversion to triangular edges will be managed by
        # TriangulatedSurface

        # End vertices of the edges
        self.edge_corners = np.zeros((0,2), dtype=int)

        # Indices of corner vertices
        self.corner_verts = np.zeros(0, dtype=int)

        # All vertices of the edges, filled after the edges are subdivided
        self.edge_verts = np.zeros((0,2))

        # Edges making face loop, negative notation means the loop traces the edge backwards
        self.face_edges = []

        # Triangulations of the original constraint planes
        self.constraint_tris = []

        self.face_corners = []

        # Orientation of the face edges
        self.face_edge_dirs = []
    
        self.face_normals = np.zeros((0,3))

        self.vertex_radii = np.zeros(0, dtype=np.float32)

        # Subdivided triangles used for generating site locations
        self.triangles = np.zeros((0,3), dtype=int)

        self.padding_triangles = np.zeros(0, dtype=bool)

        self.max_size = 0.05

        self.radius_constricted = np.zeros(0, dtype=bool)

        self.inner_loops = []

    def set_max_size(self, max_size):
        self.max_size = max_size


    def add_vertices(self, coords, constrained_radius = True):
        idx = list(range(self.vertex_coords.shape[0], self.vertex_coords.shape[0] + coords.shape[0]))
        self.vertex_coords = np.vstack((self.vertex_coords, coords))
        self.vertex_radii = np.r_[self.vertex_radii, np.zeros(len(idx))]
        if constrained_radius:
            self.radius_constricted = np.r_[self.radius_constricted, np.ones(len(idx), dtype=bool)]
        else:
            self.radius_constricted = np.r_[self.radius_constricted, np.zeros(len(idx), dtype=bool)]
        return np.array(idx) 
    

    
    def add_edges(self, vertex_idx):
        vertex_idx = vertex_idx.reshape(-1,2)
        idx = list(range(self.edge_corners.shape[0], self.edge_corners.shape[0] + vertex_idx.shape[0]))
        self.edge_corners = np.vstack((self.edge_corners, vertex_idx))
        
        for vert_id in vertex_idx.flatten():
            if vert_id not in self.corner_verts:
                self.corner_verts = np.r_[self.corner_verts, vert_id]
        
        return np.array(idx)

    def add_face(self, vertex_idx):
        
        forward_oriented = []

        edge_idx = []
        for i in range(len(vertex_idx)):
            a = vertex_idx[i]
            b = vertex_idx[(i+1)%len(vertex_idx)]
            match = np.all(self.edge_corners == [a,b], axis=1)
            if np.any(match): # [a, b] is in edges
                edge_id = np.where(match)[0][0]
                edge_idx.append(edge_id)
                forward_oriented.append(True)
                continue
            match = np.all(self.edge_corners == [b,a], axis=1)
            if np.any(match): # [a, b] is in edges
                edge_id = np.where(match)[0][0]
                edge_idx.append(edge_id)
                forward_oriented.append(False)
                continue
            
            edge_id = self.add_edges(np.array([a,b]))[0]
            edge_idx.append(edge_id)
            forward_oriented.append(True)

        
        face_normal = np.cross(
            (self.vertex_coords[self.edge_corners[edge_idx[0] , 1]] - self.vertex_coords[self.edge_corners[edge_idx[0], 0]]) * (1 if forward_oriented[0] else -1),
            (self.vertex_coords[self.edge_corners[edge_idx[-1], 0]] - self.vertex_coords[self.edge_corners[edge_idx[-1],1]]) * (1 if forward_oriented[-1] else -1),
        )
        
        face_normal = face_normal/np.sqrt(np.sum(face_normal**2))
        self.face_normals = np.vstack((self.face_normals, face_normal))
        self.face_edges.append(edge_idx)
        self.face_edge_dirs.append(forward_oriented)
        self.face_corners.append(np.array(vertex_idx))
        self.inner_loops.append(None)
        
        self.constraint_tris.append(self.triangulate_face(len(self.face_edges)-1))
        
        return len(self.face_edges)-1

    def populate_edge_vertices(self):
        
        self.edge_verts = []


        for edge_id in range(self.edge_corners.shape[0]):
            
            constraint_triangle_coords = []
            
            for face_id, face_edges in enumerate(self.face_edges):
                if edge_id not in face_edges:
                    constraint_triangle_coords.append(self.vertex_coords[self.constraint_tris[face_id]])

            constraint_triangle_coords = np.vstack(constraint_triangle_coords)
          

            constraint_edge_coords = self.vertex_coords[np.delete(self.edge_corners, edge_id, axis=0)]

            edge_corner_verts = self.edge_corners[edge_id]

            constraint_corner_ids = self.corner_verts[np.logical_and(self.corner_verts != edge_corner_verts[0], self.corner_verts != edge_corner_verts[1])]

            constraint_point_coords = self.vertex_coords[constraint_corner_ids]
            
            inner_vert_coords, inner_vert_radii = get_edge_vertices(
                self.vertex_coords[edge_corner_verts[0]], 
                self.vertex_radii[edge_corner_verts[0]],
                self.vertex_coords[edge_corner_verts[1]],
                self.vertex_radii[edge_corner_verts[1]],
                constraint_triangle_coords,
                constraint_edge_coords, 
                constraint_point_coords,
                self.max_size
            )

            new_vert_idx = self.add_vertices(inner_vert_coords)

            self.vertex_radii[new_vert_idx] = inner_vert_radii

            self.edge_verts.append(np.r_[[edge_corner_verts[0]], new_vert_idx, [edge_corner_verts[1]]])
            
            

    def calculate_edge_vertex_radii(self):

        for corner_id, corner_vert in enumerate(self.corner_verts):
            tris = np.zeros((0,3,3))

            for face_id, face_corners in enumerate(self.face_corners):
                if corner_vert not in face_corners:
                    tris = np.vstack((tris, self.vertex_coords[self.constraint_tris[face_id]]))

            edges = np.zeros((0,2,3))
            for edge_verts in self.edge_corners:
                if corner_vert not in edge_verts:
                    edges = np.vstack((edges, self.vertex_coords[edge_verts].reshape(1,2,3)))

            points = self.vertex_coords[np.delete(self.corner_verts, corner_id)]

            self.vertex_radii[corner_vert] = min(point_inscribed_distance(self.vertex_coords[corner_vert], tris, edges, points)*1/3, self.max_size*2)



    def triangulate_face(self, face_id):
        
        flattened_coords, *_ = flatten_planar_centers(self.vertex_coords[self.face_corners[face_id]], self.face_normals[face_id])
        
        face_tris = triangulate_polygon(flattened_coords[:,:2]).astype(int)

        return self.face_corners[face_id][face_tris]

    def get_constraint_trignales(self, face_id):
        
        triangles = []# np.zeros((0,3))
        for i in range(len(self.face_edges)):
            if i == face_id:
                continue
            triangles.append(self.vertex_coords[self.constraint_tris[i]])
            
        return np.vstack(triangles)
    
        

    def construct_face_padding(self, face_id = None):
        
        if face_id is None:
            for i in range(len(self.face_corners)):
                self.construct_face_padding(i)
            return
        
        face_normal = self.face_normals[face_id]
        
        constraint_triangles = self.get_constraint_trignales(face_id)
        density_field = InscribedSphereField(self.face_normals[face_id], constraint_triangles)

        # Constructs main large triangles
        loop_vert_ids = []
        for edge_dir, edge in zip(self.face_edge_dirs[face_id], self.face_edges[face_id]):
            loop_vert_ids += [self.edge_verts[edge][::(1 if edge_dir else -1)][:-1]]
            #edge_vert_coords = self.vertex_coords[self.edge_verts[edge]]
        loop_vert_ids = np.hstack(loop_vert_ids)
        edge_vert_coords = self.vertex_coords[loop_vert_ids]
        flattened_verts, invertion, origo = flatten_planar_centers(edge_vert_coords, face_normal)
        

        main_triangle_inscribed = np.zeros(loop_vert_ids.shape[0])

        try:
            _ , main_triangle_verts = circle_intersections(
                flattened_verts[:,:2], 
                np.roll(flattened_verts[:,:2], -1, axis=0), 

                self.vertex_radii[loop_vert_ids], 
                self.vertex_radii[np.roll(loop_vert_ids, -1)]
            )
        except:
            raise Exception("Error linking the inner loop vertices together")
        
        main_triangle_verts = np.pad(main_triangle_verts, ((0,0),(0,1)))
        for j, t_m in enumerate(main_triangle_verts): 
            main_triangle_inscribed[j] = min(density_field.distance(t_m @ invertion.T + origo), self.max_size / I_D_COEFF)
            
        main_triangle_radii = main_triangle_inscribed * I_R_COEFF
     
        main_triangle_vert_ids = self.add_vertices(main_triangle_verts@invertion.T + origo, constrained_radius=True)


        for i, id in enumerate(main_triangle_vert_ids):
            self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], loop_vert_ids[(i+1)%len(loop_vert_ids)], id]))
            self.padding_triangles = np.r_[self.padding_triangles, True]
        self.vertex_radii[main_triangle_vert_ids] = main_triangle_radii

        test_triangle = np.vstack((loop_vert_ids, np.roll(loop_vert_ids, -1), main_triangle_vert_ids)).T
        
        assert np.all(test_triangle_intersection(self.vertex_coords[test_triangle], self.vertex_radii[test_triangle]))
        
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

            sample_angles = np.linspace(angle_from, angle_from + angle_diff, 12,  endpoint=True)[1:]

            sample_verts = corner + np.array([np.cos(sample_angles), np.sin(sample_angles), np.zeros(sample_angles.shape)]).T * (r + avg_main_radii)

            sample_radii = np.minimum(np.array([density_field.distance(sample_vert) for sample_vert in (sample_verts@invertion.T + origo)]) * I_R_COEFF , self.max_size * D_R_COEFF) 


            radii_func = interp1d(sample_angles, sample_radii, fill_value="extrapolate")

            angles = []
            angle = angle_from
            radius = main_triangle_radii[i-1]
            angle_dir = np.sign(angle_diff)
            while 0.6 * (radius + main_triangle_radii[i]) * R_D_COEFF/r < angle_dir*(angle_from + angle_diff - angle):
                angle += angle_dir*radius*R_D_COEFF/r
                angles.append(angle)
                radius = radii_func(angle)
            angles = np.array(angles)
            if angles.shape[0] > 0 :
                target_angle = angle_from + angle_diff - angle_dir*(main_triangle_radii[i] * R_D_COEFF/r)
                angles = (angles - angle_from) * (target_angle-angle_from)/(angle-angle_from) + angle_from


            fan_verts = corner + np.array([np.cos(angles), np.sin(angles), np.zeros(angles.shape)]).T * r 
            if len(angles) > 0:
                
                for k in range(20):
                    
                    inscribed_radii = np.minimum(np.array([density_field.distance(fan_vert) for fan_vert in (fan_verts@invertion.T + origo)]), self.max_size/I_D_COEFF)
                    radii = inscribed_radii * I_R_COEFF
                    dists = np.sqrt(np.sum((np.vstack((fan_verts, main_triangle_verts[i])) - np.vstack((main_triangle_verts[i-1], fan_verts)))**2, axis=1))
                    mean_radii = (np.r_[main_triangle_radii[i-1], radii] + np.r_[radii, main_triangle_radii[i]])/2
                    forces = 1 - (dists/mean_radii)
                    force_diffs = (forces[1:] - forces[:-1])
                    angles += force_diffs * 0.01
                    
                    fan_verts = corner + np.array([np.cos(angles), np.sin(angles), np.zeros(angles.shape)]).T * r 
                   
                fan_vert_ids = self.add_vertices(fan_verts@invertion.T + origo, constrained_radius=True)
                
                inner_loop_vert_ids = inner_loop_vert_ids + fan_vert_ids.tolist()
                
                test_id_start = self.triangles.shape[0]

                self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], fan_vert_ids[0], main_triangle_vert_ids[i-1]]))
                self.padding_triangles = np.r_[self.padding_triangles, True]

                for j in range(1, fan_vert_ids.shape[0]):
                    self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], fan_vert_ids[j], fan_vert_ids[j-1]]))
                    self.padding_triangles = np.r_[self.padding_triangles, True]

                self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], main_triangle_vert_ids[i], fan_vert_ids[-1]]))
                self.padding_triangles = np.r_[self.padding_triangles, True]
                
                self.vertex_radii[fan_vert_ids] = radii 
                
                
        
            else:
                self.triangles = np.vstack((self.triangles, [loop_vert_ids[i], main_triangle_vert_ids[i], main_triangle_vert_ids[i-1]]))
                dist = np.sum((self.vertex_coords[main_triangle_vert_ids[i]] - self.vertex_coords[main_triangle_vert_ids[i-1]])**2)**0.5
                self.vertex_radii[main_triangle_vert_ids[i]] = max(self.vertex_radii[main_triangle_vert_ids[i]], dist * D_R_COEFF)
                self.vertex_radii[main_triangle_vert_ids[i-1]] = max(self.vertex_radii[main_triangle_vert_ids[i-1]], dist * D_R_COEFF)
                self.padding_triangles = np.r_[self.padding_triangles, True]              

                
        inner_loop_vert_ids = np.array(inner_loop_vert_ids)

        self.inner_loops[face_id] = inner_loop_vert_ids

        return
        
    def fill_inner_loops(self):
        for i, loop in enumerate(self.inner_loops):
            if loop is None:
                continue

            constraint_triangles = self.get_constraint_trignales(i)
            density_field = InscribedSphereField(self.face_normals[i], constraint_triangles)

            
            all_inner_vertices, all_inner_triangles = get_filled_points(self.vertex_coords[loop], self.face_normals[i], constraint_triangles, self.max_size)
            new_vertices = all_inner_vertices[loop.shape[0]:]
            
            new_vertex_ids = self.add_vertices(new_vertices, constrained_radius=False)
            # These coefficients should be wrong
            new_vertex_radii = np.minimum(np.array([density_field.distance(vert) for vert in new_vertices]) * I_R_COEFF, self.max_size * D_R_COEFF) 
    
            self.vertex_radii[new_vertex_ids] = new_vertex_radii
            
            all_inner_triangles = all_inner_triangles.reshape(-1,3) - 1 # Gmsh starts indexing at 1
            inner_vertex_ids = np.r_[loop, new_vertex_ids]

            new_triangles = inner_vertex_ids[all_inner_triangles]

            self.triangles = np.vstack((self.triangles, new_triangles))
            self.padding_triangles = np.r_[self.padding_triangles, np.zeros(new_triangles.shape[0], dtype=bool)]



