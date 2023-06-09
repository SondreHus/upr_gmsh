import gmsh 
import numpy as np
from pebi_gmsh.utils_3D.tri_mesh import inside_mesh

def add_background_sites(surface_coords, surface_edges, surface_tris,  surface_faces, constrained_sites, radii):

    gmsh.initialize()
    gmsh.model.add("volume_filler")
    
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    sites_inside = inside_mesh(constrained_sites, surface_coords, surface_tris)
    
    surface_point_tags = []
    for point in surface_coords:
        surface_point_tags.append(gmsh.model.geo.add_point(
            point[0], point[1], point[2]
        ))

    constrained_site_tags = []
    for site in constrained_sites[sites_inside]:
        constrained_site_tags.append(gmsh.model.geo.add_point(
            site[0], site[1], site[2]
        ))


    
    final_site_tag = constrained_site_tags[-1]
    
    surface_edge_tags = []
    for edge in surface_edges:
        surface_edge_tags.append(gmsh.model.geo.add_line(
            surface_point_tags[edge[0]],
            surface_point_tags[edge[1]],
        ))
    surface_edge_tags = np.array(surface_edge_tags)
    
    surface_face_tags = []
    for face in surface_faces:
        curve_loop_tag = gmsh.model.geo.add_curve_loop(surface_edge_tags[face], reorient=True)
        surface_face_tags.append(gmsh.model.geo.add_plane_surface([curve_loop_tag]))

    
    surface_loop_tag = gmsh.model.geo.add_surface_loop(surface_face_tags)
    volume_tag = gmsh.model.geo.add_volume([surface_loop_tag])

    gmsh.model.geo.synchronize()

    field_id = gmsh.model.mesh.field.add("Distance")

    gmsh.model.mesh.field.set_numbers(field_id, "PointsList", constrained_site_tags)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_id)

    gmsh.model.mesh.embed(0, constrained_site_tags, 3, volume_tag)

    gmsh.model.mesh.generate(3)

    gmsh.model.mesh.create_faces()

    _, node_coords, *_ = gmsh.model.mesh.get_nodes()
    # gmsh.fltk.run()
    gmsh.finalize()

    node_coords = node_coords.reshape(-1,3)[final_site_tag:]

    remove = np.zeros(node_coords.shape[0], dtype=bool)

    for vertex, radius in zip(surface_coords, radii):
        dists = np.sqrt(np.sum((node_coords - vertex)**2, axis=1))
        remove = np.logical_or(remove,  dists <= radius)
    

    return np.delete(node_coords, remove, axis=0)
