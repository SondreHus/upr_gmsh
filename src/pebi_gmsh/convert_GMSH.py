import gmsh
def convert_GMSH():
    if not gmsh.is_initialized():
        raise Exception("GMSH not initialized")
    
    mesh_dict = {}
    node_ids, node_coords, *_ = gmsh.model.mesh.get_nodes()
    tri_ids, tri_nodes, *_ = gmsh.model.mesh.get_all_faces(3)
    
    mesh_dict["node_ids"] = node_ids
    mesh_dict["node_coords"] = node_coords
    mesh_dict["tri_ids"] = tri_ids
    mesh_dict["tri_nodes"] = tri_nodes

    return mesh_dict