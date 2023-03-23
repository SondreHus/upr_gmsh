import gmsh
import numpy as np
from pebi_gmsh.convert_GMSH import convert_GMSH
from pebi_gmsh.site_data import (SiteData)
from typing import List
def generate_constrained_mesh_2d(site_data: SiteData , h0 = 0.1, bounding_polygon = np.array([[0,0],[1,1]]), popup = False, algorithm = 5):

    # Initialize gmesh and set up the model
    gmsh.initialize()
    gmsh.model.add("MRST")
    gmsh.option.setNumber("Mesh.Algorithm", algorithm)
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    bounding_polygon = np.array(bounding_polygon)

    # Add bounding box / polygon 
    if bounding_polygon.shape[0] == 2:
        
        gmsh.model.geo.addPoint(bounding_polygon[0,0], bounding_polygon[0,1], 0, h0, 0)
        gmsh.model.geo.addPoint(bounding_polygon[0,0], bounding_polygon[1,1], 0, h0, 1)
        gmsh.model.geo.addPoint(bounding_polygon[1,0], bounding_polygon[1,1], 0, h0, 2)
        gmsh.model.geo.addPoint(bounding_polygon[1,0], bounding_polygon[0,1], 0, h0, 3)

        gmsh.model.geo.addLine(0,1,0)
        gmsh.model.geo.addLine(1,2,1)
        gmsh.model.geo.addLine(2,3,2)
        gmsh.model.geo.addLine(3,0,3)

        gmsh.model.geo.addCurveLoop([0,1,2,3], 1)
    else:
        p_tags = []
        for node in bounding_polygon:
            p_tags.append(gmsh.model.geo.addPoint(node[0], node[1], 0, h0))
        e_tags = []
        for i in range(len(p_tags)):
            e_tags.append(gmsh.model.geo.addLine(p_tags[i], p_tags[(i+1)%len(p_tags)]))
        gmsh.model.geo.addCurveLoop(e_tags, 1)

    
    
    gmsh.model.geo.addPlaneSurface([1],1)
    
    
    sites = site_data.sites
    edges = site_data.edges 
    
    constraint_site_idx = []
    for site in sites:
        site_point = gmsh.model.geo.add_point(site[0],site[1],0, 1)
        constraint_site_idx.append(site_point)
    
    constraint_edge_idx = []
    for edge in edges:
        edge_id = gmsh.model.geo.addLine(constraint_site_idx[edge[0]], constraint_site_idx[edge[1]])
        constraint_edge_idx.append(edge_id)


    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1,constraint_edge_idx, 2, 1)
    extend = gmsh.model.mesh.field.add("Extend")
    gmsh.model.mesh.field.set_numbers(extend, "CurvesList", constraint_edge_idx + [0,1,2,3])
    gmsh.model.mesh.field.set_number(extend, "DistMax", 0.5)
    gmsh.model.mesh.field.set_number(extend, "SizeMax", h0)

    constant = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.set_string(constant, "F", str(h0))
    
    min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.set_numbers(min, "FieldsList", [constant, extend])


    gmsh.model.mesh.field.setAsBackgroundMesh(min)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.create_faces()
   

    if popup:
        gmsh.fltk.run()
    
    mesh_dict = convert_GMSH()

    gmsh.fltk.finalize()
    return mesh_dict

    