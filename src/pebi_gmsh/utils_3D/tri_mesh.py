import trimesh
import mapbox_earcut as earcut
import numpy as np

def triangulate_polygon(points):
    rings = [points.shape[0]]

    return earcut.triangulate_float32(points, rings).reshape(-1,3)

def inside_mesh(points, mesh_verts, mesh_faces):
    mesh = trimesh.Trimesh(mesh_verts, faces=np.array(mesh_faces))
    return mesh.contains(points)