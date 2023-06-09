import numpy as np

def point_to_plane(point, plane_normals, plane_d):
    dists = np.sum(point.reshape(1,3)*plane_normals, axis=1) + plane_d
    projection = point.reshape(1,3) - dists.reshape(-1,1)*plane_normals

    return dists, projection

def point_to_line(point, line_dirs, line_origins):
    diff = point.reshape(1,3)-line_origins - line_dirs * (np.sum((point.reshape(1,3) - line_origins)*line_dirs, axis=1)).reshape(-1,1)

    return np.linalg.norm(diff, axis=1), point.reshape(-1,3) - diff


def point_inscribed_distance(point, triangles, edges, points):

    # Triangles

    tri_normals = np.cross(triangles[:,1] - triangles[:,0], triangles[:,2] - triangles[:,0])
    tri_normals = tri_normals/np.linalg.norm(tri_normals, axis=1).reshape(-1,1)

    tri_plane_d = np.sum(tri_normals * triangles[:,0], axis=1)

    tri_border_normals = np.cross(np.tile(tri_normals, 3).reshape(-1,3), (np.roll(triangles, -1, axis=1) - triangles).reshape(-1,3))
    tri_border_normals = (tri_border_normals/np.linalg.norm(tri_border_normals, axis=1).reshape(-1,1))

    tri_border_ds = np.sum(tri_border_normals*triangles.reshape(-1,3), axis=1)

    tri_dists, projections = point_to_plane(point, tri_normals, tri_plane_d)

    inside_tri = np.all((np.sum(np.tile(projections, 3).reshape(-1,3)*tri_border_normals, axis=1) - tri_border_ds >= 0).reshape(-1,3), axis=1)
    
    min_plane_dist = np.min(np.abs(tri_dists[inside_tri]))

    # Edges

    edge_dirs = edges[:,1] - edges[:,0]
    edge_dirs = edge_dirs/np.linalg.norm(edge_dirs,axis=1).reshape(-1,1)

    edge_starts = np.sum(edge_dirs*edges[:,0], axis=1)
    edge_ends = np.sum(edge_dirs*edges[:,0], axis=1)

    line_dists, projections = point_to_line(point, edge_dirs, edges[:,0])

    line_projection = np.sum(projections*edge_dirs, axis=1)

    inside_line = np.logical_and(line_projection >= edge_starts, line_projection <= edge_ends)
    
    min_edge_dist = np.min(np.abs(line_dists[inside_line]))

    # Points

    min_point_dist = np.min(np.linalg.norm(point.reshape(1,3) - points, axis=1))
    return np.min((min_plane_dist, min_edge_dist, min_point_dist))

    