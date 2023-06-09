import gmsh
import numpy as np
from pebi_gmsh.utils_3D.plane_densityfield import triangle_inscribed_circle_field, InscribedSphereField
from pebi_gmsh.constraints_3D.triangulated_surface import TriangulatedSurface
from pebi_gmsh.constraints_3D.constrained_edges import ConstrainedEdgeCollection
import os
from plotly import graph_objects as go
from pebi_gmsh.plotting_utils.plot_voronoi_3d import plot_trimesh, plot_3d_points, plot_voronoi_3d, inside_mesh
from scipy.spatial import Voronoi
from plotly import graph_objects as go


def get_sphere_points(center, radius, theta_res = 100, tau_res = 50):
    theta = np.linspace(0,2*np.pi, theta_res, endpoint=True)
    tau = np.linspace(0, np.pi, tau_res, endpoint=True)

    theta, tau = np.meshgrid(theta, tau)
    z = radius*np.cos(tau) + center[2]
    r = radius*np.sin(tau)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return (x,y,z)


angle = np.pi/12
density = 0.1
verts = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [1,1,np.sin(angle)],
    [0,1,np.sin(angle)],
])

CEC = ConstrainedEdgeCollection()
CEC.set_max_size(density)
CEC.add_vertices(verts)
CEC.add_face([0,1,2,3])
CEC.add_face([0,1,4,5])
CEC.add_face([2,3,5,4])

CEC.calculate_edge_vertex_radii()
CEC.populate_edge_vertices()
CEC.construct_face_padding(0)
CEC.set_max_size(1)

CEC.fill_inner_loops()


plot_trimesh(CEC.vertex_coords, CEC.triangles)


frame_size = 0.3
x_start = 0.5-frame_size/2
x_stop = 0.5+frame_size/2
x = np.zeros(0)
y = np.zeros(0)
for tri in CEC.triangles:
    x = np.r_[x, CEC.vertex_coords[tri, 0], CEC.vertex_coords[tri[0], 0], None]
    y = np.r_[y, CEC.vertex_coords[tri, 1], CEC.vertex_coords[tri[0], 1], None]

data = []
data.append(go.Scatter(
    x = x,
    y = y,
    mode="lines",
    line=dict(
        color='black',
        width=1.5,
        # size=10,
        # dash="dash"
    ),
))
# fig.show()
# fig.write_image("./fan_dens_{}_angle_{}.svg".format(density,angle))

constraint_tris = np.array([
    [0,1,4],
    [0,4,5],
])

field = InscribedSphereField(np.array([0,0,1]), verts[constraint_tris])

corner_pos = np.array([
    [x_start, 0, 0],
    [x_stop, 0, 0],
    [x_stop, frame_size, 0],
    [x_start, frame_size, 0],
])

corner_vals = np.array([field.distance(pos) for pos in corner_pos]).reshape(2,2)

data.append(go.Heatmap(
    x = [x_start, x_stop],
    y = [0, frame_size],
    z = corner_vals,
    zmin = 0,
    zmax = 0.15,
    zsmooth='best'
))

fig = go.Figure(
    data=data,
    layout=dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width=1000,
        height = 850,
        xaxis=dict(showgrid=False, visible=False, range=[x_start, x_stop]),
        yaxis=dict(showgrid=False, visible=False, range=[0, frame_size]),
        plot_bgcolor='white',
    )
)
fig.show()
fig.write_image("./diff_angle_fan_dens_{}_angle_{}.svg".format(density,angle))

# fig = go.Figure(
#     data=go.Heatmap(
#         x = [x_start, x_stop],
#         y = [0, frame_size],
#         z = corner_vals,
#         zmin = 0,
#         zmax = 0.1,
#         zsmooth='best'
#     ),
#     layout=dict(
#         margin=dict(l=0,r=0,b=0,t=0),
#         width=1000,
#         height = 850,
#         xaxis=dict(showgrid=False, visible=False, range=[x_start, x_stop]),
#         yaxis=dict(showgrid=False, visible=False, range=[0, frame_size]),
#         plot_bgcolor='white',
#     )
# )
# fig.show()
# fig.write_image("./field_angle_{}.svg".format(angle))


center_dist = np.sum((CEC.vertex_coords - np.array([[0.5,0,0]]))**2, axis=1)
closest = np.argmin(center_dist)

neighbours = []

for tri in CEC.triangles:
    if closest in tri:
        for vert_id in tri:
            if (vert_id not in neighbours) and vert_id != closest and CEC.vertex_coords[vert_id,1] != 0:
                neighbours.append(vert_id)

data = []
sphere_points = get_sphere_points(CEC.vertex_coords[closest], CEC.vertex_radii[closest])
data.append(go.Surface(
    x = sphere_points[0],
    y = sphere_points[1],
    z = sphere_points[2],

    colorscale=[[0, "red"], [1, "black"]],
    surfacecolor = np.zeros(sphere_points[0].shape),
    # color = "cyan",
    opacity = 0.2,
    showscale=False,
))

for vert_id in neighbours:
    sphere_points = get_sphere_points(CEC.vertex_coords[vert_id], CEC.vertex_radii[vert_id])
    data.append(go.Surface(
        x = sphere_points[0],
        y = sphere_points[1],
        z = sphere_points[2],

        colorscale=[[0, "blue"], [1, "black"]],
        surfacecolor = np.zeros(sphere_points[0].shape),
        # color = "cyan",
        opacity = 0.2,
        showscale=False,
    ))

for vert_id in range(CEC.vertex_coords.shape[0]):
    if vert_id in neighbours:
        continue
    coords = CEC.vertex_coords[vert_id]
    if coords[0] < x_start or coords[0] > x_stop or coords[1] > frame_size or coords[1] == 0:
        continue
    
    radius = field.distance(coords) * np.sqrt(5)/3
    sphere_points = get_sphere_points(coords, radius)
    data.append(go.Surface(
        x = sphere_points[0],
        y = sphere_points[1],
        z = sphere_points[2],

        colorscale=[[0, "green"], [1, "black"]],
        surfacecolor = np.zeros(sphere_points[0].shape),
        # color = "cyan",
        opacity = 0.2,
        showscale=False,
    ))

data = plot_trimesh(CEC.vertex_coords, CEC.triangles, data=data)
camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0, z=1),
    projection = dict(type="orthographic")
)
center_coords = CEC.vertex_coords[closest]
fig = go.Figure(data=data,
    layout = dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width=1000,
        height = 850,
        scene_camera = camera,
        scene = dict(
            xaxis = dict(visible=False, range=[center_coords[0]-0.1, center_coords[0]+0.1]),
            yaxis = dict(visible=False, range=[0, 0.2]),
            zaxis = dict(visible=False, range=[-.1, 0.1]),
            aspectratio=dict(x=2, y=2, z=2)
            # aspectmode = "cube"
        )
    )                
)
 # fig1.update_layout(scene = {
    #     "xaxis": {"range": [bounding_box[0] - padding, bounding_box[1] + padding]},
    #     "yaxis": {"range": [bounding_box[2] - padding, bounding_box[3] + padding]},
    #     "zaxis": {"range": [bounding_box[4] - padding, bounding_box[5] + padding]},
    #     "aspectmode": 'cube'
    # })
fig.show()
fig.write_image("fan_spheres.svg")