import argparse
import numpy as np
import pebi_gmsh.generate_constrained_mesh as gcm


CLI = argparse.ArgumentParser()

CLI.add_argument(
    "--h0",
    type=float,
    default=0.1
)

CLI.add_argument(
    "--boundary",
    nargs="*",
    type=float,
    default=[0,0,1,1]
)
CLI.add_argument(
    "--fcpoints",
    nargs="*",
    type=float,
    default=[]
)
CLI.add_argument(
    "--fcres",
    type=float,
    default=1
)

CLI.add_argument(
    "--ccpoints",
    nargs="*",
    type=float,
    default=[]
)
CLI.add_argument(
    "--ccres",
    type=float,
    default=1
)

CLI.add_argument(
    "--popup",
    action='store_true'
)

if __name__ == "__main__":
    args = CLI.parse_args()
    
    popup = args.popup
    bounds = np.array(args.boundary).reshape((-1,2))
    fcpoints = np.array(args.fcpoints).reshape((-1,2))
    ccpoints = np.array(args.ccpoints).reshape((-1,2))
    print(bounds) 
    
    mesh_dict = gcm.generate_constrained_mesh_2d(
        h0 = args.h0,
        bounding_polygon = bounds,
        FC_points = fcpoints,
        CC_points = ccpoints,
        FC_resolution_factor = args.fcres,
        CC_resolution_factor = args.ccres,
        popup=popup
    )