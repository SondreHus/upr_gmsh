import struct
import sys
import math
import numpy as np
from pebi_gmsh.utils_3D.plane_densityfield import InscribedSphereField


field_coeff = float(sys.argv[1])
data_path = sys.argv[2]
data = np.load(data_path)
normal = data[:3]
tris = data[3:].reshape(-1,3,3)

field = InscribedSphereField(normal, tris) 
i = 1
while(True):
    xyz = np.array(struct.unpack("ddd", sys.stdin.buffer.read(24)))
    if math.isnan(xyz[0]):
        break
    f = field.distance(xyz) * field_coeff
    sys.stdout.buffer.write(struct.pack("d",f))
    sys.stdout.flush()
    i += 1


# Since the subroutine initiated by gmsh does not allow pprinting to terminal, comments were written to a txt file for debugging

# if i % 100 == 0:
#     with open("test_comments.txt", "a") as f:
#         f.write("program run {} times\n".format(i))

# with open("test_comments.txt", "a") as f:
#     f.write("file started\n")

# with open("test_comments.txt", "a") as f:
#     f.write("program run {} times\n".format(i))