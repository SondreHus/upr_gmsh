import struct
import sys
import math
import numpy as np
from pebi_gmsh.utils_3D.densityfield import InscribedCircleField, InscribedCircleTriangle, distance


    
data_path = sys.argv[1]
data = np.load(data_path)
normal = data[:3]
tris = data[3:].reshape(-1,3,3)
# n_o = input[0]
# p_tri = input[1:,:]
fields = [InscribedCircleTriangle(normal, tri) for tri in tris]
# field = InscribedCircleField(normal, tris)
with open("test_comments.txt", "a") as f:
    f.write("file started\n")
i = 1
while(True):
    xyz = np.array(struct.unpack("ddd", sys.stdin.buffer.read(24)))
    if i % 100 == 0:
        with open("test_comments.txt", "a") as f:
            f.write("program run {} times\n".format(i))
    if math.isnan(xyz[0]):
        break
    f = distance(xyz, fields)
    sys.stdout.buffer.write(struct.pack("d",f))
    sys.stdout.flush()
    i += 1
with open("test_comments.txt", "a") as f:
    f.write("program run {} times\n".format(i))