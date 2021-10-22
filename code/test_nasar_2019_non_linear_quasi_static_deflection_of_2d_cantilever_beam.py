from pysph.solver.utils import iter_output, load
import os
from matplotlib import pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join

mypath = 'nasar_2019_non_linear_quasi_static_deflection_of_2d_cantilever_beam_output'
files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('hdf5')]

data = load(files[-1])
arrays = data['arrays']
pa = arrays['plate']

d_idx = 59
p = pa.cnt_limits[2 * d_idx]
q = pa.cnt_limits[2 * d_idx + 1]

for i in range(p, q):
    sidx = pa.cnt_idxs[i]
    if sidx == 60:
        print(sidx)

        print(pa.bc_normal_x0[i])
        print(pa.bc_normal_y0[i])

        print(pa.bc_normal_x[i])
        print(pa.bc_normal_y[i])

        print("norm", (pa.bc_normal_x0[i]**2. + pa.bc_normal_y0[i]**2.)**0.5)
        print("norm", (pa.bc_normal_x[i]**2. + pa.bc_normal_y[i]**2.)**0.5)

        print(pa.bc_normal_contact_x0[i])
        print(pa.bc_normal_contact_y0[i])

        print(pa.bc_normal_contact_x[i])
        print(pa.bc_normal_contact_y[i])

        print("norm", (pa.bc_normal_contact_x0[i]**2. + pa.bc_normal_contact_y0[i]**2.)**0.5)
        print("norm", (pa.bc_normal_contact_x[i]**2. + pa.bc_normal_contact_y[i]**2.)**0.5)

        print("bond B1", pa.bc_B1[i])
        print("bond B2", pa.bc_B2[i])
        print("bond B3", pa.bc_B3[i])

        print("Bond length", pa.bc_l0[i])

# print("total contacts", pa.tot_cnts[59])
# print(pa.cnt_idxs[59*8:60*8])
